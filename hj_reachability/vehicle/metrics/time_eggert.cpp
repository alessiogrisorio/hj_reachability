/*
 * Signed time to the rolling Eggert-risk boundary.
 * Source model: hj_reachability/vehicle/metrics/eggert.py, thesis branch,
 * commit 39abc291094f1193f08fb786943530ff3ab7324e.
 *
 * Build on Linux, next to eggert.py and time_eggert.py:
 *   g++ -O3 -march=native -std=c++17 -fopenmp -fPIC -shared \
 *       time_eggert.cpp -o _time_eggert.so
 * Without OpenMP, omit -fopenmp (serial fallback). Do not use -ffast-math.
 * No pybind11, Python headers, or third-party C++ libraries are required.
 *
 * R(x) = Eggert probability over [0, horizon], including absorption of
 * surviving probability at first geometric contact. The probability is
 * RESET at every evaluation of R(Phi_t(x)); this is a rolling risk, not
 * the cumulative probability measured from the original state.
 *
 * For R(x) < p_crit: search forward for the first R(Phi_t(x)) >= p_crit.
 * For R(x) > p_crit: search backward for the first R(Phi_-t(x)) <= p_crit.
 * For R(x) == p_crit: return zero. No crossing -> saturate at +/-time_max.
 * No smoothing, geometric sign override, or probability-threshold offset.
 *
 * Fast mode computes R at grid nodes and uses trilinear interpolation in
 * (x_rel, y_rel, theta_rel). Speeds and steering stay at their original
 * grid nodes. Theta is periodic. Outside the XY domain, R is evaluated
 * directly: no extrapolation, clamping, or trajectory termination.
 * Direct mode recomputes R at every search point, for validation.
 *
 * Numerical limits: the fast result approximates the rolling-risk field.
 * Time crossings are sampled every search_dt/2 before bisection; crossings
 * between samples can be missed. General curved geometric contact uses
 * the SAME midpoint/endpoint sampling as eggert.py, with its limitations.
 * time_tolerance controls a detected bracket, not a global error bound.
 * A risk jump is located as a first entry even when equality is skipped.
 *
 * C ABI (used by the companion Python wrapper):
 *  te_default_config, te_config_size, te_abi_version, te_last_error,
 *  te_compute_grid, te_probability_states, te_time_states.
 * All arrays: native float64, C-contiguous; shape: uint64[6].
 * Grid axes are concatenated [x, y, theta, v_H, delta_E, v_E].
 * Grid output order: NumPy C order, shape (nx,ny,nt,nvh,nd,nve).
 * State batches: shape (n,6), same state order.
 * Outputs must have enough storage and must not alias inputs/each other.
 * Optional contact-time output can be nullptr; no contact gives +infinity.
 * Return 0 on success, -1 on error; te_last_error() describes the error.
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(_WIN32)
#define TE_EXPORT __declspec(dllexport)
#else
#define TE_EXPORT __attribute__((visibility("default")))
#endif

extern "C" {
struct TEConfig {
    double horizon, time_max, dt, search_dt, time_tolerance;
    double beta_d, tau_d0, escape_rate, p_crit;
    double collision_tolerance, contact_time_tolerance;
    double lf, lr, ego_length, ego_width, human_length, human_width;
    double theta_period;
    std::int32_t threads, interpolate, use_symmetry;
};
}

namespace te {
constexpr double pi = 3.141592653589793238462643383279502884;
constexpr double infinity = std::numeric_limits<double>::infinity();
thread_local std::string error;

void require(bool condition, const char* message) {
    if (!condition) throw std::invalid_argument(message);
}

int threads(const TEConfig& p) {
#ifdef _OPENMP
    return p.threads > 0 ? p.threads : omp_get_max_threads();
#else
    (void)p;
    return 1;
#endif
}

void validate(const TEConfig& p) {
    const double positive[] = {p.horizon, p.time_max, p.dt, p.search_dt,
        p.time_tolerance, p.beta_d, p.tau_d0, p.contact_time_tolerance,
        p.lf, p.lr, p.ego_length, p.ego_width, p.human_length,
        p.human_width, p.theta_period};
    for (double v : positive)
        require(std::isfinite(v) && v > 0, "Positive parameters must be finite and > 0");
    require(p.dt <= p.horizon, "dt must not exceed horizon");
    require(p.search_dt <= p.time_max, "search_dt must not exceed time_max");
    require(std::isfinite(p.escape_rate) && p.escape_rate >= 0,
            "escape_rate must be finite and >= 0");
    require(std::isfinite(p.collision_tolerance) && p.collision_tolerance >= 0,
            "collision_tolerance must be finite and >= 0");
    require(std::isfinite(p.p_crit) && p.p_crit > 0 && p.p_crit < 1,
            "p_crit must be strictly between 0 and 1");
    require(std::abs(p.theta_period - 2 * pi) < 1e-12,
            "theta_period must be 2*pi for vehicle headings");
    require(p.threads >= 0, "threads must be >= 0");
    require(p.interpolate == 0 || p.interpolate == 1, "interpolate must be 0 or 1");
    require(p.use_symmetry == 0 || p.use_symmetry == 1, "use_symmetry must be 0 or 1");
    require(std::ceil(p.horizon / p.dt) <= 1000000 &&
            std::ceil(p.time_max / p.search_dt) <= 1000000,
            "Too many integration/search steps");
    require(p.contact_time_tolerance >= p.dt * 1e-15,
            "contact_time_tolerance is below usable double precision");
    require(p.time_tolerance >= p.time_max * 1e-15,
            "time_tolerance is below usable double precision");
}

struct Pose { double x, y, theta; };

struct Motion {
    double theta, vh, delta, ve, beta, omega, ct, st;
    bool aligned;
    Motion(double th, double h, double d, double e, const TEConfig& p)
        : theta(th), vh(h), delta(d), ve(e) {
        const double tangent = std::tan(delta);
        beta = std::atan(p.lr / (p.lf + p.lr) * tangent);
        omega = ve * std::cos(beta) * tangent / (p.lf + p.lr);
        ct = std::cos(theta); st = std::sin(theta);
        aligned = delta == 0 && std::abs(std::atan2(st, ct)) < 1e-12;
    }
};

// Affine XY flow shared by all positions with the same nominal kinematics.
struct Frame {
    double c, s, ox, oy, theta, ct, st;
    double bx, by, bu, bv;
    Frame(const Motion& m, double t, const TEConfig& p) {
        const double angle = m.omega * t, half = angle / 2;
        const double sinc = std::abs(half) < 1e-8
            ? 1 - half * half / 6 : std::sin(half) / half;
        const double travel = m.ve * t * sinc;
        const double dx = m.vh * t * m.ct - travel * std::cos(m.beta + half);
        const double dy = m.vh * t * m.st - travel * std::sin(m.beta + half);
        c = std::cos(angle); s = std::sin(angle);
        ox = c * dx + s * dy; oy = -s * dx + c * dy;
        theta = m.theta - angle;
        ct = std::cos(theta); st = std::sin(theta);
        const double ac = std::abs(ct), as = std::abs(st);
        const double a = p.ego_length / 2, b = p.ego_width / 2;
        const double h = p.human_length / 2, w = p.human_width / 2;
        bx = a + h * ac + w * as; by = b + h * as + w * ac;
        bu = h + a * ac + b * as; bv = w + a * as + b * ac;
    }
    Pose pose(double x, double y) const {
        return {c * x + s * y + ox, -s * x + c * y + oy, theta};
    }
    double margin(double x, double y) const {
        return std::max({std::abs(x) - bx, std::abs(y) - by,
            std::abs(ct * x + st * y) - bu,
            std::abs(-st * x + ct * y) - bv});
    }
    double distance(double x, double y, const TEConfig& p) const {
        if (margin(x, y) <= 0) return 0;
        const double a = p.ego_length / 2, b = p.ego_width / 2;
        const double h = p.human_length / 2, w = p.human_width / 2;
        double best = infinity;
        for (int i = 0; i < 4; ++i) {
            const double hx = (i & 1) ? h : -h, hy = (i & 2) ? w : -w;
            const double dx = std::max(std::abs(x + ct * hx - st * hy) - a, 0.0);
            const double dy = std::max(std::abs(y + st * hx + ct * hy) - b, 0.0);
            best = std::min(best, dx * dx + dy * dy);
            const double ex = ((i & 1) ? a : -a) - x;
            const double ey = ((i & 2) ? b : -b) - y;
            const double ux = std::max(std::abs(ct * ex + st * ey) - h, 0.0);
            const double uy = std::max(std::abs(-st * ex + ct * ey) - w, 0.0);
            best = std::min(best, ux * ux + uy * uy);
        }
        return std::sqrt(best);
    }
};

struct Interval {
    double start, end;
    Frame middle, last;
    Interval(const Motion& m, double a, double b, const TEConfig& p)
        : start(a), end(b), middle(m, (a + b) / 2, p), last(m, b, p) {}
};

std::vector<Interval> make_plan(const Motion& m, const TEConfig& p) {
    const auto n = static_cast<std::size_t>(std::ceil(p.horizon / p.dt));
    std::vector<Interval> result;
    result.reserve(n);
    for (std::size_t i = 0; i < n; ++i)
        result.emplace_back(m, i * p.dt, std::min((i + 1) * p.dt, p.horizon), p);
    return result;
}

struct Risk { double probability, contact; };

Risk risk(double x, double y, const Motion& m, const TEConfig& p,
          const std::vector<Interval>* plan = nullptr) {
    const Frame initial(m, 0, p);
    if (initial.margin(x, y) <= p.collision_tolerance) return {1, 0};
    const double a = (p.ego_length + p.human_length) / 2;
    const double b = (p.ego_width + p.human_width) / 2;
    double scheduled = infinity;
    if (m.aligned && m.vh != m.ve && std::abs(y) <= b + p.collision_tolerance) {
        const double limit = a + p.collision_tolerance;
        const double t1 = (-limit - x) / (m.vh - m.ve);
        const double t2 = (limit - x) / (m.vh - m.ve);
        const double entry = std::max(std::min(t1, t2), 0.0);
        if (std::max(t1, t2) >= entry && entry <= p.horizon) scheduled = entry;
    }
    double probability = 0, survival = 1;
    const auto steps = static_cast<std::size_t>(std::ceil(p.horizon / p.dt));
    const int refinements = std::max(0, static_cast<int>(
        std::ceil(std::log2(p.dt / p.contact_time_tolerance))));
    for (std::size_t i = 0; i < steps; ++i) {
        const double start = i * p.dt, end = std::min((i + 1) * p.dt, p.horizon);
        double stop = end, distance;
        bool hit;
        if (m.aligned) {
            hit = scheduled <= end;
            stop = std::min(scheduled, end);
            const double t = start + std::max(stop - start, 0.0) / 2;
            distance = std::hypot(std::max(std::abs(x + (m.vh - m.ve) * t) - a, 0.0),
                                  std::max(std::abs(y) - b, 0.0));
        } else {
            // The group plan eliminates trig evaluations inside the XY loop.
            const Frame middle = plan ? (*plan)[i].middle : Frame(m, (start + end) / 2, p);
            const Pose mid = middle.pose(x, y);
            const bool mid_hit = middle.margin(mid.x, mid.y) <= p.collision_tolerance;
            bool end_hit = false;
            if (!mid_hit) {
                const Frame last = plan ? (*plan)[i].last : Frame(m, end, p);
                const Pose pos = last.pose(x, y);
                end_hit = last.margin(pos.x, pos.y) <= p.collision_tolerance;
            }
            hit = mid_hit || end_hit;
            if (hit) {
                double lo = mid_hit ? start : (start + end) / 2;
                double hi = mid_hit ? (start + end) / 2 : end;
                for (int k = 0; k < refinements; ++k) {
                    const double t = (lo + hi) / 2;
                    const Frame trial(m, t, p);
                    const Pose pos = trial.pose(x, y);
                    if (trial.margin(pos.x, pos.y) <= p.collision_tolerance) hi = t;
                    else lo = t;
                }
                stop = hi;
                const Frame short_mid(m, start + std::max(stop - start, 0.0) / 2, p);
                const Pose pos = short_mid.pose(x, y);
                distance = short_mid.distance(pos.x, pos.y, p);
            } else distance = middle.distance(mid.x, mid.y, p);
        }
        const double critical = std::exp(-p.beta_d * distance) / p.tau_d0;
        const double total = critical + p.escape_rate;
        const double width = std::max(stop - start, 0.0);
        const double fraction = -std::expm1(-total * width);
        probability += survival * (total > 0 ? critical / total : 0) * fraction;
        survival *= std::exp(-total * width);
        if (hit) return {std::clamp(probability + survival, 0.0, 1.0), stop};
    }
    return {std::clamp(probability, 0.0, 1.0), infinity};
}

template<class Evaluate>
double signed_time(double initial, const TEConfig& p, Evaluate evaluate) {
    if (initial == p.p_crit) return 0;
    const double direction = initial < p.p_crit ? 1 : -1;
    const auto entered = [&](double r) {
        return direction > 0 ? r >= p.p_crit : r <= p.p_crit;
    };
    double lower = 0;
    const auto steps = static_cast<std::size_t>(std::ceil(p.time_max / p.search_dt));
    for (std::size_t i = 0; i < steps; ++i) {
        const double end = std::min((i + 1) * p.search_dt, p.time_max);
        const double samples[2] = {(i * p.search_dt + end) / 2, end};
        for (std::size_t j = 0; j < 2; ++j) {
            const double t = samples[j];
            if (entered(evaluate(direction * t, 2 * i + j))) {
                double upper = t;
                while (upper - lower > p.time_tolerance) {
                    const double trial = (lower + upper) / 2;
                    if (entered(evaluate(direction * trial, std::numeric_limits<std::size_t>::max()))) upper = trial;
                    else lower = trial;
                }
                // The entered endpoint avoids a false zero for a nearby state.
                return direction * upper;
            }
            lower = t;
        }
    }
    return direction * p.time_max;
}

struct Bracket { std::size_t lo, hi; double w; };
struct Axis {
    const double* a;
    std::size_t n;
    double inverse;
    bool uniform;
    Axis(const double* data, std::size_t size) : a(data), n(size), inverse(0), uniform(false) {
        require(n > 0, "Grid axes must be nonempty");
        for (std::size_t i = 0; i < n; ++i) {
            require(std::isfinite(a[i]), "Grid axes must be finite");
            if (i) require(a[i] > a[i - 1], "Grid axes must be strictly increasing");
        }
        if (n > 1) {
            const double h = (a[n - 1] - a[0]) / (n - 1);
            inverse = 1 / h; uniform = true;
            for (std::size_t i = 1; i < n; ++i)
                if (std::abs(a[i] - (a[0] + i * h)) > 1e-5 * h) uniform = false;
        }
    }
    bool contains(double x) const { return x >= a[0] && x <= a[n - 1]; }
    Bracket bracket(double x) const {
        if (n == 1) return {0, 0, 0};
        std::size_t lo;
        if (uniform) {
            const double q = std::clamp((x - a[0]) * inverse, 0.0, double(n - 1));
            lo = std::min(static_cast<std::size_t>(q), n - 2);
            while (lo && a[lo] > x) --lo;
            while (lo + 1 < n - 1 && a[lo + 1] < x) ++lo;
        } else {
            const auto it = std::upper_bound(a, a + n, x);
            lo = it == a ? 0 : std::min(std::size_t(it - a - 1), n - 2);
        }
        return {lo, lo + 1, std::clamp((x - a[lo]) / (a[lo + 1] - a[lo]), 0.0, 1.0)};
    }
    Bracket angular(double theta, double period) const {
        if (n == 1) return {0, 0, 0};
        double t = std::fmod(theta - a[0], period);
        if (t < 0) t += period;
        t += a[0];
        if (t <= a[n - 1]) return bracket(t);
        const double gap = a[0] + period - a[n - 1];
        return gap > 0 ? Bracket{n - 1, 0, (t - a[n - 1]) / gap}
                       : Bracket{n - 1, n - 1, 0};
    }
};

struct Grid {
    std::vector<Axis> axes;
    std::array<std::size_t, 6> n;
    std::size_t total = 1, xy, fixed, groups;
    Grid(const double* data, const std::uint64_t* shape, const TEConfig& p) {
        require(data && shape, "Null grid pointer");
        for (int d = 0; d < 6; ++d) {
            require(shape[d] > 0 && shape[d] <= std::uint64_t(std::numeric_limits<std::size_t>::max()),
                    "Invalid grid shape");
            n[d] = static_cast<std::size_t>(shape[d]);
            require(n[d] <= std::size_t(std::numeric_limits<std::ptrdiff_t>::max()) / total,
                    "Grid shape overflow");
            axes.emplace_back(data, n[d]); data += n[d]; total *= n[d];
        }
        require(total <= std::numeric_limits<std::size_t>::max() / sizeof(double),
                "Grid byte-size overflow");
        require(axes[2].a[n[2] - 1] - axes[2].a[0] <= p.theta_period + 1e-6,
                "Theta axis spans more than one period");
        xy = n[0] * n[1]; fixed = n[3] * n[4] * n[5]; groups = n[2] * fixed;
    }
    Motion motion(std::size_t g, const TEConfig& p) const {
        const std::size_t t = g / fixed, f = g % fixed;
        const std::size_t h = f / (n[4] * n[5]);
        const std::size_t d = (f / n[5]) % n[4], e = f % n[5];
        return Motion(axes[2].a[t], axes[3].a[h], axes[4].a[d], axes[5].a[e], p);
    }
    std::size_t packed(std::size_t g, std::size_t pos) const {
        return ((g % fixed) * n[2] + g / fixed) * xy + pos;
    }
    std::size_t flat(std::size_t g, std::size_t pos) const { return pos * groups + g; }
};

std::vector<std::size_t> reflection(const Axis& a, bool angular = false) {
    std::vector<std::size_t> map(a.n), used(a.n, 0);
    for (std::size_t i = 0; i < a.n; ++i) {
        double best = infinity;
        for (std::size_t j = 0; j < a.n; ++j) {
            double gap = a.a[i] + a.a[j];
            if (angular) gap = std::atan2(std::sin(gap), std::cos(gap));
            if (std::abs(gap) < best) { best = std::abs(gap); map[i] = j; }
        }
        if (best > 1e-6 || used[map[i]]++) return {};
    }
    for (std::size_t i = 0; i < a.n; ++i) if (map[map[i]] != i) return {};
    return map;
}

double interpolate(const Grid& grid, const std::vector<double>& field,
                   std::size_t f, const Pose& pos, const TEConfig& p,
                   const Bracket* cached_theta = nullptr) {
    const Bracket x = grid.axes[0].bracket(pos.x), y = grid.axes[1].bracket(pos.y);
    const Bracket t = cached_theta ? *cached_theta : grid.axes[2].angular(pos.theta, p.theta_period);
    const auto at = [&](std::size_t it, std::size_t ix, std::size_t iy) {
        return field[(f * grid.n[2] + it) * grid.xy + ix * grid.n[1] + iy];
    };
    const auto lerp = [](double a, double b, double w) { return a + w * (b - a); };
    const auto plane = [&](std::size_t it) {
        return lerp(lerp(at(it, x.lo, y.lo), at(it, x.lo, y.hi), y.w),
                    lerp(at(it, x.hi, y.lo), at(it, x.hi, y.hi), y.w), x.w);
    };
    return std::clamp(lerp(plane(t.lo), plane(t.hi), t.w), 0.0, 1.0);
}

// Prevent exceptions from escaping an OpenMP worker or crossing the C ABI.
struct Failure {
    bool failed = false;
    std::string message;
    void capture() {
#ifdef _OPENMP
#pragma omp critical(te_failure)
#endif
        {
            if (!failed) {
                failed = true;
                try { throw; }
                catch (const std::exception& e) { message = e.what(); }
                catch (...) { message = "Unknown worker error"; }
            }
        }
    }
    void check() const { if (failed) throw std::runtime_error(message); }
};

void compute(const Grid& g, const TEConfig& p, double* probability,
             double* values, double* contact) {
    const int workers = threads(p);
    std::vector<double> field(p.interpolate ? g.total : 0);
    auto ry = p.use_symmetry ? reflection(g.axes[1]) : std::vector<std::size_t>{};
    auto rt = p.use_symmetry ? reflection(g.axes[2], true) : std::vector<std::size_t>{};
    auto rd = p.use_symmetry ? reflection(g.axes[4]) : std::vector<std::size_t>{};
    const bool symmetry = !ry.empty() && !rt.empty() && !rd.empty();
    Failure failure;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1) num_threads(workers)
#endif
    for (std::ptrdiff_t group = 0; group < static_cast<std::ptrdiff_t>(g.groups); ++group) {
        try {
            const auto gi = static_cast<std::size_t>(group);
            std::size_t mirror = gi;
            if (symmetry) {
                const std::size_t f = gi % g.fixed, h = f / (g.n[4] * g.n[5]);
                const std::size_t d = (f / g.n[5]) % g.n[4], e = f % g.n[5];
                mirror = rt[gi / g.fixed] * g.fixed + (h * g.n[4] + rd[d]) * g.n[5] + e;
                if (gi > mirror) continue;
            }
            const Motion m = g.motion(gi, p);
            const auto plan = make_plan(m, p);
            for (std::size_t ix = 0; ix < g.n[0]; ++ix) {
                for (std::size_t iy = 0; iy < g.n[1]; ++iy) {
                    if (symmetry && gi == mirror && iy > ry[iy]) continue;
                    const std::size_t pos = ix * g.n[1] + iy;
                    const Risk r = risk(g.axes[0].a[ix], g.axes[1].a[iy], m, p, &plan);
                    const auto put = [&](std::size_t group_index, std::size_t xy_index) {
                        probability[g.flat(group_index, xy_index)] = r.probability;
                        if (contact) contact[g.flat(group_index, xy_index)] = r.contact;
                        if (p.interpolate) field[g.packed(group_index, xy_index)] = r.probability;
                    };
                    put(gi, pos);
                    if (symmetry) put(mirror, ix * g.n[1] + ry[iy]);
                }
            }
        } catch (...) { failure.capture(); }
    }
    failure.check();

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1) num_threads(workers)
#endif
    for (std::ptrdiff_t group = 0; group < static_cast<std::ptrdiff_t>(g.groups); ++group) {
        try {
            const auto gi = static_cast<std::size_t>(group);
            const Motion m = g.motion(gi, p);
            std::size_t mirror = gi;
            if (symmetry) {
                const std::size_t f = gi % g.fixed, h = f / (g.n[4] * g.n[5]);
                const std::size_t d = (f / g.n[5]) % g.n[4], e = f % g.n[5];
                mirror = rt[gi / g.fixed] * g.fixed + (h * g.n[4] + rd[d]) * g.n[5] + e;
                if (gi > mirror) continue;
            }
            // Share trigonometry and angular interpolation indices across XY.
            std::vector<Frame> forward, backward;
            std::vector<Bracket> forward_theta, backward_theta;
            std::vector<Motion> forward_motion, backward_motion;
            const auto count = static_cast<std::size_t>(std::ceil(p.time_max / p.search_dt));
            std::vector<std::vector<Interval>> forward_risk(2 * count), backward_risk(2 * count);
            forward.reserve(2 * count); backward.reserve(2 * count);
            forward_theta.reserve(2 * count); backward_theta.reserve(2 * count);
            forward_motion.reserve(2 * count); backward_motion.reserve(2 * count);
            for (std::size_t i = 0; i < count; ++i) {
                const double end = std::min((i + 1) * p.search_dt, p.time_max);
                const double times[2] = {(i * p.search_dt + end) / 2, end};
                for (double time : times) {
                    forward.emplace_back(m, time, p); backward.emplace_back(m, -time, p);
                    forward_theta.push_back(g.axes[2].angular(forward.back().theta, p.theta_period));
                    backward_theta.push_back(g.axes[2].angular(backward.back().theta, p.theta_period));
                    forward_motion.emplace_back(forward.back().theta, m.vh, m.delta, m.ve, p);
                    backward_motion.emplace_back(backward.back().theta, m.vh, m.delta, m.ve, p);
                }
            }
            for (std::size_t ix = 0; ix < g.n[0]; ++ix) {
                for (std::size_t iy = 0; iy < g.n[1]; ++iy) {
                    if (symmetry && gi == mirror && iy > ry[iy]) continue;
                    const std::size_t out = g.flat(gi, ix * g.n[1] + iy);
                    const double x = g.axes[0].a[ix], y = g.axes[1].a[iy];
                    const auto evaluate = [&](double time, std::size_t sample) {
                        const bool cached = sample != std::numeric_limits<std::size_t>::max();
                        const Frame frame = cached ? (time > 0 ? forward[sample] : backward[sample])
                                                   : Frame(m, time, p);
                        const Pose pos = frame.pose(x, y);
                        const Bracket* angular = cached ? (time > 0 ? &forward_theta[sample]
                                                                 : &backward_theta[sample]) : nullptr;
                        if (p.interpolate && g.axes[0].contains(pos.x) && g.axes[1].contains(pos.y))
                            return interpolate(g, field, gi % g.fixed, pos, p, angular);
                        // Lazy motion-group plans also accelerate out-of-grid
                        // evaluations, without approximating their probability.
                        if (cached) {
                            const Motion& shifted = time > 0 ? forward_motion[sample] : backward_motion[sample];
                            auto& plan = time > 0 ? forward_risk[sample] : backward_risk[sample];
                            if (plan.empty() && !shifted.aligned) plan = make_plan(shifted, p);
                            return risk(pos.x, pos.y, shifted, p, plan.empty() ? nullptr : &plan).probability;
                        }
                        const Motion shifted(pos.theta, m.vh, m.delta, m.ve, p);
                        return risk(pos.x, pos.y, shifted, p).probability;
                    };
                    values[out] = signed_time(probability[out], p, evaluate);
                    if (symmetry)
                        values[g.flat(mirror, ix * g.n[1] + ry[iy])] = values[out];
                }
            }
        } catch (...) { failure.capture(); }
    }
    failure.check();
    (void)workers;
}

template<class Work> int protect(Work work) noexcept {
    error.clear();
    try { work(); return 0; }
    catch (const std::exception& e) { error = e.what(); return -1; }
    catch (...) { error = "Unknown native error"; return -1; }
}

void validate_states(const double* states, std::uint64_t count) {
    require(states || count == 0, "Null states pointer");
    require(count <= std::uint64_t(std::numeric_limits<std::ptrdiff_t>::max()) / 6,
            "State count overflow");
    for (std::uint64_t i = 0; i < count * 6; ++i)
        require(std::isfinite(states[i]), "States must be finite");
}
} // namespace te

extern "C" {
TE_EXPORT int te_abi_version() { return 1; }
TE_EXPORT std::size_t te_config_size() { return sizeof(TEConfig); }
TE_EXPORT const char* te_last_error() { return te::error.c_str(); }

TE_EXPORT void te_default_config(TEConfig* p) {
    if (!p) return;
    *p = TEConfig{3, 3, .05, .05, .01, 2, .3, .1, .7, 1e-9, 1e-5,
                  1.2, 1.5, 4.68, 2.20, 4.28, 1.80, 2 * te::pi, 0, 1, 1};
}

TE_EXPORT int te_compute_grid(const double* axes, const std::uint64_t* shape,
                             const TEConfig* config, double* probability,
                             double* terminal_values, double* first_contact_time) {
    return te::protect([&] {
        te::require(config && probability && terminal_values, "Null required pointer");
        te::validate(*config);
        const te::Grid grid(axes, shape, *config);
        te::compute(grid, *config, probability, terminal_values, first_contact_time);
    });
}

TE_EXPORT int te_probability_states(const double* states, std::uint64_t count,
                                   const TEConfig* config, double* probability,
                                   double* first_contact_time) {
    return te::protect([&] {
        te::require(config && (probability || count == 0), "Null required pointer");
        te::validate(*config); te::validate_states(states, count);
        const int workers = te::threads(*config);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 64) num_threads(workers)
#endif
        for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(count); ++i) {
            const double* s = states + 6 * i;
            const te::Motion motion(s[2], s[3], s[4], s[5], *config);
            const te::Risk r = te::risk(s[0], s[1], motion, *config);
            probability[i] = r.probability;
            if (first_contact_time) first_contact_time[i] = r.contact;
        }
        (void)workers;
    });
}

TE_EXPORT int te_time_states(const double* states, std::uint64_t count,
                            const TEConfig* config, double* probability,
                            double* terminal_values) {
    return te::protect([&] {
        te::require(config && ((probability && terminal_values) || count == 0),
                    "Null required pointer");
        te::validate(*config); te::validate_states(states, count);
        const int workers = te::threads(*config);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 16) num_threads(workers)
#endif
        for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(count); ++i) {
            const double* s = states + 6 * i;
            const te::Motion motion(s[2], s[3], s[4], s[5], *config);
            const double r = te::risk(s[0], s[1], motion, *config).probability;
            const auto evaluate = [&](double t, std::size_t) {
                const te::Frame frame(motion, t, *config);
                const te::Pose pos = frame.pose(s[0], s[1]);
                const te::Motion shifted(pos.theta, s[3], s[4], s[5], *config);
                return te::risk(pos.x, pos.y, shifted, *config).probability;
            };
            probability[i] = r;
            terminal_values[i] = te::signed_time(r, *config, evaluate);
        }
        (void)workers;
    });
}
} // extern C
