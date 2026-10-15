// g++ -O3 -march=native -std=c++17 -fopenmp -fPIC -shared long_lat.cpp -o _long_lat.so
// C ABI: float64 arrays, uint64 shape[6], NumPy C order [x,y,theta,vH,deltaE,vE].
// Vlong=min_t d_brake; Vlat=max(min_t d_left,min_t d_right); V0=min(Vlong,Vlat).
// d is signed rectangle clearance in metres minus buffer. Times include 0 and H.
// Human: constant speed and heading. Ego: bicycle model, centred rectangles.
// Grid calls take steering bounds from axis 4; point calls use LLConfig bounds.
// Minimum is over sampled times, not a continuous-time collision certificate.
// Output buffers must not alias each other or inputs. Components may be null.

#include <algorithm>
#include <array>
#include <atomic>
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

extern "C" {
struct LLConfig {
    double horizon, dt, buffer, ego_braking, steering_rate;
    double delta_min, delta_max, lf, lr;
    double ego_length, ego_width, human_length, human_width;
    std::int32_t threads, use_symmetry;
};
struct LLStats {
    std::uint64_t total_groups, evaluated_groups, time_samples, ego_trajectories;
    std::int32_t symmetry_used, threads_used;
};
}

namespace ll {
constexpr double pi = 3.14159265358979323846;
constexpr double inf = std::numeric_limits<double>::infinity();
thread_local std::string error;
void require(bool ok, const char* message) {
    if (!ok) throw std::invalid_argument(message);
}
void validate(const LLConfig& p) {
    for (double v : {p.horizon, p.dt, p.ego_braking, p.steering_rate,
                     p.lf, p.lr, p.ego_length, p.ego_width,
                     p.human_length, p.human_width})
        require(std::isfinite(v) && v > 0, "Positive parameters must be finite and > 0");
    require(std::isfinite(p.buffer) && p.buffer >= 0, "buffer must be finite and >= 0");
    require(std::isfinite(p.delta_min) && std::isfinite(p.delta_max) &&
            p.delta_min <= p.delta_max && p.delta_min > -pi/2 && p.delta_max < pi/2,
            "Steering bounds must satisfy -pi/2 < delta_min <= delta_max < pi/2");
    require(p.threads >= 0, "threads must be >= 0");
    require(p.use_symmetry == 0 || p.use_symmetry == 1, "use_symmetry must be 0 or 1");
    require(p.dt <= p.horizon && std::ceil(p.horizon/p.dt) <= 1000000,
            "Require dt <= horizon and at most one million intervals");
}
struct Pose { double x=0, y=0, psi=0; };
struct Kinematics { double beta, curvature; };
Kinematics kinematics(double delta, const LLConfig& p) {
    const double tangent = std::tan(delta);
    const double beta = std::atan(p.lr/(p.lf+p.lr)*tangent);
    return {beta, std::cos(beta)*tangent/(p.lf+p.lr)};
}
Pose advance_arc(Pose q, double travel, double delta, const LLConfig& p) {
    const auto k = kinematics(delta, p);
    const double half = k.curvature*travel/2;
    const double sinc = std::abs(half) < 1e-7
        ? 1-half*half/6+half*half*half*half/120 : std::sin(half)/half;
    q.x += travel*sinc*std::cos(q.psi+k.beta+half);
    q.y += travel*sinc*std::sin(q.psi+k.beta+half);
    q.psi += 2*half;
    return q;
}
Pose derivative(const Pose& q, double speed, double delta, const LLConfig& p) {
    const auto k = kinematics(delta, p);
    return {speed*std::cos(q.psi+k.beta), speed*std::sin(q.psi+k.beta),
            speed*k.curvature};
}
Pose shifted(Pose q, const Pose& k, double h) {
    return {q.x+h*k.x, q.y+h*k.y, q.psi+h*k.psi};
}
Pose advance_ramp(Pose q, double speed, double delta, double rate,
                  double h, const LLConfig& p) {
    const auto a = derivative(q, speed, delta, p);
    const auto b = derivative(shifted(q,a,h/2), speed, delta+rate*h/2, p);
    const auto c = derivative(shifted(q,b,h/2), speed, delta+rate*h/2, p);
    const auto d = derivative(shifted(q,c,h), speed, delta+rate*h, p);
    q.x += h/6*(a.x+2*b.x+2*c.x+d.x);
    q.y += h/6*(a.y+2*b.y+2*c.y+d.y);
    q.psi += h/6*(a.psi+2*b.psi+2*c.psi+d.psi);
    return q;
}
using Trajectories = std::array<std::vector<Pose>,3>;
std::vector<double> sample_times(const LLConfig& p) {
    const auto n = static_cast<std::size_t>(std::ceil(p.horizon/p.dt));
    std::vector<double> times(n+1);
    for (std::size_t i=0; i<=n; ++i) times[i]=std::min(i*p.dt,p.horizon);
    return times;
}
Trajectories trajectories(double delta, double speed, const LLConfig& p,
                           const std::vector<double>& times) {
    require(std::isfinite(delta) && delta >= p.delta_min && delta <= p.delta_max,
            "Initial steering is outside steering bounds");
    require(std::isfinite(speed) && speed >= 0, "Ego speed must be finite and >= 0");
    Trajectories result;
    for (auto& path : result) path.resize(times.size());
    for (std::size_t i=0; i<times.size(); ++i) {
        const double t = std::min(times[i],speed/p.ego_braking);
        result[0][i]=advance_arc({},speed*t-0.5*p.ego_braking*t*t,delta,p);
    }
    for (int m=1; m<3; ++m) {
        const double rate = m==1 ? p.steering_rate : -p.steering_rate;
        const double limit = m==1 ? p.delta_max : p.delta_min;
        const double hit = (limit-delta)/rate;
        Pose q;
        for (std::size_t i=1; i<times.size(); ++i) {
            const double start=times[i-1], end=times[i];
            const double ramp=std::max(0.0,std::min(end,hit)-start);
            if (ramp>0) q=advance_ramp(q,speed,delta+rate*start,rate,ramp,p);
            const double hold=end-start-ramp;
            if (hold>0) q=advance_arc(q,speed*hold,limit,p);
            result[m][i]=q;
        }
    }
    return result;
}

// Each frame is an affine map from initial XY to the instantaneous ego frame.
struct Frame {
    double c,s,ox,oy,ct,st,bx,by,bu,bv;
    std::array<double,4> hx,hy;
    Frame(const Pose& e, double time, double theta, double vh, const LLConfig& p) {
        c=std::cos(e.psi); s=std::sin(e.psi);
        const double dx=vh*time*std::cos(theta)-e.x;
        const double dy=vh*time*std::sin(theta)-e.y;
        ox=c*dx+s*dy; oy=-s*dx+c*dy;
        ct=std::cos(theta-e.psi); st=std::sin(theta-e.psi);
        const double a=p.ego_length/2, b=p.ego_width/2;
        const double h=p.human_length/2, w=p.human_width/2;
        bx=a+h*std::abs(ct)+w*std::abs(st);
        by=b+h*std::abs(st)+w*std::abs(ct);
        bu=h+a*std::abs(ct)+b*std::abs(st);
        bv=w+a*std::abs(st)+b*std::abs(ct);
        for (int i=0;i<4;++i) {
            const double u=(i&1)?h:-h, v=(i&2)?w:-w;
            hx[i]=ct*u-st*v; hy[i]=st*u+ct*v;
        }
    }
    double margin(double x, double y) const {
        return std::max({std::abs(x)-bx, std::abs(y)-by,
            std::abs(ct*x+st*y)-bu, std::abs(-st*x+ct*y)-bv});
    }
    double separation(double x, double y, const LLConfig& p) const {
        const double a=p.ego_length/2, b=p.ego_width/2;
        const double h=p.human_length/2, w=p.human_width/2;
        double best=inf;
        for (int i=0;i<4;++i) {
            const double dx=std::max(std::abs(x+hx[i])-a,0.0);
            const double dy=std::max(std::abs(y+hy[i])-b,0.0);
            best=std::min(best,dx*dx+dy*dy);
            const double ex=((i&1)?a:-a)-x, ey=((i&2)?b:-b)-y;
            const double ux=std::max(std::abs(ct*ex+st*ey)-h,0.0);
            const double uy=std::max(std::abs(-st*ex+ct*ey)-w,0.0);
            best=std::min(best,ux*ux+uy*uy);
        }
        return std::sqrt(best);
    }
};
using Plans = std::array<std::vector<Frame>,3>;
Plans plans(const Trajectories& paths, const std::vector<double>& times,
            double theta, double vh, const LLConfig& p) {
    require(std::isfinite(theta) && std::isfinite(vh) && vh>=0,
            "Human heading must be finite; speed must be finite and >= 0");
    Plans result;
    for (int m=0;m<3;++m) {
        result[m].reserve(times.size());
        for (std::size_t i=0;i<times.size();++i)
            result[m].emplace_back(paths[m][i],times[i],theta,vh,p);
    }
    return result;
}
double minimum_clearance(double x, double y, const std::vector<Frame>& plan,
                         const LLConfig& p) {
    double best=inf;
    for (const auto& f : plan) {
        const double u=f.c*x+f.s*y+f.ox, v=-f.s*x+f.c*y+f.oy;
        const double gap=f.margin(u,v);
        // SAT separation is a lower bound outside and penetration depth inside.
        if (gap>=best) continue;
        const double distance=gap<=0 ? gap : f.separation(u,v,p);
        best=std::min(best,distance);
    }
    return best-p.buffer;
}
struct Values { double combined, longitudinal, lateral; };
Values evaluate(double x,double y,const Plans& plan,const LLConfig& p) {
    const double longitudinal=minimum_clearance(x,y,plan[0],p);
    const double left=minimum_clearance(x,y,plan[1],p);
    const double right=minimum_clearance(x,y,plan[2],p);
    const double lateral=std::max(left,right);
    return {std::min(longitudinal,lateral),longitudinal,lateral};
}
struct Axis { const double* a; std::size_t n; };
struct Grid {
    std::array<Axis,6> axes;
    std::size_t total=1,groups,xy;
    Grid(const double* data,const std::uint64_t* shape) {
        require(data && shape,"Null grid pointer");
        for (int d=0;d<6;++d) {
            require(shape[d]>0 && shape[d]<=
                    std::uint64_t(std::numeric_limits<std::ptrdiff_t>::max())/total,
                    "Empty axis or grid shape overflow");
            const auto n=static_cast<std::size_t>(shape[d]);
            axes[d]={data,n}; total*=n;
            for (std::size_t i=0;i<n;++i) {
                require(std::isfinite(data[i]),"Grid coordinates must be finite");
                if(i) require(data[i]>data[i-1],"Axes must be strictly increasing");
            }
            data+=n;
        }
        require(total<=std::numeric_limits<std::size_t>::max()/sizeof(double),
                "Grid byte-size overflow");
        require(axes[3].a[0]>=0 && axes[5].a[0]>=0,"Speeds must be >= 0");
        require(axes[2].a[axes[2].n-1]-axes[2].a[0]<=2*pi+1e-6,
                "Heading axis spans more than 2*pi");
        xy=axes[0].n*axes[1].n; groups=total/xy;
    }
    std::array<std::size_t,4> unpack(std::size_t g) const {
        std::array<std::size_t,4> i;
        for (int d=5;d>=2;--d) { i[d-2]=g%axes[d].n;g/=axes[d].n; }
        return i;
    }
    std::size_t group(std::size_t t,std::size_t h,std::size_t d,std::size_t e) const {
        return ((t*axes[3].n+h)*axes[4].n+d)*axes[5].n+e;
    }
};
std::vector<std::size_t> reflection(Axis axis, bool angular=false) {
    std::vector<std::size_t> map(axis.n);
    // Tolerance accommodates coordinates generated in JAX float32.
    for(std::size_t i=0;i<axis.n;++i) {
        double best=inf;
        for(std::size_t j=0;j<axis.n;++j) {
            double gap=axis.a[i]+axis.a[j];
            if(angular) gap=std::remainder(gap,2*pi);
            if(std::abs(gap)<best) {best=std::abs(gap);map[i]=j;}
        }
        if(best>1e-6) return {};
    }
    for(std::size_t i=0;i<axis.n;++i) if(map[map[i]]!=i) return {};
    return map;
}
int workers(const LLConfig& p) {
#ifdef _OPENMP
    return p.threads ? p.threads : omp_get_max_threads();
#else
    (void)p; return 1;
#endif
}
void store(std::size_t index,Values v,double* combined,double* longitudinal,double* lateral) {
    combined[index]=v.combined;
    if(longitudinal) longitudinal[index]=v.longitudinal;
    if(lateral) lateral[index]=v.lateral;
}
void compute(const Grid& g,const LLConfig& p,double* combined,double* longitudinal,
             double* lateral,LLStats& stats) {
    const auto times=sample_times(p);
    const auto nd=g.axes[4].n, ne=g.axes[5].n;
    std::vector<Trajectories> paths;
    paths.reserve(nd*ne);
    for(std::size_t d=0;d<nd;++d) for(std::size_t e=0;e<ne;++e)
        paths.push_back(trajectories(g.axes[4].a[d],g.axes[5].a[e],p,times));
    std::vector<std::size_t> my,mt,md;
    if(p.use_symmetry && std::abs(p.delta_min+p.delta_max)<=1e-12) {
        my=reflection(g.axes[1]);mt=reflection(g.axes[2],true);md=reflection(g.axes[4]);
    }
    const bool symmetric=!my.empty() && !mt.empty() && !md.empty();
    std::vector<std::size_t> jobs;
    for(std::size_t j=0;j<g.groups;++j) {
        auto i=g.unpack(j);
        const auto partner=symmetric ? g.group(mt[i[0]],i[1],md[i[2]],i[3]) : j;
        if(j<=partner) jobs.push_back(j);
    }
    stats={g.groups,jobs.size(),times.size(),3*nd*ne,symmetric,1};
    std::atomic<bool> failed{false};
    std::string failure;
    const int count=workers(p);
    (void)count;
#ifdef _OPENMP
#pragma omp parallel num_threads(count)
#endif
    {
#ifdef _OPENMP
#pragma omp single
        stats.threads_used=omp_get_num_threads();
#pragma omp for schedule(dynamic,1)
#endif
        for(std::ptrdiff_t job=0;job<static_cast<std::ptrdiff_t>(jobs.size());++job) {
            if(failed.load(std::memory_order_relaxed)) continue;
            try {
                const auto group=jobs[job];
                const auto i=g.unpack(group);
                const auto partner=symmetric ? g.group(mt[i[0]],i[1],md[i[2]],i[3]) : group;
                const auto plan=plans(paths[i[2]*ne+i[3]],times,
                    g.axes[2].a[i[0]],g.axes[3].a[i[1]],p);
                for(std::size_t ix=0;ix<g.axes[0].n;++ix)
                    for(std::size_t iy=0;iy<g.axes[1].n;++iy) {
                        if(symmetric && group==partner && iy>my[iy]) continue;
                        const auto v=evaluate(g.axes[0].a[ix],g.axes[1].a[iy],plan,p);
                        store((ix*g.axes[1].n+iy)*g.groups+group,v,
                              combined,longitudinal,lateral);
                        if(symmetric) store((ix*g.axes[1].n+my[iy])*g.groups+partner,v,
                                            combined,longitudinal,lateral);
                    }
            } catch (...) {
#ifdef _OPENMP
#pragma omp critical(ll_failure)
#endif
                {
                    if(!failed.load()) {
                        try {throw;}
                        catch(const std::exception& e) {failure=e.what();}
                        catch(...) {failure="Unknown worker error";}
                        failed.store(true);
                    }
                }
            }
        }
    }
    if(failed.load()) throw std::runtime_error(failure);
}
template<class Function> int guarded(Function action) noexcept {
    error.clear();
    try {action();return 0;}
    catch(const std::exception& e) {error=e.what();return -1;}
    catch(...) {error="Unknown native error";return -1;}
}
} // namespace ll

extern "C" {
int ll_abi_version() { return 1; }
std::size_t ll_config_size() { return sizeof(LLConfig); }
std::size_t ll_stats_size() { return sizeof(LLStats); }
const char* ll_last_error() { return ll::error.c_str(); }
void ll_default_config(LLConfig* p) {
    if(p) *p={3.0,0.05,0.0,7.0,0.087,-ll::pi/12,ll::pi/12,
              1.2,1.5,4.68,2.20,4.28,1.80,0,1};
}
int ll_compute_grid(const double* axes,const std::uint64_t* shape,const LLConfig* config,
                    double* combined,double* longitudinal,double* lateral,LLStats* stats) {
    return ll::guarded([&] {
        ll::require(config && combined,"Null configuration or output");
        const ll::Grid grid(axes,shape);
        LLConfig p=*config;
        p.delta_min=grid.axes[4].a[0];
        p.delta_max=grid.axes[4].a[grid.axes[4].n-1];
        ll::validate(p);
        LLStats result{};
        ll::compute(grid,p,combined,longitudinal,lateral,result);
        if(stats) *stats=result;
    });
}
int ll_compute_points(const double* xy,std::uint64_t count,const double* eta,
                      const LLConfig* config,double* combined,
                      double* longitudinal,double* lateral) {
    return ll::guarded([&] {
        ll::require(config && eta && (count==0 || (xy && combined)),"Null point input or output");
        ll::require(count<=std::uint64_t(std::numeric_limits<std::ptrdiff_t>::max())/2,
                    "Point count overflow");
        ll::validate(*config);
        for(std::uint64_t i=0;i<2*count;++i)
            ll::require(std::isfinite(xy[i]),"XY coordinates must be finite");
        const auto times=ll::sample_times(*config);
        const auto paths=ll::trajectories(eta[2],eta[3],*config,times);
        const auto plan=ll::plans(paths,times,eta[0],eta[1],*config);
        const int threads=ll::workers(*config);
        (void)threads;
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads) schedule(static)
#endif
        for(std::ptrdiff_t i=0;i<static_cast<std::ptrdiff_t>(count);++i)
            ll::store(i,ll::evaluate(xy[2*i],xy[2*i+1],plan,*config),
                      combined,longitudinal,lateral);
    });
}
}
