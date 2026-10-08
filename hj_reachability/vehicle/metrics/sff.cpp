// SFF-based signed XY distance, not NVIDIA's original safety potential.
// C++17 / OpenMP, no external dependencies. State: x,y,theta,vH,deltaE,vE.
// U = union_t [E(t) + (-H0(t))], with equal-time occupied rectangles.
// Exact polygon-union distance for the sampled times (up to floating point).
// No XY rasterization, grid clipping, footprint inflation or normalization.
// During reaction delays: constant speed; afterwards: constant braking to rest.
// Ego uses the project's kinematic bicycle with frozen steering; human yaw=0.

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

extern "C" {
struct SFFConfig {
    double dt, ego_braking, human_braking, ego_reaction_time, human_reaction_time;
    double lf, lr, ego_length, ego_width, human_length, human_width;
    std::int32_t threads, use_symmetry;
};
struct SFFStats {
    std::uint64_t total_groups, evaluated_groups, max_time_samples;
    std::uint64_t total_boundary_segments;
    std::int32_t symmetry_used, threads_used;
};
}

namespace {
constexpr double pi = 3.1415926535897932384626433832795;
constexpr double eps = 1e-10; // Roundoff tolerance, metres; no geometric margin.
thread_local std::string last_error;
void require(bool ok, const char* msg) { if (!ok) throw std::invalid_argument(msg); }
struct P {
    double x=0, y=0;
    P operator+(P b) const { return {x+b.x,y+b.y}; }
    P operator-(P b) const { return {x-b.x,y-b.y}; }
    P operator*(double s) const { return {x*s,y*s}; }
};
double dot(P a,P b) { return a.x*b.x+a.y*b.y; }
double cross(P a,P b) { return a.x*b.y-a.y*b.x; }
struct Box {
    double x0=INFINITY,y0=INFINITY,x1=-INFINITY,y1=-INFINITY;
    void add(P p) { x0=std::min(x0,p.x); y0=std::min(y0,p.y);
                   x1=std::max(x1,p.x); y1=std::max(y1,p.y); }
    void add(const Box& b) { add(P{b.x0,b.y0}); add(P{b.x1,b.y1}); }
    bool contains(P p) const { return p.x>=x0-eps && p.x<=x1+eps &&
                                     p.y>=y0-eps && p.y<=y1+eps; }
    bool overlaps(const Box& b) const { return x0<=b.x1+eps && b.x0<=x1+eps &&
                                             y0<=b.y1+eps && b.y0<=y1+eps; }
    double distance2(P p) const {
        double dx=std::max({x0-p.x,0.0,p.x-x1});
        double dy=std::max({y0-p.y,0.0,p.y-y1}); return dx*dx+dy*dy;
    }
};
struct Plane { P n; double c; }; // n points inward; n.p >= c.
struct Poly {
    std::array<P,8> v;
    std::array<Plane,8> h;
    int size=0;
    Box box;
    bool contains(P p) const {
        if (!box.contains(p)) return false;
        for(int i=0;i<size;++i) if(dot(h[i].n,p)<h[i].c-eps) return false;
        return true;
    }
};
struct Segment {
    P a,d; double inv_length2;
    Box box;
    Segment(P a0,P b):a(a0),d(b-a0),inv_length2(1/dot(d,d)) {
        box.add(a);box.add(b);
    }
    double distance2(P p) const {
        double t=std::clamp(dot(p-a,d)*inv_length2,0.0,1.0);
        P q=p-(a+d*t);return dot(q,q);
    }
};

// Balanced bounding-volume tree, shared by polygon and boundary queries.
template<class Item> class Tree {
    struct Node { Box box; int begin,end,left=-1,right=-1; };
    const std::vector<Item>* items;
    std::vector<int> order;
    std::vector<Node> nodes;
    int build(int begin,int end) {
        int k=static_cast<int>(nodes.size()); nodes.emplace_back();
        Box box; for(int i=begin;i<end;++i) box.add((*items)[order[i]].box);
        nodes[k].box=box;nodes[k].begin=begin;nodes[k].end=end;
        if(end-begin>8) {
            bool x=box.x1-box.x0>=box.y1-box.y0; int mid=(begin+end)/2;
            std::nth_element(order.begin()+begin,order.begin()+mid,order.begin()+end,
                [&](int a,int b){const Box& A=(*items)[a].box; const Box& B=(*items)[b].box;
                    return x ? A.x0+A.x1<B.x0+B.x1 : A.y0+A.y1<B.y0+B.y1;});
            int left=build(begin,mid),right=build(mid,end);
            nodes[k].left=left;nodes[k].right=right;
        }
        return k;
    }
    bool inside(int k,P p) const {
        const Node& n=nodes[k]; if(!n.box.contains(p)) return false;
        if(n.left<0) {
            for(int i=n.begin;i<n.end;++i) if((*items)[order[i]].contains(p)) return true;
            return false;
        }
        return inside(n.left,p)||inside(n.right,p);
    }
    void nearest(int k,P p,double& best) const {
        const Node& n=nodes[k]; if(n.box.distance2(p)>=best) return;
        if(n.left<0) {
            for(int i=n.begin;i<n.end;++i) {
                const Item& s=(*items)[order[i]];
                if(s.box.distance2(p)<best) best=std::min(best,s.distance2(p));
            }
            return;
        }
        double dl=nodes[n.left].box.distance2(p),dr=nodes[n.right].box.distance2(p);
        if(dl<dr) { nearest(n.left,p,best);nearest(n.right,p,best); }
        else { nearest(n.right,p,best);nearest(n.left,p,best); }
    }
    void overlaps(int k,const Box& box,std::vector<int>& out) const {
        const Node& n=nodes[k]; if(!n.box.overlaps(box)) return;
        if(n.left<0) {
            for(int i=n.begin;i<n.end;++i) if((*items)[order[i]].box.overlaps(box))
                out.push_back(order[i]);
        } else { overlaps(n.left,box,out);overlaps(n.right,box,out); }
    }
public:
    explicit Tree(const std::vector<Item>& v):items(&v),order(v.size()) {
        require(!v.empty(),"Empty geometry");
        require(v.size()<static_cast<std::size_t>(std::numeric_limits<int>::max()/2),
                "Geometry is too large");
        std::iota(order.begin(),order.end(),0);nodes.reserve(v.size()*2);build(0,order.size());
    }
    bool contains(P p) const { return inside(0,p); }
    double distance2(P p) const { double b=INFINITY;nearest(0,p,b);return b; }
    void overlapping(const Box& box,std::vector<int>& out) const {out.clear();overlaps(0,box,out);}
};

void validate(const SFFConfig& c) {
    const double fields[]={c.dt,c.ego_braking,c.human_braking,c.ego_reaction_time,
        c.human_reaction_time,c.lf,c.lr,c.ego_length,c.ego_width,c.human_length,c.human_width};
    for(double x:fields) require(std::isfinite(x),"Parameters must be finite");
    require(c.dt>0 && c.ego_braking>0 && c.human_braking>0,"dt and braking must be positive");
    require(c.ego_reaction_time>=0 && c.human_reaction_time>=0,"Reaction times must be nonnegative");
    require(c.lf>0 && c.lr>0 && c.ego_length>0 && c.ego_width>0 &&
            c.human_length>0 && c.human_width>0,"Vehicle dimensions must be positive");
    require(c.threads>=0,"threads must be nonnegative");
    require(c.use_symmetry==0 || c.use_symmetry==1,"use_symmetry must be 0 or 1");
}
void validate_eta(const double* e) {
    for(int i=0;i<4;++i) require(std::isfinite(e[i]),"Kinematic states must be finite");
    require(e[1]>=0 && e[3]>=0,"Speeds must be nonnegative");
    require(std::abs(e[2])<pi/2-1e-6,"Steering must lie strictly between -pi/2 and pi/2");
}
double stopping_time(double v,double b,double tau) { return v==0?0:tau+v/b; }
double travelled(double v,double b,double tau,double t) {
    if(t<=tau) return v*t;
    double u=std::min(t-tau,v/b);return v*tau+v*u-0.5*b*u*u;
}
std::vector<double> time_nodes(const double* e,const SFFConfig& c) {
    double te=stopping_time(e[3],c.ego_braking,c.ego_reaction_time);
    double th=stopping_time(e[1],c.human_braking,c.human_reaction_time);
    double T=std::max(te,th),steps=std::ceil(T/c.dt);
    require(std::isfinite(steps) && steps<=100000,"Too many time samples (limit 100000)");
    std::vector<double> ts;ts.reserve(static_cast<std::size_t>(steps)+6);
    for(std::size_t i=0;i<static_cast<std::size_t>(steps);++i) ts.push_back(i*c.dt);
    ts.push_back(0);ts.push_back(T);ts.push_back(te);ts.push_back(th);
    if(c.ego_reaction_time<T) ts.push_back(c.ego_reaction_time);
    if(c.human_reaction_time<T) ts.push_back(c.human_reaction_time);
    std::sort(ts.begin(),ts.end());
    ts.erase(std::unique(ts.begin(),ts.end(),[](double a,double b){return std::abs(a-b)<1e-13;}),ts.end());
    return ts;
}

// Minkowski sum of two centered rectangles is a 2D zonotope with <=8 edges.
Poly collision_poly(P center,double psi,double theta,const SFFConfig& c) {
    double ce=std::cos(psi),se=std::sin(psi),ch=std::cos(theta),sh=std::sin(theta);
    std::array<P,4> g={P{ce*c.ego_length/2,se*c.ego_length/2},
        P{-se*c.ego_width/2,ce*c.ego_width/2},
        P{ch*c.human_length/2,sh*c.human_length/2},
        P{-sh*c.human_width/2,ch*c.human_width/2}};
    for(P& q:g) if(q.y<0 || (q.y==0 && q.x<0)) q=q*(-1);
    std::sort(g.begin(),g.end(),[](P a,P b){return cross(a,b)>0;});
    std::array<P,4> merged;int n=0;
    for(P q:g) {
        if(n && std::abs(cross(merged[n-1],q))<=1e-13*std::sqrt(dot(merged[n-1],merged[n-1])*dot(q,q)))
            merged[n-1]=merged[n-1]+q;
        else merged[n++]=q;
    }
    P start=center;for(int i=0;i<n;++i) start=start-merged[i];
    Poly p;p.size=2*n;P q=start;
    for(int i=0;i<2*n;++i) {
        p.v[i]=q;p.box.add(q);q=q+merged[i%n]*(i<n?2:-2);
    }
    for(int i=0;i<p.size;++i) {
        P d=p.v[(i+1)%p.size]-p.v[i]; double inv=1/std::sqrt(dot(d,d));
        P normal=P{-d.y,d.x}*inv;p.h[i]={normal,dot(normal,p.v[i])};
    }
    return p;
}
std::vector<Poly> polygons(const double* e,const SFFConfig& c,std::uint64_t& samples) {
    auto ts=time_nodes(e,c);samples=ts.size();
    std::vector<Poly> ps;ps.reserve(ts.size());
    double tangent=std::tan(e[2]),beta=std::atan(c.lr/(c.lf+c.lr)*tangent);
    double curvature=std::cos(beta)*tangent/(c.lf+c.lr);
    double ch=std::cos(e[0]),sh=std::sin(e[0]);
    for(double t:ts) {
        double se=travelled(e[3],c.ego_braking,c.ego_reaction_time,t);
        double sH=travelled(e[1],c.human_braking,c.human_reaction_time,t);
        double psi=curvature*se,half=psi/2;
        double sinc=std::abs(half)<1e-8?1-half*half/6:std::sin(half)/half;
        P posE={se*sinc*std::cos(beta+half),se*sinc*std::sin(beta+half)};
        Poly p=collision_poly(posE-P{sH*ch,sH*sh},psi,e[0],c);
        bool same=!ps.empty() && p.size==ps.back().size;
        if(same) for(int i=0;i<p.size;++i) {
            P d=p.v[i]-ps.back().v[i];if(dot(d,d)>1e-26) {same=false;break;}
        }
        if(!same) ps.push_back(p);
    }
    return ps;
}

struct Interval { double lo,hi; };
// Clip an edge against a convex polygon. Shared outer edges belong to one
// polygon only; opposite-facing coincident edges are interior to the union.
bool covered_interval(P a,P d,const Poly& p,int owner,int other,Interval& out) {
    double lo=0,hi=1;
    bool coincident_same=false;
    const double parallel_tolerance=1e-14*std::max(1.0,std::sqrt(dot(d,d)));
    for(int i=0;i<p.size;++i) {
        const P origin=p.v[i],normal=p.h[i].n;
        long double f=static_cast<long double>(normal.x)*(static_cast<long double>(a.x)-origin.x)
                     +static_cast<long double>(normal.y)*(static_cast<long double>(a.y)-origin.y);
        long double v=static_cast<long double>(normal.x)*d.x+static_cast<long double>(normal.y)*d.y;
        if(std::abs(v)<=parallel_tolerance) {
            if(f<-1e-13) return false;
            if(std::abs(f)<=1e-13) {
                P edge=p.v[(i+1)%p.size]-p.v[i];
                if(dot(edge,d)>0) coincident_same=true;
            }
        } else {
            double u=-f/v;
            if(v>0) lo=std::max(lo,u);else hi=std::min(hi,u);
            if(hi<=lo) return false;
        }
    }
    if(coincident_same && other>owner) return false;
    out={lo,hi};return hi>lo;
}
void subtract_interval(std::vector<Interval>& live,Interval cover) {
    for(std::size_t i=0;i<live.size();) {
        Interval q=live[i];
        if(cover.hi<=q.lo || cover.lo>=q.hi) {++i;continue;}
        if(cover.lo<=q.lo && cover.hi>=q.hi) {live.erase(live.begin()+i);continue;}
        if(cover.lo<=q.lo) {live[i].lo=cover.hi;++i;}
        else if(cover.hi>=q.hi) {live[i].hi=cover.lo;++i;}
        else {live[i].hi=cover.lo;live.insert(live.begin()+i+1,{cover.hi,q.hi});i+=2;}
    }
}
std::vector<Segment> union_boundary(const std::vector<Poly>& ps,const Tree<Poly>& tree) {
    std::vector<Segment> segments;segments.reserve(ps.size()*4);
    std::vector<int> candidates;std::vector<Interval> live;live.reserve(8);
    for(int i=0;i<static_cast<int>(ps.size());++i) {
        const Poly& p=ps[i];
        tree.overlapping(p.box,candidates);
        // Neighboring times usually cover an interior edge immediately.
        std::sort(candidates.begin(),candidates.end(),[i](int a,int b){return std::abs(a-i)<std::abs(b-i);});
        for(int k=0;k<p.size;++k) {
            P a=p.v[k],d=p.v[(k+1)%p.size]-a;
            Box edgebox;edgebox.add(a);edgebox.add(a+d);
            live.clear();live.push_back({0,1});
            for(int j:candidates) {
                if(j==i || !edgebox.overlaps(ps[j].box)) continue;
                Interval cover;
                if(covered_interval(a,d,ps[j],i,j,cover)) subtract_interval(live,cover);
                if(live.empty()) break;
            }
            for(Interval q:live) {
                P A=a+d*q.lo,B=a+d*q.hi,ab=B-A;
                if(dot(ab,ab)>1e-22) segments.emplace_back(A,B);
            }
        }
    }
    require(!segments.empty(),"Union boundary extraction failed");return segments;
}
struct Slice {
    std::uint64_t samples;
    std::vector<Poly> ps;
    Tree<Poly> poly_tree;
    std::vector<Segment> segments;
    Tree<Segment> edge_tree;
    Slice(const double* e,const SFFConfig& c):samples(0),ps(polygons(e,c,samples)),
        poly_tree(ps),segments(union_boundary(ps,poly_tree)),edge_tree(segments) {}
    double value(P p) const {
        double d=std::sqrt(edge_tree.distance2(p));
        if(d<=eps) return 0;
        return poly_tree.contains(p)?-d:d;
    }
};
int thread_count(int requested,std::uint64_t tasks) {
    int n=1;
#ifdef _OPENMP
    n=requested?requested:omp_get_max_threads();
#else
    (void)requested;
#endif
    return static_cast<int>(std::min<std::uint64_t>(n,std::max<std::uint64_t>(tasks,1)));
}
std::vector<std::uint64_t> reflection(const double* a,std::uint64_t n,bool angular) {
    std::vector<std::uint64_t> result(n,n);
    for(std::uint64_t i=0;i<n;++i) {
        double best=INFINITY;
        for(std::uint64_t j=0;j<n;++j) {
            double d=a[i]+a[j];if(angular) d=std::remainder(d,2*pi);
            if(std::abs(d)<best) {best=std::abs(d);result[i]=j;}
        }
        if(best>1e-12) return {};
    }
    for(std::uint64_t i=0;i<n;++i) if(result[result[i]]!=i) return {};
    return result;
}
template<class F> int guarded(F f) {
    try {last_error.clear();f();return 0;}
    catch(const std::exception& e) {last_error=e.what();return 1;}
    catch(...) {last_error="Unknown native error";return 1;}
}
}

extern "C" {
int sff_abi_version() {return 1;}
std::size_t sff_config_size() {return sizeof(SFFConfig);}
std::size_t sff_stats_size() {return sizeof(SFFStats);}
const char* sff_last_error() {return last_error.c_str();}
void sff_default_config(SFFConfig* c) {
    if(c) *c={0.01,7.0,7.0,0.0,0.0,1.2,1.5,4.68,2.20,4.28,1.80,0,1};
}
// Pairwise XY points sharing [theta,vH,deltaE,vE]. Builds the union once.
int sff_compute_points(const double* xy,std::uint64_t n,const double* eta,
                       const SFFConfig* config,double* out,SFFStats* stats) {
    return guarded([&]{
        require(config && eta && stats,"Null pointer");validate(*config);validate_eta(eta);
        require(n==0 || (xy && out),"Null XY/output pointer");
        for(std::uint64_t i=0;i<2*n;++i) require(std::isfinite(xy[i]),"XY must be finite");
        Slice s(eta,*config);int nt=thread_count(config->threads,n);
        *stats={1,1,s.samples,s.segments.size(),0,nt};
        #pragma omp parallel for num_threads(nt) schedule(static)
        for(std::uint64_t i=0;i<n;++i) out[i]=s.value({xy[2*i],xy[2*i+1]});
    });
}
// Axis data = concatenation of six axes. Output = C-contiguous 6D array.
int sff_compute_grid(const double* axes,const std::uint64_t* shape,
                    const SFFConfig* config,double* out,SFFStats* stats) {
    return guarded([&]{
        require(axes && shape && config && out && stats,"Null pointer");validate(*config);
        std::array<const double*,6> a;std::uint64_t offset=0,total=1;
        for(int k=0;k<6;++k) {
            require(shape[k]>0,"Empty grid axis");
            require(shape[k]<=std::numeric_limits<std::uint64_t>::max()/total,"Grid size overflow");
            total*=shape[k];a[k]=axes+offset;offset+=shape[k];
            for(std::uint64_t i=0;i<shape[k];++i) {
                require(std::isfinite(a[k][i]),"Grid axes must be finite");
                if(i) require(a[k][i]>a[k][i-1],"Grid axes must be strictly increasing");
            }
        }
        require(total<=std::numeric_limits<std::size_t>::max()/sizeof(double),"Grid output is too large");
        require(a[3][0]>=0 && a[5][0]>=0,"Grid speeds must be nonnegative");
        require(a[4][0]>-pi/2+1e-6 && a[4][shape[4]-1]<pi/2-1e-6,"Invalid grid steering");
        // Check the largest stop horizon before launching worker threads.
        double worst[]={a[2][0],a[3][shape[3]-1],a[4][0],a[5][shape[5]-1]};
        time_nodes(worst,*config);
        auto yr=config->use_symmetry?reflection(a[1],shape[1],false):std::vector<std::uint64_t>{};
        auto tr=config->use_symmetry?reflection(a[2],shape[2],true):std::vector<std::uint64_t>{};
        auto dr=config->use_symmetry?reflection(a[4],shape[4],false):std::vector<std::uint64_t>{};
        bool symmetry=!yr.empty() && !tr.empty() && !dr.empty();
        std::uint64_t groups=shape[2]*shape[3]*shape[4]*shape[5];
        struct Task { std::uint64_t group,mirror,it,ih,id,ie; };
        std::vector<Task> tasks;tasks.reserve(groups);
        for(std::uint64_t it=0;it<shape[2];++it) for(std::uint64_t ih=0;ih<shape[3];++ih)
        for(std::uint64_t id=0;id<shape[4];++id) for(std::uint64_t ie=0;ie<shape[5];++ie) {
            std::uint64_t g=((it*shape[3]+ih)*shape[4]+id)*shape[5]+ie;
            std::uint64_t m=symmetry?((tr[it]*shape[3]+ih)*shape[4]+dr[id])*shape[5]+ie:g;
            if(g<=m) tasks.push_back({g,m,it,ih,id,ie});
        }
        int nt=thread_count(config->threads,tasks.size());
        std::atomic<bool> failed{false};std::string error;
        std::uint64_t samples=0,segments=0;
        #pragma omp parallel for num_threads(nt) schedule(dynamic,1) reduction(max:samples) reduction(+:segments)
        for(std::uint64_t q=0;q<tasks.size();++q) {
            if(failed.load(std::memory_order_relaxed)) continue;
            try {
                const Task& task=tasks[q];
                double e[]={a[2][task.it],a[3][task.ih],a[4][task.id],a[5][task.ie]};
                Slice s(e,*config);samples=std::max(samples,s.samples);segments+=s.segments.size();
                for(std::uint64_t ix=0;ix<shape[0];++ix) for(std::uint64_t iy=0;iy<shape[1];++iy) {
                    double v=s.value({a[0][ix],a[1][iy]});
                    out[(ix*shape[1]+iy)*groups+task.group]=v;
                    if(task.mirror!=task.group) out[(ix*shape[1]+yr[iy])*groups+task.mirror]=v;
                }
            } catch(const std::exception& ex) {
                #pragma omp critical(sff_error)
                {if(!failed.exchange(true)) error=ex.what();}
            } catch(...) {
                #pragma omp critical(sff_error)
                {if(!failed.exchange(true)) error="Unknown OpenMP worker error";}
            }
        }
        if(failed) throw std::runtime_error(error);
        *stats={groups,tasks.size(),samples,segments,static_cast<std::int32_t>(symmetry),nt};
    });
}
}
