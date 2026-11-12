// SPDX-License-Identifier: MIT
#pragma once
#include <xr/hand/state.h>
#include <array>
#include <algorithm>
#include <cmath>

namespace xr::hand {
// OpenXR default joint order. The wire v1 remains unchanged.
// Five metacarpal positions and radii are an anatomical approximation from
// the decoded palm geometry, not additional independent model detections.
struct Joint { std::array<float,7> pose{}; float radius{}; bool inferred{}; };
using Joints = std::array<Joint,26>;
using Vec = std::array<float,3>;
inline Vec sub(Vec a,Vec b){return {a[0]-b[0],a[1]-b[1],a[2]-b[2]};}
inline Vec cross(Vec a,Vec b){return {a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]};}
inline float dot(Vec a,Vec b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
inline Vec scale(Vec a,float s){return {a[0]*s,a[1]*s,a[2]*s};}
inline bool unit(Vec &a){float n=std::sqrt(dot(a,a));if(!std::isfinite(n)||n<1e-6f)return false;a=scale(a,1/n);return true;}
inline std::array<float,4> multiply(std::array<float,4>a,std::array<float,4>b){return {a[3]*b[0]+a[0]*b[3]+a[1]*b[2]-a[2]*b[1],a[3]*b[1]-a[0]*b[2]+a[1]*b[3]+a[2]*b[0],a[3]*b[2]+a[0]*b[1]-a[1]*b[0]+a[2]*b[3],a[3]*b[3]-a[0]*b[0]-a[1]*b[1]-a[2]*b[2]};}
inline Vec rotate(std::array<float,4>q,Vec p){Vec v{q[0],q[1],q[2]};auto a=cross(v,p);auto b=cross(v,a);return {p[0]+2*(q[3]*a[0]+b[0]),p[1]+2*(q[3]*a[1]+b[1]),p[2]+2*(q[3]*a[2]+b[2])};}
inline std::array<float,4> orientation(Vec x,Vec y,Vec z){
    float r[3][3]={{x[0],y[0],z[0]},{x[1],y[1],z[1]},{x[2],y[2],z[2]}};
    std::array<float,4> q{};float tr=r[0][0]+r[1][1]+r[2][2];
    if(tr>0){float s=2*std::sqrt(tr+1);q={(r[2][1]-r[1][2])/s,(r[0][2]-r[2][0])/s,(r[1][0]-r[0][1])/s,s/4};}
    else {unsigned i=r[1][1]>r[0][0]?1:0;if(r[2][2]>r[i][i])i=2;unsigned j=(i+1)%3,k=(i+2)%3;float s=2*std::sqrt(std::max(0.f,1+r[i][i]-r[j][j]-r[k][k]));q[i]=s/4;q[j]=(r[i][j]+r[j][i])/s;q[k]=(r[i][k]+r[k][i])/s;q[3]=(r[k][j]-r[j][k])/s;}
    float n=0;for(float a:q)n+=a*a;for(float &a:q)a/=std::sqrt(n);return q;
}
inline bool joints(const xr_hand_state &h,Joints &out){
    if(h.hand>1 || !(h.flags&XR_HAND_VALID))return false;
    constexpr int landmark[26]={20,5,-1,6,7,0,-1,8,9,10,1,-1,11,12,13,2,-1,14,15,16,3,-1,17,18,19,4};
    Vec p[26]{};for(unsigned j=0;j<26;++j)if(landmark[j]>=0)std::copy_n(h.landmarks[landmark[j]],3,p[j].begin());
    float width=std::sqrt(dot(sub(p[7],p[22]),sub(p[7],p[22])));if(!(width>.025f&&width<.15f))return false;
    // CMC locations reconstructed within the measured palm; do not copy MCP
    // into metacarpal slots or invent independent tracking confidence.
    for(unsigned j:{2u,6u,11u,16u,21u}){auto d=sub(p[j+1],p[1]);p[j]={p[1][0]+d[0]*.28f,p[1][1]+d[1]*.28f,p[1][2]+d[2]*.28f};}
    Vec dorsal=scale(cross(sub(p[7],p[1]),sub(p[22],p[1])),h.hand?1.f:-1.f);if(!unit(dorsal))return false;
    for(unsigned j=0;j<26;++j){
        unsigned next=j<2?12:j==5?5:j==10?10:j==15?15:j==20?20:j==25?25:j+1;
        Vec distal=next==j?sub(p[j],p[j-1]):sub(p[next],p[j]);if(!unit(distal))return false;
        Vec z=scale(distal,-1),x=cross(dorsal,z);if(!unit(x))return false;Vec y=cross(z,x);auto q=orientation(x,y,z);
        auto &v=out[j];std::copy(p[j].begin(),p[j].end(),v.pose.begin());std::copy(q.begin(),q.end(),v.pose.begin()+3);
        v.inferred=landmark[j]<0;v.radius=width*(j<2?.21f:(j==5||j==10||j==15||j==20||j==25)?.075f:.105f);
        for(float f:v.pose)if(!std::isfinite(f))return false;
    }return true;
}
} // namespace xr::hand
