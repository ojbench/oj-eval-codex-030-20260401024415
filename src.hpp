// Heuristic digit recognizer for 28x28 grayscale images (values in [0,1])
// Implements judge(IMAGE_T&) returning 0..9.
#pragma once
#include <vector>
#include <queue>
#include <cmath>
#include <algorithm>

typedef std::vector<std::vector<double>> IMAGE_T;

namespace nr_heur {
struct BBox { int r0=0,c0=0,r1=0,c1=0; int h() const { return r1-r0+1; } int w() const { return c1-c0+1; } };

static inline void binarize(const IMAGE_T &img, std::vector<std::vector<unsigned char>> &bin,
                            BBox &box, double thresh=0.5) {
    int n = (int)img.size();
    int m = n ? (int)img[0].size() : 0;
    bin.assign(n, std::vector<unsigned char>(m, 0));
    int r0=n, c0=m, r1=-1, c1=-1;
    for (int i=0;i<n;i++) {
        for (int j=0;j<m;j++) {
            unsigned char v = (unsigned char)(img[i][j] > thresh);
            bin[i][j] = v;
            if (v) {
                if (i<r0) r0=i; if (i>r1) r1=i; if (j<c0) c0=j; if (j>c1) c1=j;
            }
        }
    }
    if (r1==-1) { // no foreground
        box = {0,0,0,0};
    } else {
        box = {r0,c0,r1,c1};
    }
}

static inline int count_white(const std::vector<std::vector<unsigned char>> &bin, const BBox &b) {
    int cnt=0; for(int i=b.r0;i<=b.r1;i++) for(int j=b.c0;j<=b.c1;j++) if(bin[i][j]) cnt++; return cnt;
}

static inline int count_holes_and_center(const std::vector<std::vector<unsigned char>> &bin, const BBox &b, double &hole_cy_rel) {
    int h=b.h(), w=b.w();
    if (h<=0 || w<=0) { hole_cy_rel=0.5; return 0; }
    std::vector<std::vector<int>> vis(h, std::vector<int>(w,0));
    auto inside=[&](int r,int c){return r>=0&&r<h&&c>=0&&c<w;};
    auto bg=[&](int r,int c){return bin[b.r0+r][b.c0+c]==0;};
    // mark border-connected background
    std::queue<std::pair<int,int>> q;
    auto push_if=[&](int r,int c){ if(inside(r,c)&&!vis[r][c]&&bg(r,c)){ vis[r][c]=1; q.push({r,c}); }};
    for(int i=0;i<h;i++){ push_if(i,0); push_if(i,w-1);} for(int j=0;j<w;j++){ push_if(0,j); push_if(h-1,j);}    
    const int dr[4]={1,-1,0,0}, dc[4]={0,0,1,-1};
    while(!q.empty()){
        auto [r,c]=q.front(); q.pop();
        for(int k=0;k<4;k++){ int nr=r+dr[k], nc=c+dc[k]; if(inside(nr,nc)&&!vis[nr][nc]&&bg(nr,nc)){vis[nr][nc]=1; q.push({nr,nc});}}
    }
    int holes=0; long long sum_r=0, sum_sz=0;
    for(int i=0;i<h;i++){
        for(int j=0;j<w;j++){
            if(!vis[i][j] && bg(i,j)){
                holes++;
                // flood this hole to compute centroid y
                std::queue<std::pair<int,int>> q2; vis[i][j]=1; q2.push({i,j}); long long cnt=0, acc_r=0;
                while(!q2.empty()){
                    auto [r,c]=q2.front(); q2.pop(); cnt++; acc_r+=r;
                    for(int k=0;k<4;k++){ int nr=r+dr[k], nc=c+dc[k]; if(inside(nr,nc)&&!vis[nr][nc]&&bg(nr,nc)){vis[nr][nc]=1; q2.push({nr,nc});}}
                }
                sum_r += acc_r; sum_sz += cnt;
            }
        }
    }
    hole_cy_rel = (sum_sz>0) ? (double)sum_r/(double)sum_sz / (double)h : 0.5;
    return holes;
}

static inline void projections(const std::vector<std::vector<unsigned char>> &bin, const BBox &b,
                               std::vector<int> &row, std::vector<int> &col) {
    int h=b.h(), w=b.w(); row.assign(h,0); col.assign(w,0);
    for(int i=b.r0;i<=b.r1;i++){
        for(int j=b.c0;j<=b.c1;j++){
            if(bin[i][j]){ row[i-b.r0]++; col[j-b.c0]++; }
        }
    }
}

static inline void thirds(const std::vector<int> &arr, double &t, double &m, double &b) {
    int n=(int)arr.size(); int a=n/3, c=(2*n)/3; long long s1=0,s2=0,s3=0; for(int i=0;i<n;i++){ if(i<a) s1+=arr[i]; else if(i<c) s2+=arr[i]; else s3+=arr[i]; }
    t=s1; m=s2; b=s3;
}

static inline void thirds_col(const std::vector<int> &arr, double &l, double &m, double &r) {
    int n=(int)arr.size(); int a=n/3, c=(2*n)/3; long long s1=0,s2=0,s3=0; for(int i=0;i<n;i++){ if(i<a) s1+=arr[i]; else if(i<c) s2+=arr[i]; else s3+=arr[i]; }
    l=s1; m=s2; r=s3;
}
}

int judge(IMAGE_T &img) {
    using namespace nr_heur;
    std::vector<std::vector<unsigned char>> bin;
    BBox box; binarize(img, bin, box, 0.5);
    int n = (int)img.size(); if(n==0) return 0;
    if (box.r1 < box.r0 || box.c1 < box.c0) return 0;
    int h = box.h(), w = box.w();
    int white = count_white(bin, box);
    if (white == 0) return 0;

    // projections and coarse features
    std::vector<int> row, col; projections(bin, box, row, col);
    double top, midr, bot; thirds(row, top, midr, bot);
    double lef, midc, rig; thirds_col(col, lef, midc, rig);
    double total = top+midr+bot;
    double hole_cy_rel=0.5; int holes = count_holes_and_center(bin, box, hole_cy_rel);

    // Two holes -> 8
    if (holes >= 2) return 8;

    // One hole: 0/6/9
    if (holes == 1) {
        // Vertical distribution hint
        if (bot > top * 1.25 && hole_cy_rel > 0.55) return 6;
        if (top > bot * 1.25 && hole_cy_rel < 0.45) return 9;
        // Symmetry and central mass suggests 0
        double lr_ratio = (rig+1) / (lef+1);
        double tb_diff = std::fabs(bot - top) / (total+1e-9);
        if (tb_diff < 0.08 && lr_ratio < 1.2 && lr_ratio > 0.8) return 0;
        // Fallback by which half heavier
        return (bot >= top) ? 6 : 9;
    }

    // No holes: 1/2/3/4/5/7
    // Detect 1: tall and narrow, strong central column
    int max_col_idx = (int)(std::max_element(col.begin(), col.end()) - col.begin());
    double ar = (double)h / (double)w;
    if (ar > 1.8 && w < n*0.8) {
        int center_band_l = std::max(0, w/2 - std::max(1,w/10));
        int center_band_r = std::min(w-1, w/2 + std::max(1,w/10));
        long long center_sum=0; for(int j=center_band_l;j<=center_band_r;j++) center_sum += col[j];
        if (center_sum > total * 0.5) return 1;
    }

    // Detect 7: very top-heavy and right-heavy
    if (top > total * 0.42 && bot < total * 0.25 && rig > lef * 1.15) return 7;

    // Detect 4: strong mid band and right column heavy, top relatively light
    if (midr > top * 0.9 && rig > lef * 1.1 && top < total * 0.35) return 4;

    // Distinguish 2/3/5 using quadrants heuristics
    // Compute coarse 3x3 grid sums
    int rcut1 = box.r0 + h/3, rcut2 = box.r0 + (2*h)/3;
    int ccut1 = box.c0 + w/3, ccut2 = box.c0 + (2*w)/3;
    long long g[3][3] = {{0}};
    for (int i=box.r0;i<=box.r1;i++){
        for (int j=box.c0;j<=box.c1;j++) if (bin[i][j]){
            int ri = (i<rcut1)?0:((i<rcut2)?1:2);
            int ci = (j<ccut1)?0:((j<ccut2)?1:2);
            g[ri][ci]++;
        }
    }
    long long top_row = g[0][0]+g[0][1]+g[0][2];
    long long mid_row = g[1][0]+g[1][1]+g[1][2];
    long long bot_row = g[2][0]+g[2][1]+g[2][2];
    long long left_col = g[0][0]+g[1][0]+g[2][0];
    long long right_col = g[0][2]+g[1][2]+g[2][2];
    long long bl = g[2][0], br = g[2][2], tl = g[0][0], tr = g[0][2];

    // 3: right side dominates strongly
    if (right_col > left_col * 1.6 && bot_row > total * 0.25) return 3;

    // 2: top heavy and bottom-right heavy
    if (top_row > bot_row * 0.9 && br > bl * 1.3 && right_col >= left_col) return 2;

    // 5: top heavy and bottom-left heavy
    if (top_row > bot_row * 0.9 && bl > br * 1.2 && left_col >= right_col*0.9) return 5;

    // Fallbacks based on simple dominance
    if (ar > 1.3 && w < h) return 1;
    if (right_col > left_col * 1.2) return 3;
    if (bl > br) return 5;
    if (br >= bl) return 2;
    return 4;
}

