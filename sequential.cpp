#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include <string.h>
#include <time.h>

// ── Lex comparison helper ─────────────────────────────────────────────────────
static inline bool lex_smaller(
    float ax, float ay, float az,
    float bx, float by, float bz
) {
    if (ax != bx) return ax < bx;
    if (ay != by) return ay < by;
    return az < bz;
}

// ── Shared formula: histogram equalization ────────────────────────────────────
void equalize(int* intensity, int* neighbors, int* m_per_point,
              int* out_intensity, int N, int k) {
    for (int i = 0; i < N; i++) {
        int hist[256] = {0};
        int m = m_per_point[i];
        for (int j = 0; j < m; j++)
            hist[intensity[neighbors[i * k + j]]]++;
        int cdf[256];
        cdf[0] = hist[0];
        for (int v = 1; v < 256; v++) cdf[v] = cdf[v-1] + hist[v];
        int cmin = 0;
        for (int v = 0; v < 256; v++) if (cdf[v] > 0) { cmin = cdf[v]; break; }
        int Ii = intensity[i];
        if (m == cmin) out_intensity[i] = Ii;
        else {
            int val = (int)((float)(cdf[Ii] - cmin) / (float)(m - cmin) * 255.0f);
            out_intensity[i] = val < 0 ? 0 : val > 255 ? 255 : val;
        }
    }
}

// ── Sequential KNN ────────────────────────────────────────────────────────────
void seq_knn(float* x, float* y, float* z, int* intensity,
             int N, int k, int* out_intensity) {
    int* neighbors = (int*)malloc(N * k * sizeof(int));
    int* m         = (int*)malloc(N * sizeof(int));
    for (int i = 0; i < N; i++) m[i] = k;

    float* dists = (float*)malloc(N * sizeof(float));
    int*   idx   = (int*)  malloc(N * sizeof(int));

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float dx=x[i]-x[j], dy=y[i]-y[j], dz=z[i]-z[j];
            dists[j] = dx*dx + dy*dy + dz*dz;
            idx[j]   = j;
        }
        dists[i] = FLT_MAX;

        // Selection sort with lex tiebreaking
        for (int a = 0; a < k; a++) {
            int best = a;
            for (int b = a+1; b < N; b++) {
                float da = dists[idx[best]];
                float db = dists[idx[b]];
                bool b_is_better =
                    db < da ||
                    (db == da && lex_smaller(
                        x[idx[b]], y[idx[b]], z[idx[b]],
                        x[idx[best]], y[idx[best]], z[idx[best]]
                    ));
                if (b_is_better) best = b;
            }
            int tmp = idx[a]; idx[a] = idx[best]; idx[best] = tmp;
            neighbors[i*k + a] = idx[a];
        }
    }

    free(dists); free(idx);
    equalize(intensity, neighbors, m, out_intensity, N, k);
    free(neighbors); free(m);
}

// ── Sequential Approximate KNN ────────────────────────────────────────────────
void seq_approx_knn(float* x, float* y, float* z, int* intensity,
                    int N, int k, int* out_intensity) {

    float xmin=FLT_MAX, ymin=FLT_MAX, zmin=FLT_MAX;
    float xmax=-FLT_MAX, ymax=-FLT_MAX, zmax=-FLT_MAX;
    for (int i = 0; i < N; i++) {
        if(x[i]<xmin) xmin=x[i]; if(x[i]>xmax) xmax=x[i];
        if(y[i]<ymin) ymin=y[i]; if(y[i]>ymax) ymax=y[i];
        if(z[i]<zmin) zmin=z[i]; if(z[i]>zmax) zmax=z[i];
    }

    int G = (int)cbrt((double)N / 8.0);
    if (G < 1) G = 1;
    float cell_x = (xmax-xmin)/G + 1e-6f;
    float cell_y = (ymax-ymin)/G + 1e-6f;
    float cell_z = (zmax-zmin)/G + 1e-6f;
    int total_cells = G*G*G;

    int* cell_count = (int*)calloc(total_cells, sizeof(int));
    for (int i = 0; i < N; i++) {
        int gx=(int)((x[i]-xmin)/cell_x); if(gx>=G) gx=G-1;
        int gy=(int)((y[i]-ymin)/cell_y); if(gy>=G) gy=G-1;
        int gz=(int)((z[i]-zmin)/cell_z); if(gz>=G) gz=G-1;
        cell_count[gx*G*G + gy*G + gz]++;
    }

    int* cell_start = (int*)malloc(total_cells*sizeof(int));
    cell_start[0] = 0;
    for (int c = 1; c < total_cells; c++)
        cell_start[c] = cell_start[c-1] + cell_count[c-1];

    int* cell_pts = (int*)malloc(N*sizeof(int));
    int* fill     = (int*)calloc(total_cells, sizeof(int));
    for (int i = 0; i < N; i++) {
        int gx=(int)((x[i]-xmin)/cell_x); if(gx>=G) gx=G-1;
        int gy=(int)((y[i]-ymin)/cell_y); if(gy>=G) gy=G-1;
        int gz=(int)((z[i]-zmin)/cell_z); if(gz>=G) gz=G-1;
        int c = gx*G*G + gy*G + gz;
        cell_pts[cell_start[c] + fill[c]] = i;
        fill[c]++;
    }
    free(fill);

    int* neighbors = (int*)malloc(N*k*sizeof(int));
    int* m_arr     = (int*)malloc(N*sizeof(int));
    float* cdist   = (float*)malloc(N*sizeof(float));
    int*   cidx    = (int*)  malloc(N*sizeof(int));

    for (int i = 0; i < N; i++) {
        int gx=(int)((x[i]-xmin)/cell_x); if(gx>=G) gx=G-1;
        int gy=(int)((y[i]-ymin)/cell_y); if(gy>=G) gy=G-1;
        int gz=(int)((z[i]-zmin)/cell_z); if(gz>=G) gz=G-1;
        float xi=x[i], yi=y[i], zi=z[i];
        int nc=0;

        for (int dx=-1;dx<=1;dx++)
        for (int dy=-1;dy<=1;dy++)
        for (int dz=-1;dz<=1;dz++) {
            int nx=gx+dx, ny=gy+dy, nz=gz+dz;
            if(nx<0||nx>=G||ny<0||ny>=G||nz<0||nz>=G) continue;
            int c=nx*G*G+ny*G+nz;
            for (int jj=0;jj<cell_count[c];jj++) {
                int j=cell_pts[cell_start[c]+jj];
                if(j==i) continue;
                float ddx=xi-x[j],ddy=yi-y[j],ddz=zi-z[j];
                cdist[nc]=ddx*ddx+ddy*ddy+ddz*ddz;
                cidx[nc]=j;
                nc++;
            }
        }

        int take = nc < k ? nc : k;
        // Selection sort with lex tiebreaking
        for (int a=0;a<take;a++) {
            int best=a;
            for (int b=a+1;b<nc;b++) {
                float da = cdist[best];
                float db = cdist[b];
                bool b_is_better =
                    db < da ||
                    (db == da && lex_smaller(
                        x[cidx[b]], y[cidx[b]], z[cidx[b]],
                        x[cidx[best]], y[cidx[best]], z[cidx[best]]
                    ));
                if (b_is_better) best = b;
            }
            float td=cdist[a]; cdist[a]=cdist[best]; cdist[best]=td;
            int   ti=cidx[a];  cidx[a]=cidx[best];   cidx[best]=ti;
        }
        for (int j=0;j<k;j++)
            neighbors[i*k+j] = j < take ? cidx[j] : 0;
        m_arr[i] = k;
    }

    free(cdist); free(cidx);
    free(cell_count); free(cell_start); free(cell_pts);
    equalize(intensity, neighbors, m_arr, out_intensity, N, k);
    free(neighbors); free(m_arr);
}

// ── Sequential K-means ────────────────────────────────────────────────────────
void seq_kmeans(float* x, float* y, float* z, int* intensity,
                int N, int k, int T, int* out_intensity) {

    float* cx = (float*)malloc(k*sizeof(float));
    float* cy = (float*)malloc(k*sizeof(float));
    float* cz = (float*)malloc(k*sizeof(float));
    int* assignments = (int*)malloc(N*sizeof(int));
    int* prev        = (int*)malloc(N*sizeof(int));

    for (int c=0;c<k;c++) { cx[c]=x[c]; cy[c]=y[c]; cz[c]=z[c]; }
    memset(prev, -1, N*sizeof(int));

    for (int iter=0;iter<T;iter++) {
        for (int i=0;i<N;i++) {
            float best=FLT_MAX; int bc=0;
            for (int c=0;c<k;c++) {
                float dx=x[i]-cx[c],dy=y[i]-cy[c],dz=z[i]-cz[c];
                float d=dx*dx+dy*dy+dz*dz;
                if(d<best){best=d;bc=c;}
            }
            assignments[i]=bc;
        }
        int changed=0;
        for (int i=0;i<N;i++) if(assignments[i]!=prev[i]){changed=1;break;}
        if(!changed) break;
        memcpy(prev,assignments,N*sizeof(int));

        float* sx=(float*)calloc(k,sizeof(float));
        float* sy=(float*)calloc(k,sizeof(float));
        float* sz=(float*)calloc(k,sizeof(float));
        int*  cnt=(int*)  calloc(k,sizeof(int));
        for (int i=0;i<N;i++){
            int c=assignments[i];
            sx[c]+=x[i]; sy[c]+=y[i]; sz[c]+=z[i]; cnt[c]++;
        }
        for (int c=0;c<k;c++) if(cnt[c]>0){
            cx[c]=sx[c]/cnt[c]; cy[c]=sy[c]/cnt[c]; cz[c]=sz[c]/cnt[c];
        }
        free(sx);free(sy);free(sz);free(cnt);
    }

    int* cluster_size  = (int*)calloc(k,sizeof(int));
    int* cluster_start = (int*)malloc(k*sizeof(int));
    for (int i=0;i<N;i++) cluster_size[assignments[i]]++;
    cluster_start[0]=0;
    for (int c=1;c<k;c++) cluster_start[c]=cluster_start[c-1]+cluster_size[c-1];
    int* sorted_pts=(int*)malloc(N*sizeof(int));
    int* fill=(int*)calloc(k,sizeof(int));
    for (int i=0;i<N;i++){
        int c=assignments[i];
        sorted_pts[cluster_start[c]+fill[c]]=i;
        fill[c]++;
    }
    free(fill);

    for (int i=0;i<N;i++){
        int ci=assignments[i];
        int cs=cluster_start[ci];
        int csz=cluster_size[ci];
        int hist[256]={0};
        for (int j=0;j<csz;j++) hist[intensity[sorted_pts[cs+j]]]++;
        int cdf[256]; cdf[0]=hist[0];
        for (int v=1;v<256;v++) cdf[v]=cdf[v-1]+hist[v];
        int cmin=0;
        for (int v=0;v<256;v++) if(cdf[v]>0){cmin=cdf[v];break;}
        int Ii=intensity[i];
        if(csz==cmin) out_intensity[i]=Ii;
        else {
            int val=(int)((float)(cdf[Ii]-cmin)/(float)(csz-cmin)*255.0f);
            out_intensity[i]=val<0?0:val>255?255:val;
        }
    }

    free(cx);free(cy);free(cz);
    free(assignments);free(prev);
    free(cluster_size);free(cluster_start);free(sorted_pts);
}

// ── Main ──────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    if (argc < 3) {
        fprintf(stderr, "Usage: ./sequential input.txt output_dir/\n");
        return 1;
    }

    FILE* f = fopen(argv[1], "r");
    if (!f) { fprintf(stderr, "Cannot open %s\n", argv[1]); return 1; }

    int N, k, T;
    (void)fscanf(f, "%d %d %d", &N, &k, &T);

    float* x        = (float*)malloc(N*sizeof(float));
    float* y        = (float*)malloc(N*sizeof(float));
    float* z        = (float*)malloc(N*sizeof(float));
    int*   intensity = (int*)  malloc(N*sizeof(int));

    for (int i=0;i<N;i++)
        (void)fscanf(f, "%f %f %f %d", &x[i], &y[i], &z[i], &intensity[i]);
    fclose(f);

    int* out_knn    = (int*)malloc(N*sizeof(int));
    int* out_approx = (int*)malloc(N*sizeof(int));
    int* out_kmeans = (int*)malloc(N*sizeof(int));

    char outdir[512];
    snprintf(outdir, sizeof(outdir), "%s", argv[2]);

    struct timespec t0, t1;
    printf("N=%d  k=%d  T=%d\n\n", N, k, T);

    clock_gettime(CLOCK_MONOTONIC, &t0);
    seq_knn(x, y, z, intensity, N, k, out_knn);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double knn_ms = (t1.tv_sec-t0.tv_sec)*1000.0 + (t1.tv_nsec-t0.tv_nsec)/1e6;
    printf("  Sequential KNN        : %.3f ms\n", knn_ms);

    clock_gettime(CLOCK_MONOTONIC, &t0);
    seq_approx_knn(x, y, z, intensity, N, k, out_approx);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double approx_ms = (t1.tv_sec-t0.tv_sec)*1000.0 + (t1.tv_nsec-t0.tv_nsec)/1e6;
    printf("  Sequential Approx KNN : %.3f ms\n", approx_ms);

    clock_gettime(CLOCK_MONOTONIC, &t0);
    seq_kmeans(x, y, z, intensity, N, k, T, out_kmeans);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double kmeans_ms = (t1.tv_sec-t0.tv_sec)*1000.0 + (t1.tv_nsec-t0.tv_nsec)/1e6;
    printf("  Sequential K-means    : %.3f ms\n", kmeans_ms);

    char path[1024];
    auto write_file = [&](const char* name, int* out) {
        snprintf(path, sizeof(path), "%s/%s", outdir, name);
        FILE* fo = fopen(path, "w");
        for (int i=0;i<N;i++)
            fprintf(fo, "%f %f %f %d\n", x[i], y[i], z[i], out[i]);
        fclose(fo);
    };
    write_file("knn.txt",        out_knn);
    write_file("approx_knn.txt", out_approx);
    write_file("kmeans.txt",     out_kmeans);

    snprintf(path, sizeof(path), "%s/time.txt", outdir);
    FILE* ft = fopen(path, "w");
    fprintf(ft, "Sequential KNN        : %.3f ms\n", knn_ms);
    fprintf(ft, "Sequential Approx KNN : %.3f ms\n", approx_ms);
    fprintf(ft, "Sequential K-means    : %.3f ms\n", kmeans_ms);
    fclose(ft);

    printf("\nOutputs written to %s/\n", outdir);

    free(x);free(y);free(z);free(intensity);
    free(out_knn);free(out_approx);free(out_kmeans);
    return 0;
}