#include <cuda_runtime.h>
#include <float.h>
#include <omp.h>
#include <algorithm>
#include <vector>
#include <cmath>
#include <string.h>
#include "histogram.cuh"

void run_approx_knn(float* x, float* y, float* z, int* intensity,
                    int N, int k, int* out_intensity) {

    // Bounding box with OpenMP
    float xmin=FLT_MAX, ymin=FLT_MAX, zmin=FLT_MAX;
    float xmax=-FLT_MAX, ymax=-FLT_MAX, zmax=-FLT_MAX;
    #pragma omp parallel for reduction(min:xmin,ymin,zmin) reduction(max:xmax,ymax,zmax)
    for (int i = 0; i < N; i++) {
        if(x[i]<xmin) xmin=x[i]; if(x[i]>xmax) xmax=x[i];
        if(y[i]<ymin) ymin=y[i]; if(y[i]>ymax) ymax=y[i];
        if(z[i]<zmin) zmin=z[i]; if(z[i]>zmax) zmax=z[i];
    }

    // Grid setup
    int G = (int)cbrt((double)N / 8.0);
    G = std::max(G, 1);

    float cell_x = (xmax - xmin) / G + 1e-6f;
    float cell_y = (ymax - ymin) / G + 1e-6f;
    float cell_z = (zmax - zmin) / G + 1e-6f;
    int total_cells = G * G * G;

    auto get_gx = [&](int i){ return std::min((int)((x[i]-xmin)/cell_x), G-1); }; // Inside column along x axis
    auto get_gy = [&](int i){ return std::min((int)((y[i]-ymin)/cell_y), G-1); };
    auto get_gz = [&](int i){ return std::min((int)((z[i]-zmin)/cell_z), G-1); };
    auto cell_id = [&](int i){ return get_gx(i)*G*G + get_gy(i)*G + get_gz(i); };

    // Build flat cell arrays
    int* cell_count = (int*)calloc(total_cells, sizeof(int));
    for (int i = 0; i < N; i++) cell_count[cell_id(i)]++;

    int* cell_start = (int*)malloc(total_cells * sizeof(int));
    cell_start[0] = 0;
    for (int c = 1; c < total_cells; c++)
        cell_start[c] = cell_start[c-1] + cell_count[c-1];

    int* cell_pts = (int*)malloc(N * sizeof(int));
    int* fill     = (int*)calloc(total_cells, sizeof(int));
    for (int i = 0; i < N; i++) {
        int c = cell_id(i);
        cell_pts[cell_start[c] + fill[c]] = i;
        fill[c]++;
    }
    free(fill);

    // For each point find k nearest using flat arrays
    int* neighbors = (int*)malloc(N * k * sizeof(int));

    #pragma omp parallel
    {
        // Per-thread scratch buffers allocated once
        float* cdist = (float*)malloc(N * sizeof(float));
        int*   cidx  = (int*)  malloc(N * sizeof(int));

        #pragma omp for schedule(dynamic, 128)
        for (int i = 0; i < N; i++) {
            int gx = get_gx(i), gy = get_gy(i), gz = get_gz(i);
            float xi = x[i], yi = y[i], zi = z[i];

            int nc = 0;

            // Search 3x3x3 neighboring cells
            for (int dx = -1; dx <= 1; dx++)
            for (int dy = -1; dy <= 1; dy++)
            for (int dz = -1; dz <= 1; dz++) {
                int nx = gx+dx, ny = gy+dy, nz = gz+dz;
                if (nx<0||nx>=G||ny<0||ny>=G||nz<0||nz>=G) continue;
                int c   = nx*G*G + ny*G + nz;
                int cs  = cell_start[c];
                int cnt = cell_count[c];
                for (int jj = 0; jj < cnt; jj++) {
                    int j = cell_pts[cs + jj];
                    if (j == i) continue;
                    float ddx=xi-x[j], ddy=yi-y[j], ddz=zi-z[j];
                    cdist[nc] = ddx*ddx + ddy*ddy + ddz*ddz;
                    cidx [nc] = j;
                    nc++;
                }
            }

            // Pack into pairs and nth_element O(nc)
            std::vector<std::pair<float,int>> tmp(nc);
            for (int jj = 0; jj < nc; jj++)
                tmp[jj] = {cdist[jj], cidx[jj]};

            int take = std::min(nc, k);
            if (nc > k)
                std::nth_element(tmp.begin(), tmp.begin()+k, tmp.end());

            for (int j = 0; j < k; j++)
                neighbors[i*k+j] = (j < take) ? tmp[j].second : 0;
        }

        free(cdist);
        free(cidx);
    }

    free(cell_count); free(cell_start); free(cell_pts);

    // GPU histogram equalization
    int *d_intensity, *d_neighbors, *d_out, *d_m;
    cudaMalloc(&d_intensity, N*sizeof(int));
    cudaMalloc(&d_neighbors, N*k*sizeof(int));
    cudaMalloc(&d_out,       N*sizeof(int));
    cudaMalloc(&d_m,         N*sizeof(int));

    cudaMemcpy(d_intensity, intensity, N*sizeof(int),   cudaMemcpyHostToDevice);
    cudaMemcpy(d_neighbors, neighbors, N*k*sizeof(int), cudaMemcpyHostToDevice);

    int* h_m = (int*)malloc(N*sizeof(int));
    for (int i = 0; i < N; i++) h_m[i] = k;
    cudaMemcpy(d_m, h_m, N*sizeof(int), cudaMemcpyHostToDevice);
    free(h_m); free(neighbors);

    int threads=256, blocks=(N+255)/256;
    histogram_equalize_kernel<<<blocks,threads>>>(
        d_intensity, d_neighbors, d_m, d_out, N, k);
    cudaDeviceSynchronize();

    cudaMemcpy(out_intensity, d_out, N*sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_intensity); cudaFree(d_neighbors);
    cudaFree(d_out); cudaFree(d_m);
}