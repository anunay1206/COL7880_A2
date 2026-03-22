#include <stdio.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdlib.h>
#include <string.h>
#include "histogram.cuh"

__global__ void assign_kernel(
    const float* px, const float* py, const float* pz,
    const float* cx, const float* cy, const float* cz,
    int* assignments, int N, int k
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float xi=px[i], yi=py[i], zi=pz[i];
    float best_d = FLT_MAX; int best_c = 0;
    for (int c = 0; c < k; c++) {
        float dx=xi-cx[c], dy=yi-cy[c], dz=zi-cz[c];
        float d = dx*dx+dy*dy+dz*dz;
        if (d < best_d) { best_d=d; best_c=c; }
    }
    assignments[i] = best_c;
}

__global__ void update_kernel(
    const float* px, const float* py, const float* pz,
    const int* assignments,
    float* sum_x, float* sum_y, float* sum_z, int* counts,
    int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int c = assignments[i];
    atomicAdd(&sum_x[c], px[i]);
    atomicAdd(&sum_y[c], py[i]);
    atomicAdd(&sum_z[c], pz[i]);
    atomicAdd(&counts[c], 1);
}

__global__ void kmeans_equalize_kernel(
    const int* intensity,
    const int* assignments,
    const int* sorted_points,
    const int* cluster_start,
    const int* cluster_size,
    int* out_intensity,
    int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    int ci  = assignments[i];
    int cs  = cluster_start[ci];
    int csz = cluster_size[ci];

    int hist[256] = {0};
    for (int j = 0; j < csz; j++) {
        int pt = sorted_points[cs + j];
        hist[intensity[pt]]++;
    }

    int m = csz;

    int cdf[256];
    cdf[0] = hist[0];
    for (int v = 1; v < 256; v++)
        cdf[v] = cdf[v-1] + hist[v];

    int cmin = 0;
    for (int v = 0; v < 256; v++) {
        if (cdf[v] > 0) { cmin = cdf[v]; break; }
    }

    int Ii = intensity[i];
    if (m == cmin)
        out_intensity[i] = Ii;
    else
        out_intensity[i] = (int)((float)(cdf[Ii] - cmin) / (float)(m - cmin) * 255.0f);

    if (out_intensity[i] < 0)   out_intensity[i] = 0;
    if (out_intensity[i] > 255) out_intensity[i] = 255;
}

void run_kmeans(float* x, float* y, float* z, int* intensity,
                int N, int k, int T, int* out_intensity) {

    float *d_x, *d_y, *d_z;
    float *d_cx, *d_cy, *d_cz;
    float *d_sx, *d_sy, *d_sz;
    int   *d_assign, *d_counts, *d_intensity, *d_out;
    int   *d_sorted, *d_cstart, *d_csize;

    cudaMalloc(&d_x,         N * sizeof(float));
    cudaMalloc(&d_y,         N * sizeof(float));
    cudaMalloc(&d_z,         N * sizeof(float));
    cudaMalloc(&d_cx,        k * sizeof(float));
    cudaMalloc(&d_cy,        k * sizeof(float));
    cudaMalloc(&d_cz,        k * sizeof(float));
    cudaMalloc(&d_sx,        k * sizeof(float));
    cudaMalloc(&d_sy,        k * sizeof(float));
    cudaMalloc(&d_sz,        k * sizeof(float));
    cudaMalloc(&d_assign,    N * sizeof(int));
    cudaMalloc(&d_counts,    k * sizeof(int));
    cudaMalloc(&d_intensity, N * sizeof(int));
    cudaMalloc(&d_out,       N * sizeof(int));
    cudaMalloc(&d_sorted,    N * sizeof(int));
    cudaMalloc(&d_cstart,    k * sizeof(int));
    cudaMalloc(&d_csize,     k * sizeof(int));

    cudaMemcpy(d_x,         x,         N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y,         y,         N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_z,         z,         N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_intensity, intensity, N * sizeof(int),   cudaMemcpyHostToDevice);

    // Init centroids = first k points
    cudaMemcpy(d_cx, x, k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cy, y, k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cz, z, k * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks  = (N + threads - 1) / threads;

    int* h_assign      = (int*)malloc(N * sizeof(int));
    int* h_assign_prev = (int*)malloc(N * sizeof(int));
    memset(h_assign_prev, -1, N * sizeof(int));

    for (int iter = 0; iter < T; iter++) {

        // Assign
        assign_kernel<<<blocks, threads>>>(
            d_x, d_y, d_z, d_cx, d_cy, d_cz, d_assign, N, k);
        cudaDeviceSynchronize();

        // Check convergence
        cudaMemcpy(h_assign, d_assign, N * sizeof(int), cudaMemcpyDeviceToHost);
        bool converged = true;
        for (int i = 0; i < N; i++) {
            if (h_assign[i] != h_assign_prev[i]) { converged = false; break; }
        }
        if (converged) {
            printf("  K-means converged at iteration %d\n", iter);
            break;
        }
        memcpy(h_assign_prev, h_assign, N * sizeof(int));

        // Update centroids
        cudaMemset(d_sx,     0, k * sizeof(float));
        cudaMemset(d_sy,     0, k * sizeof(float));
        cudaMemset(d_sz,     0, k * sizeof(float));
        cudaMemset(d_counts, 0, k * sizeof(int));

        update_kernel<<<blocks, threads>>>(
            d_x, d_y, d_z, d_assign, d_sx, d_sy, d_sz, d_counts, N);
        cudaDeviceSynchronize();

        float* h_sx  = (float*)malloc(k * sizeof(float));
        float* h_sy  = (float*)malloc(k * sizeof(float));
        float* h_sz  = (float*)malloc(k * sizeof(float));
        int*   h_cnt = (int*)  malloc(k * sizeof(int));

        cudaMemcpy(h_sx,  d_sx,     k * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_sy,  d_sy,     k * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_sz,  d_sz,     k * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_cnt, d_counts, k * sizeof(int),   cudaMemcpyDeviceToHost);

        for (int c = 0; c < k; c++) {
            if (h_cnt[c] > 0) {
                h_sx[c] /= h_cnt[c];
                h_sy[c] /= h_cnt[c];
                h_sz[c] /= h_cnt[c];
            }
        }

        cudaMemcpy(d_cx, h_sx, k * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_cy, h_sy, k * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_cz, h_sz, k * sizeof(float), cudaMemcpyHostToDevice);

        free(h_sx); free(h_sy); free(h_sz); free(h_cnt);
    }

    // Build sorted cluster list on CPU with OpenMP
    int* cluster_size  = (int*)calloc(k, sizeof(int));
    int* cluster_start = (int*)malloc(k * sizeof(int));

    #pragma omp parallel for
    for (int i = 0; i < N; i++)
        #pragma omp atomic
        cluster_size[h_assign[i]]++;

    cluster_start[0] = 0;
    for (int c = 1; c < k; c++)
        cluster_start[c] = cluster_start[c-1] + cluster_size[c-1];

    int* sorted_pts = (int*)malloc(N * sizeof(int));
    int* fill       = (int*)calloc(k, sizeof(int));
    for (int i = 0; i < N; i++) {
        int c = h_assign[i];
        sorted_pts[cluster_start[c] + fill[c]] = i;
        fill[c]++;
    }
    free(fill);

    // Copy to GPU
    cudaMemcpy(d_sorted, sorted_pts,    N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cstart, cluster_start, k * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csize,  cluster_size,  k * sizeof(int), cudaMemcpyHostToDevice);

    // Equalize
    kmeans_equalize_kernel<<<blocks, threads>>>(
        d_intensity, d_assign, d_sorted, d_cstart, d_csize, d_out, N);
    cudaDeviceSynchronize();

    cudaMemcpy(out_intensity, d_out, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Cleanup
    free(h_assign); free(h_assign_prev);
    free(cluster_size); free(cluster_start); free(sorted_pts);
    cudaFree(d_x);    cudaFree(d_y);    cudaFree(d_z);
    cudaFree(d_cx);   cudaFree(d_cy);   cudaFree(d_cz);
    cudaFree(d_sx);   cudaFree(d_sy);   cudaFree(d_sz);
    cudaFree(d_assign);    cudaFree(d_counts);
    cudaFree(d_intensity); cudaFree(d_out);
    cudaFree(d_sorted);    cudaFree(d_cstart); cudaFree(d_csize);
}