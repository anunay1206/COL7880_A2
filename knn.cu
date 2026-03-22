#include <cuda_runtime.h>
#include <float.h>
#include "histogram.cuh"

#define TILE 256

// Lexicographic comparison — returns true if (ax,ay,az) < (bx,by,bz)
__device__ bool lex_smaller(
    float ax, float ay, float az,
    float bx, float by, float bz
) {
    if (ax != bx) return ax < bx;
    if (ay != by) return ay < by;
    return az < bz;
}

__global__ void knn_kernel(
    const float* __restrict__ px,
    const float* __restrict__ py,
    const float* __restrict__ pz,
    int* __restrict__ neighbors,
    int N, int k
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    float xi = 0, yi = 0, zi = 0;
    if (i < N) { xi = px[i]; yi = py[i]; zi = pz[i]; }

    // Max-heap for k nearest — also store coordinates for tiebreaking
    float heap_dist[128];
    int   heap_idx [128];
    float heap_x   [128];
    float heap_y   [128];
    float heap_z   [128];
    int   heap_size = 0;

    __shared__ float sx[TILE], sy[TILE], sz[TILE];
    __shared__ int   si[TILE];

    int num_tiles = (N + TILE - 1) / TILE;

    for (int t = 0; t < num_tiles; t++) {

        // Cooperatively load tile
        int load_idx = t * TILE + tid;
        if (load_idx < N) {
            sx[tid] = px[load_idx];
            sy[tid] = py[load_idx];
            sz[tid] = pz[load_idx];
            si[tid] = load_idx;
        } else {
            sx[tid] = 0; sy[tid] = 0; sz[tid] = 0;
            si[tid] = -1;
        }
        __syncthreads();

        if (i < N) {
            int tile_end = min(TILE, N - t * TILE);
            for (int j = 0; j < tile_end; j++) {
                int jj = si[j];
                if (jj == i) continue;

                float dx = xi - sx[j];
                float dy = yi - sy[j];
                float dz = zi - sz[j];
                float d2 = dx*dx + dy*dy + dz*dz;

                if (heap_size < k) {
                    // Heap not full — just insert
                    heap_dist[heap_size] = d2;
                    heap_idx [heap_size] = jj;
                    heap_x   [heap_size] = sx[j];
                    heap_y   [heap_size] = sy[j];
                    heap_z   [heap_size] = sz[j];
                    heap_size++;

                    // Sift up — max-heap by distance, lex tiebreak
                    int pos = heap_size - 1;
                    while (pos > 0) {
                        int parent = (pos-1)/2;
                        bool should_swap =
                            heap_dist[parent] < heap_dist[pos] ||
                            (heap_dist[parent] == heap_dist[pos] &&
                             lex_smaller(heap_x[pos], heap_y[pos], heap_z[pos],
                                         heap_x[parent], heap_y[parent], heap_z[parent]));
                        if (should_swap) {
                            float td = heap_dist[parent]; heap_dist[parent] = heap_dist[pos]; heap_dist[pos] = td;
                            int   ti = heap_idx [parent]; heap_idx [parent] = heap_idx [pos]; heap_idx [pos] = ti;
                            float tx = heap_x[parent]; heap_x[parent] = heap_x[pos]; heap_x[pos] = tx;
                            float ty = heap_y[parent]; heap_y[parent] = heap_y[pos]; heap_y[pos] = ty;
                            float tz = heap_z[parent]; heap_z[parent] = heap_z[pos]; heap_z[pos] = tz;
                            pos = parent;
                        } else break;
                    }
                } else {
                    // Heap full — check if new point beats the root (worst neighbor)
                    bool should_replace =
                        d2 < heap_dist[0] ||
                        (d2 == heap_dist[0] &&
                         lex_smaller(sx[j], sy[j], sz[j],
                                     heap_x[0], heap_y[0], heap_z[0]));

                    if (should_replace) {
                        heap_dist[0] = d2;
                        heap_idx [0] = jj;
                        heap_x   [0] = sx[j];
                        heap_y   [0] = sy[j];
                        heap_z   [0] = sz[j];

                        // Sift down
                        int pos = 0;
                        while (true) {
                            int left=2*pos+1, right=2*pos+2, largest=pos;

                            if (left < k) {
                                bool left_larger =
                                    heap_dist[left] > heap_dist[largest] ||
                                    (heap_dist[left] == heap_dist[largest] &&
                                     lex_smaller(heap_x[largest], heap_y[largest], heap_z[largest],
                                                 heap_x[left], heap_y[left], heap_z[left]));
                                if (left_larger) largest = left;
                            }

                            if (right < k) {
                                bool right_larger =
                                    heap_dist[right] > heap_dist[largest] ||
                                    (heap_dist[right] == heap_dist[largest] &&
                                     lex_smaller(heap_x[largest], heap_y[largest], heap_z[largest],
                                                 heap_x[right], heap_y[right], heap_z[right]));
                                if (right_larger) largest = right;
                            }

                            if (largest == pos) break;

                            float td = heap_dist[pos]; heap_dist[pos] = heap_dist[largest]; heap_dist[largest] = td;
                            int   ti = heap_idx [pos]; heap_idx [pos] = heap_idx [largest]; heap_idx [largest] = ti;
                            float tx = heap_x[pos]; heap_x[pos] = heap_x[largest]; heap_x[largest] = tx;
                            float ty = heap_y[pos]; heap_y[pos] = heap_y[largest]; heap_y[largest] = ty;
                            float tz = heap_z[pos]; heap_z[pos] = heap_z[largest]; heap_z[largest] = tz;
                            pos = largest;
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

    if (i < N)
        for (int j = 0; j < k; j++)
            neighbors[i * k + j] = heap_idx[j];
}

void run_knn(float* x, float* y, float* z, int* intensity,
             int N, int k, int* out_intensity) {
    float *d_x, *d_y, *d_z;
    int   *d_intensity, *d_neighbors, *d_out, *d_m;

    cudaMalloc(&d_x,         N * sizeof(float));
    cudaMalloc(&d_y,         N * sizeof(float));
    cudaMalloc(&d_z,         N * sizeof(float));
    cudaMalloc(&d_intensity, N * sizeof(int));
    cudaMalloc(&d_neighbors, N * k * sizeof(int));
    cudaMalloc(&d_out,       N * sizeof(int));
    cudaMalloc(&d_m,         N * sizeof(int));

    cudaMemcpy(d_x,         x,         N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y,         y,         N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_z,         z,         N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_intensity, intensity, N*sizeof(int),   cudaMemcpyHostToDevice);

    int* h_m = (int*)malloc(N * sizeof(int));
    for (int i = 0; i < N; i++) h_m[i] = k;
    cudaMemcpy(d_m, h_m, N*sizeof(int), cudaMemcpyHostToDevice);
    free(h_m);

    int threads = TILE;
    int blocks  = (N + threads - 1) / threads;

    knn_kernel<<<blocks, threads>>>(d_x, d_y, d_z, d_neighbors, N, k);
    cudaDeviceSynchronize();

    histogram_equalize_kernel<<<blocks, threads>>>(
        d_intensity, d_neighbors, d_m, d_out, N, k);
    cudaDeviceSynchronize();

    cudaMemcpy(out_intensity, d_out, N*sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_x); cudaFree(d_y); cudaFree(d_z);
    cudaFree(d_intensity); cudaFree(d_neighbors);
    cudaFree(d_out); cudaFree(d_m);
}