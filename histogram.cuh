#pragma once
#include <cuda_runtime.h>

// Given neighbor indices for each point, compute equalized intensity.
// neighbors: [N * k] array — row i has k neighbor indices for point i
// m_per_point: neighborhood size per point (k for KNN, cluster size for kmeans)
static __global__ void histogram_equalize_kernel(
    const int* __restrict__ intensity,   // original intensities [N]
    const int* __restrict__ neighbors,   // neighbor indices [N * k]
    const int* __restrict__ m_per_point, // neighborhood size per point [N]
    int* __restrict__ out_intensity,     // output [N]
    int N, int k
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    int hist[256] = {0};
    int m = m_per_point[i]; // for KNN this is always k, for K-means this is cluster size
 
    // Build local histogram
    for (int j = 0; j < m; j++) {
        int nb = neighbors[i * k + j];
        hist[intensity[nb]]++;
    }

    // CDF
    int cdf[256];
    cdf[0] = hist[0];
    for (int v = 1; v < 256; v++)
        cdf[v] = cdf[v-1] + hist[v];

    // Find cmin (first nonzero CDF value)
    int cmin = 0;
    for (int v = 0; v < 256; v++) {
        if (cdf[v] > 0) { cmin = cdf[v]; break; }
    }

    int Ii = intensity[i];
    if (m == cmin) {
        out_intensity[i] = Ii;
    } else {
        float val = (float)(cdf[Ii] - cmin) / (float)(m - cmin) * 255.0f;
        out_intensity[i] = (int)(val);  // floor — just truncate
        if (out_intensity[i] > 255) out_intensity[i] = 255;
        if (out_intensity[i] < 0)   out_intensity[i] = 0;
    }
}