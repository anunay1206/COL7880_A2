#include <iostream>
#include <vector>
#include <queue>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <omp.h>

using namespace std;
using namespace chrono;


#define BLOCK_SIZE 128
#define TILE_SIZE 256
#define K_SIZE 128

__device__ inline bool isLexSmallerIdx(
    int a, int b,
    const int *x, const int *y, const int *z)
{
    if (x[a] != x[b]) return x[a] < x[b];
    if (y[a] != y[b]) return y[a] < y[b];
    return z[a] < z[b];
}

__device__ bool isLexSmaller(int ax, int ay, int az,
                            int bx, int by, int bz) {
    if (ax != bx) return ax < bx;
    if (ay != by) return ay < by;
    return az < bz;
}

__global__ void gpuKnn(int *x, int *y, int *z, uint8_t *I, uint8_t *minKIntensities, int n, int k) {

    // shared memory is 32 KB
    // Register memory : 256KB available

    int globalIndex = threadIdx.x + blockDim.x * blockIdx.x;

    // this uses (4 * 256 * 3) + 256 => 3KB (29KB Available)
    __shared__ int s_x[TILE_SIZE];
    __shared__ int s_y[TILE_SIZE];
    __shared__ int s_z[TILE_SIZE];

    // 4B * K_SIZE * BLOCK_SIZE (4 * 128 * 256) => 128 KB (128 KB available)
    float minK[K_SIZE];
    //  4B * K_SIZE * BLOCK_SIZE (4 * 128 * 256) => 128 KB (0 KB available)
    int minIndex[K_SIZE];
    int xlocal[K_SIZE], ylocal[K_SIZE], zlocal[K_SIZE];

    // register spilling will happen O(-_-)O
    if(globalIndex >= n) return;

    // 4 * 7 + 1 = 29B
    int gx = x[globalIndex];
    int gy = y[globalIndex];
    int gz = z[globalIndex];
    uint8_t gi = I[globalIndex];

    int currentIndex   = 0;
    float maxElement   = 0.0f;
    int maxElementIndex = 0;
    int numTiles = (n + TILE_SIZE - 1) / TILE_SIZE;

    // I dont think Register spilling will happen now
    for(int t = 0; t < numTiles; t++) {

        int tileStart = t * TILE_SIZE;

        s_x[threadIdx.x] = (tileStart + threadIdx.x < n) ? x[tileStart + threadIdx.x] : 0;
        s_y[threadIdx.x] = (tileStart + threadIdx.x < n) ? y[tileStart + threadIdx.x] : 0;
        s_z[threadIdx.x] = (tileStart + threadIdx.x < n) ? z[tileStart + threadIdx.x] : 0;

        __syncthreads();

        for(int local_j = 0; local_j < TILE_SIZE; local_j++) {

            int globalJ = tileStart + local_j;
            if(globalJ >= n) break;
            if(globalJ == globalIndex) continue;

            int dx = gx - s_x[local_j];
            int dy = gy - s_y[local_j];
            int dz = gz - s_z[local_j];
            float d = (float)(dx*dx + dy*dy + dz*dz);

            if(currentIndex < k) {
                minK[currentIndex] = d;
                minIndex[currentIndex] = globalJ;
                xlocal[currentIndex] = s_x[local_j];
                ylocal[currentIndex] = s_y[local_j];
                zlocal[currentIndex] = s_z[local_j];
                currentIndex++;

                if(currentIndex == k) {
                    // initial full scan once
                    maxElement = minK[0];
                    maxElementIndex = 0;
                    for(int t2 = 1; t2 < k; t2++) {
                        if (
                            minK[t2] > maxElement ||
                            (minK[t2] == maxElement &&
                            isLexSmaller(
                                xlocal[maxElementIndex],
                                ylocal[maxElementIndex],
                                zlocal[maxElementIndex],
                                xlocal[t2],
                                ylocal[t2],
                                zlocal[t2]
                            ))
                        ) {
                            maxElement = minK[t2];
                            maxElementIndex = t2;
                        }
                    }
                }
            }
            else {
                // check if candidate is better than current max
                if (
                    d < maxElement ||
                    (d == maxElement &&
                    isLexSmaller(
                        s_x[local_j], s_y[local_j], s_z[local_j],
                        xlocal[maxElementIndex],
                        ylocal[maxElementIndex],
                        zlocal[maxElementIndex]
                    ))
                ) {
                    // store old max index
                    int replacedIndex = maxElementIndex;

                    // replace
                    minK[replacedIndex] = d;
                    minIndex[replacedIndex] = globalJ;
                    xlocal[replacedIndex] = s_x[local_j];
                    ylocal[replacedIndex] = s_y[local_j];
                    zlocal[replacedIndex] = s_z[local_j];
                    maxElement = minK[0];
                    maxElementIndex = 0;

                    for(int t2 = 1; t2 < k; t2++) {
                        if (
                            minK[t2] > maxElement ||
                            (minK[t2] == maxElement &&
                            isLexSmaller(
                                xlocal[maxElementIndex],
                                ylocal[maxElementIndex],
                                zlocal[maxElementIndex],
                                xlocal[t2],
                                ylocal[t2],
                                zlocal[t2]
                            ))
                        ) {
                            maxElement = minK[t2];
                            maxElementIndex = t2;
                        }
                    }
                }
            }
        }

        __syncthreads();
    }

    int less = 0;
    int equal = 0;
    uint8_t minVal = 255;
    int minCount = 0;

    for(int t = 0; t < k; t++) {
        uint8_t val = I[minIndex[t]];

        if(val < gi) less++;
        else if(val == gi) equal++;

        if(val < minVal) {
            minVal = val;
            minCount = 1;
        } else if(val == minVal) {
            minCount++;
        }
    }

    int cdf_val = less + equal;
    int cdf_min = minCount;
    uint8_t newIntensity;
    if(k <= cdf_min) {
        newIntensity = gi;
    } 
    else {
        float val = ((float)(cdf_val - cdf_min)) / (k - cdf_min);
        newIntensity = (uint8_t)(val * 255.0f);
    }

    minKIntensities[globalIndex] = newIntensity;

}

int main(int argc, char *argv[]) {

    if(argc < 2) {
        cerr << "Usage: " << argv[0] << " <input_file>\n";
        return 1;
    }

    ifstream fin(argv[1]);
    if(!fin.is_open()) {
        cerr << "Error opening file\n";
        return 1;
    }

    int n, k, t;
    fin >> n >> k >> t;

    vector<int> x(n),y(n),z(n);
    vector<int> I2(n), newI(n);
    int *gpuX = new int[n];
    int *gpuY = new int[n];
    int *gpuZ = new int[n];
    uint8_t *gpuI = new uint8_t[n];
    uint8_t *gpunewI = new uint8_t[n];

    for(int i = 0; i < n; i++) {
        int temp;
        fin >> gpuX[i]
            >> gpuY[i]
            >> gpuZ[i]
            >> temp;
        gpuI[i] = (uint8_t)temp;
    }

    fin.close();


    int *d_x, *d_y, *d_z;
    uint8_t *d_I, *d_newI;
    auto start = high_resolution_clock::now();
    // Allocate device memory
    cudaMalloc((void**)&d_x, n * sizeof(int));
    cudaMalloc((void**)&d_y, n * sizeof(int));
    cudaMalloc((void**)&d_z, n * sizeof(int));
    cudaMalloc((void**)&d_I, n * sizeof(uint8_t));
    cudaMalloc((void**)&d_newI, n * sizeof(uint8_t));

    // Copy data from host → device
    cudaMemcpy(d_x, gpuX, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, gpuY, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_z, gpuZ, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_I, gpuI, n * sizeof(uint8_t), cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = BLOCK_SIZE;
    int numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    gpuKnn<<<numBlocks, threadsPerBlock>>>(d_x, d_y, d_z, d_I, d_newI, n, k);

    // Wait for GPU to finish
    cudaDeviceSynchronize();
    // Copy result back to host
    cudaMemcpy(gpunewI, d_newI, n * sizeof(uint8_t), cudaMemcpyDeviceToHost);


    // Free device memory
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
    cudaFree(d_I);
    cudaFree(d_newI);


    // Write knn.txt
    ofstream fout("knn.txt");
    if (!fout.is_open()) {
        cerr << "Error opening output file\n";
        return 1;
    }

    for (int i = 0; i < n; i++) {
        fout << gpuX[i] << " "
            << gpuY[i] << " "
            << gpuZ[i] << " "
            << (int)gpunewI[i] << "\n";
    }

    fout.close();

    auto end = high_resolution_clock::now();
    double opTime = duration<double, milli>(end - start).count();

    ofstream fout2("time.txt");
    if (!fout2.is_open()) {
        cerr << "Error opening output file\n";
        return 1;
    }
    fout2 << "KNN : " << opTime << endl;
    fout2.close();
    // printPoints(points_aknn);
    return 0;
}