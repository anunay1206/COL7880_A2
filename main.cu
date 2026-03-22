#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "histogram.cuh"

void run_knn(float* x, float* y, float* z, int* intensity, int N, int k, int* out);
void run_approx_knn(float* x, float* y, float* z, int* intensity, int N, int k, int* out);
void run_kmeans(float* x, float* y, float* z, int* intensity, int N, int k, int T, int* out);

#define TIME_IT(label, call) \
    do { \
        cudaEvent_t _s, _e; \
        cudaEventCreate(&_s); cudaEventCreate(&_e); \
        cudaEventRecord(_s); \
        call; \
        cudaEventRecord(_e); \
        cudaEventSynchronize(_e); \
        float _ms = 0; \
        cudaEventElapsedTime(&_ms, _s, _e); \
        printf("  %-25s : %.3f ms\n", label, _ms); \
        cudaEventDestroy(_s); cudaEventDestroy(_e); \
    } while(0)

int main(int argc, char* argv[]) {
    if (argc < 3) {
        fprintf(stderr, "Usage: ./a2 input.txt [knn|approx_knn|kmeans]\n");
        return 1;
    }

    const char* input_file = argv[1];
    const char* mode       = argv[2];

    FILE* f = fopen(input_file, "r");
    if (!f) { fprintf(stderr, "Cannot open %s\n", input_file); return 1; }

    int N, k, T;
    (void)fscanf(f, "%d %d %d", &N, &k, &T);

    float* x         = (float*)malloc(N * sizeof(float));
    float* y         = (float*)malloc(N * sizeof(float));
    float* z         = (float*)malloc(N * sizeof(float));
    int*   intensity = (int*)  malloc(N * sizeof(int));

    for (int i = 0; i < N; i++)
        (void)fscanf(f, "%f %f %f %d", &x[i], &y[i], &z[i], &intensity[i]);
    fclose(f);

    int*  out       = (int*)malloc(N * sizeof(int));
    const char* outfile = NULL;

    printf("\nN=%d  k=%d  T=%d\n\n", N, k, T);

    if (strcmp(mode, "knn") == 0) {
        TIME_IT("Exact KNN", run_knn(x, y, z, intensity, N, k, out));
        outfile = "knn.txt";
    }
    else if (strcmp(mode, "approx_knn") == 0) {
        TIME_IT("Approximate KNN", run_approx_knn(x, y, z, intensity, N, k, out));
        outfile = "approx_knn.txt";
    }
    else if (strcmp(mode, "kmeans") == 0) {
        TIME_IT("K-means", run_kmeans(x, y, z, intensity, N, k, T, out));
        outfile = "kmeans.txt";
    }
    else {
        fprintf(stderr, "Unknown mode: %s\n", mode);
        fprintf(stderr, "Use: knn | approx_knn | kmeans\n");
        return 1;
    }

    // Write output
    FILE* fo = fopen(outfile, "w");
    if (!fo) { fprintf(stderr, "Cannot open %s for writing\n", outfile); return 1; }
    for (int i = 0; i < N; i++)
        fprintf(fo, "%f %f %f %d\n", x[i], y[i], z[i], out[i]);
    fclose(fo);
    printf("Written %s\n", outfile);

    free(x); free(y); free(z); free(intensity); free(out);
    return 0;
}