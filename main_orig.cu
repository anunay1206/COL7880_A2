#include <stdio.h>
#include <stdlib.h>
#include "histogram.cuh"

// Forward declarations
void run_knn(float* x, float* y, float* z, int* intensity, int N, int k, int* out);
void run_approx_knn(float* x, float* y, float* z, int* intensity, int N, int k, int* out);
void run_kmeans(float* x, float* y, float* z, int* intensity, int N, int k, int T, int* out);

int main(int argc, char* argv[]) {
    if (argc < 2) { fprintf(stderr, "Usage: ./a2 input.txt\n"); return 1; }

    // --- Read input ---
    FILE* f = fopen(argv[1], "r");
    if (!f) { fprintf(stderr, "Cannot open %s\n", argv[1]); return 1; }

    int N, k, T;
    fscanf(f, "%d %d %d", &N, &k, &T);

    float* x         = (float*)malloc(N * sizeof(float));
    float* y         = (float*)malloc(N * sizeof(float));
    float* z         = (float*)malloc(N * sizeof(float));
    int*   intensity = (int*)  malloc(N * sizeof(int));

    for (int i = 0; i < N; i++)
        fscanf(f, "%f %f %f %d", &x[i], &y[i], &z[i], &intensity[i]);
    fclose(f);

    int* out_knn    = (int*)malloc(N * sizeof(int));
    int* out_approx = (int*)malloc(N * sizeof(int));
    int* out_kmeans = (int*)malloc(N * sizeof(int));

    // --- Run three methods ---
    printf("Running exact KNN...\n");
    run_knn(x, y, z, intensity, N, k, out_knn);

    printf("Running approximate KNN...\n");
    run_approx_knn(x, y, z, intensity, N, k, out_approx);

    printf("Running K-means...\n");
    run_kmeans(x, y, z, intensity, N, k, T, out_kmeans);

    // --- Write outputs ---
    auto write_out = [&](const char* fname, int* out) {
        FILE* fo = fopen(fname, "w");
        for (int i = 0; i < N; i++)
            fprintf(fo, "%f %f %f %d\n", x[i], y[i], z[i], out[i]);
        fclose(fo);
        printf("Written %s\n", fname);
    };

    write_out("knn.txt",        out_knn);
    write_out("approx_knn.txt", out_approx);
    write_out("kmeans.txt",     out_kmeans);

    free(x); free(y); free(z); free(intensity);
    free(out_knn); free(out_approx); free(out_kmeans);
    return 0;
}