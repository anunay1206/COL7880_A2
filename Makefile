NVCC = nvcc
CFLAGS = -Xcompiler -fopenmp
TARGET = a2
SRCS = main.cu knn.cu approx_knn.cu kmeans.cu

$(TARGET): $(SRCS)
	$(NVCC) $(CFLAGS) -o $(TARGET) $(SRCS)

clean:
	rm -f $(TARGET) knn.txt approx_knn.txt kmeans.txt