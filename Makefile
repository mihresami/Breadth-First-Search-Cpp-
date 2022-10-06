CC=gcc
CXX=g++
NVCC=nvcc

CXXFLAGS=-w -O3 -std=c++11

bfs-exec: main.cu bfsCUDA.cu bfsCPU.cpp graph.cpp
	$(NVCC) $(CXXFLAGS) main.cu bfsCPU.cpp graph.cpp -o bfs-exec

clean:
	rm -f bfs-exec
