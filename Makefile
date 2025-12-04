# ===== Compilers =====
CXX           := g++
NVCC          := nvcc
HOST_COMPILER := /opt/apps/gcc/6.3.0/bin/g++   # 如需改系統 g++ 就改成 g++

# ===== CUDA paths =====
CUDA_HOME ?= /usr/local/cuda
CUDA_INC  := $(CUDA_HOME)/include
CUDA_LIB1 := $(CUDA_HOME)/lib64
CUDA_LIB2 := $(CUDA_HOME)/targets/x86_64-linux/lib

# ===== GPU arch =====
# 依卡型調整：sm_75(Quadro RTX 5000), sm_86(RTX 30/40)
ARCH ?= sm_75

# ===== Files =====
CPP_SRCS := main.cpp matmul_sparse_cpu.cpp
CU_SRCS  := matmul_base.cu matmul_sparse_csr.cu matmul_sparse_csr_fast.cu matmul_sparse_csr_pipeline.cu
OBJS     := $(CPP_SRCS:.cpp=.o) $(CU_SRCS:.cu=.o)

TARGET := matmul_sparse

# ===== Flags =====
CXXFLAGS   := -O2 -std=c++14 -fopenmp -I. -I$(CUDA_INC)

# nvcc 透過 -Xcompiler 把 -fopenmp 丟給 host compiler
NVCCFLAGS  := -O2 -std=c++14 -I. -I$(CUDA_INC) \
              -gencode arch=compute_$(ARCH:sm_%=%),code=$(ARCH) \
              -ccbin $(HOST_COMPILER) \
              -Xcompiler -fopenmp

# 連結：不要寫 -fopenmp，改成連 libgomp（OpenMP runtime）
LDFLAGS    := -L$(CUDA_LIB1) -L$(CUDA_LIB2) -lcudart -lgomp -lcusparse -lcublas\
              -Xlinker -rpath -Xlinker $(CUDA_LIB1) \
              -Xlinker -rpath -Xlinker $(CUDA_LIB2)

# ===== Rules =====
.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $(NVCCFLAGS) $(OBJS) $(LDFLAGS) -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

# 方便直接執行：make run N=512 S=0.9 SEED=42
run: $(TARGET)
	./$(TARGET) $(N) $(S) $(SEED)
