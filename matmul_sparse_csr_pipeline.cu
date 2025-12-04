// ---------------- GPU Kernel: Segmented shared memory with atomic ----------------
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <cmath>
#include <omp.h>

#define CUDA_CHECK(err) \
    do { \
        cudaError_t err__ = (err); \
        if (err__ != cudaSuccess) { \
            std::cerr << "CUDA error: " << cudaGetErrorString(err__) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            std::exit(1); \
        } \
    } while (0)

// ---------------- GPU Kernel ----------------
// 每 block 處理一行，thread 處理 row 的部分元素
template<int BLOCK_SIZE>
__global__ void spgemm_pipeline_kernel(int row_start_global, int rows,
                                       int N,
                                       const int* __restrict__ rowPtrA,
                                       const int* __restrict__ colIdxA,
                                       const float* __restrict__ valA,
                                       const int* __restrict__ rowPtrB,
                                       const int* __restrict__ colIdxB,
                                       const float* __restrict__ valB,
                                       float* __restrict__ C)
{
    extern __shared__ float s_row[]; // accumulator per row
    int tid = threadIdx.x;
    int row_local = blockIdx.x;
    if(row_local >= rows) return;

    int row_global = row_start_global + row_local;

    // initialize shared accumulator
    for(int j = tid; j < N; j += BLOCK_SIZE) s_row[j] = 0.0f;
    __syncthreads();

    int row_startA_local = rowPtrA[row_local];
    int row_endA_local   = rowPtrA[row_local + 1];

    for(int idx = row_startA_local; idx < row_endA_local; idx++) {
        int k = colIdxA[idx];
        float a = valA[idx];

        int row_startB = rowPtrB[k];
        int row_endB   = rowPtrB[k+1];

        for(int t = tid; t < row_endB - row_startB; t += BLOCK_SIZE) {
            int col = colIdxB[row_startB + t];
            float val = a * valB[row_startB + t];
            atomicAdd(&s_row[col], val); // 保護累加
        }
    }
    __syncthreads();

    // write back to global memory
    float* C_row_ptr = C + (long long)row_global * N;
    for(int j = tid; j < N; j += BLOCK_SIZE) {
        if(s_row[j] != 0.0f) C_row_ptr[j] = s_row[j];
    }
}

static void dense_to_csr(const std::vector<float>& dense, int N,
                         int row_start, int row_end,
                         std::vector<int>& rowPtr,
                         std::vector<int>& colIdx,
                         std::vector<float>& values)
{
    int rows = row_end - row_start;
    rowPtr.assign(rows + 1, 0);
    std::vector<int> row_nnz(rows, 0);

    for (int i = 0; i < rows; i++) {
        int cnt = 0;
        int base = (row_start + i) * N;
        for (int j = 0; j < N; j++) {
            if (fabsf(dense[base + j]) > 1e-8f) cnt++;
        }
        row_nnz[i] = cnt;
    }

    rowPtr[0] = 0;
    for (int i = 0; i < rows; i++) {
        rowPtr[i + 1] = rowPtr[i] + row_nnz[i];
    }

    int total_nnz = rowPtr[rows];
    colIdx.resize(total_nnz);
    values.resize(total_nnz);

    for (int i = 0; i < rows; i++) {
        int base = (row_start + i) * N;
        int offset = rowPtr[i];
        for (int j = 0; j < N; j++) {
            float v = dense[base + j];
            if (fabsf(v) > 1e-8f) {
                colIdx[offset] = j;
                values[offset] = v;
                offset++;
            }
        }
    }
}

// ---------------- CPU Dense -> CSR ----------------
static void dense_to_csr_block(const std::vector<float>& dense, int N,
                               int row_start, int row_end,
                               std::vector<int>& rowPtr,
                               std::vector<int>& colIdx,
                               std::vector<float>& values)
{
    int rows = row_end - row_start;
    rowPtr.assign(rows+1, 0);
    std::vector<int> row_nnz(rows,0);

    #pragma omp parallel for schedule(static)
    for(int i=0;i<rows;i++){
        int cnt = 0;
        int base = (row_start+i)*N;
        for(int j=0;j<N;j++){
            if(fabs(dense[base+j])>1e-8f) cnt++;
        }
        row_nnz[i]=cnt;
    }

    rowPtr[0]=0;
    for(int i=0;i<rows;i++) rowPtr[i+1]=rowPtr[i]+row_nnz[i];

    int total_nnz = rowPtr[rows];
    colIdx.resize(total_nnz);
    values.resize(total_nnz);

    #pragma omp parallel for schedule(static)
    for(int i=0;i<rows;i++){
        int base = (row_start+i)*N;
        int offset = rowPtr[i];
        for(int j=0;j<N;j++){
            float v = dense[base+j];
            if(fabs(v)>1e-8f){
                colIdx[offset]=j;
                values[offset]=v;
                offset++;
            }
        }
    }
}

// ---------------- Host Function: CPU-GPU Pipeline ----------------
void matmul_sparse_csr_pipeline(const std::vector<float>& A,
                                const std::vector<float>& B,
                                std::vector<float>& C,
                                int N,
                                int block_rows = 128)
{
    if((int)C.size() != N*N) C.resize(N*N,0.0f);

    // --- B CSR 一次生成 ---
    std::vector<int> rowPtrB, colIdxB;
    std::vector<float> valB;
    dense_to_csr_block(B, N, 0, N, rowPtrB, colIdxB, valB);
    // dense_to_csr(B, N, 0, N, rowPtrB, colIdxB, valB);

    int *d_rowPtrB, *d_colIdxB; float *d_valB, *d_C;
    CUDA_CHECK(cudaMalloc(&d_rowPtrB, (N+1)*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_colIdxB, colIdxB.size()*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_valB, valB.size()*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, N*N*sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_rowPtrB, rowPtrB.data(), (N+1)*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_colIdxB, colIdxB.data(), colIdxB.size()*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_valB, valB.data(), valB.size()*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_C, 0, N*N*sizeof(float)));

    // --- double buffer for A tiles ---
    std::vector<int> rowPtrA[2], colIdxA[2];
    std::vector<float> valA[2];
    int buffer_idx=0;

    int* d_rowPtrA[2]; int* d_colIdxA[2]; float* d_valA[2];
    cudaStream_t streams[2];
    CUDA_CHECK(cudaStreamCreate(&streams[0]));
    CUDA_CHECK(cudaStreamCreate(&streams[1]));

    int num_tiles = (N+block_rows-1)/block_rows;
    for(int t=0;t<num_tiles+1;t++){
        int row_start = t*block_rows;
        int row_end   = std::min(N,(t+1)*block_rows);
        int rows = row_end - row_start;

        // CPU 生成 CSR
        if(rows>0)
            dense_to_csr_block(A,N,row_start,row_end,rowPtrA[buffer_idx],colIdxA[buffer_idx],valA[buffer_idx]);
            // dense_to_csr(A,N,row_start,row_end,rowPtrA[buffer_idx],colIdxA[buffer_idx],valA[buffer_idx]);

        // GPU 計算上一塊
        if(t>0){
            int prev_idx = 1-buffer_idx;
            int prev_row_start = (t-1)*block_rows;
            int prev_rows = std::min(block_rows,N-prev_row_start);

            CUDA_CHECK(cudaMalloc(&d_rowPtrA[prev_idx], (prev_rows+1)*sizeof(int)));
            CUDA_CHECK(cudaMalloc(&d_colIdxA[prev_idx], colIdxA[prev_idx].size()*sizeof(int)));
            CUDA_CHECK(cudaMalloc(&d_valA[prev_idx], valA[prev_idx].size()*sizeof(float)));

            CUDA_CHECK(cudaMemcpyAsync(d_rowPtrA[prev_idx], rowPtrA[prev_idx].data(), (prev_rows+1)*sizeof(int), cudaMemcpyHostToDevice, streams[prev_idx]));
            CUDA_CHECK(cudaMemcpyAsync(d_colIdxA[prev_idx], colIdxA[prev_idx].data(), colIdxA[prev_idx].size()*sizeof(int), cudaMemcpyHostToDevice, streams[prev_idx]));
            CUDA_CHECK(cudaMemcpyAsync(d_valA[prev_idx], valA[prev_idx].data(), valA[prev_idx].size()*sizeof(float), cudaMemcpyHostToDevice, streams[prev_idx]));

            size_t shared_mem_bytes = N*sizeof(float);
            int grid_size = prev_rows;
            spgemm_pipeline_kernel<1024><<<grid_size,1024,shared_mem_bytes,streams[prev_idx]>>>(
                prev_row_start, prev_rows, N,
                d_rowPtrA[prev_idx], d_colIdxA[prev_idx], d_valA[prev_idx],
                d_rowPtrB, d_colIdxB, d_valB,
                d_C
            );
            CUDA_CHECK(cudaStreamSynchronize(streams[prev_idx]));
            CUDA_CHECK(cudaFree(d_rowPtrA[prev_idx]));
            CUDA_CHECK(cudaFree(d_colIdxA[prev_idx]));
            CUDA_CHECK(cudaFree(d_valA[prev_idx]));
        }

        buffer_idx = 1-buffer_idx;
    }

    CUDA_CHECK(cudaMemcpy(C.data(), d_C, N*N*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_rowPtrB));
    CUDA_CHECK(cudaFree(d_colIdxB));
    CUDA_CHECK(cudaFree(d_valB));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaStreamDestroy(streams[0]));
    CUDA_CHECK(cudaStreamDestroy(streams[1]));
}
