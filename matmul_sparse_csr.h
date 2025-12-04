// matmul_sparse_csr.h
#pragma once
#include <vector>

// A, B, C 都是 dense N×N，跟 main.cpp 現在用的一樣
// 裡面會在 CPU 把 A/B 轉成 CSR，再丟到 GPU 做 sparse x sparse
extern "C" void matmul_sparse_csr_gpu(const std::vector<float>& A,
                                      const std::vector<float>& B,
                                      std::vector<float>& C,
                                      int N);
