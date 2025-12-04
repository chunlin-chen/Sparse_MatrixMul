#pragma once
#include <vector>
void dense_to_csr_block(const std::vector<float>& dense, int N,
                        int row_start, int row_end,
                        std::vector<int>& rowPtr,
                        std::vector<int>& colIdx,
                        std::vector<float>& values);
void matmul_sparse_csr_pipeline(const std::vector<float>& A,
                                    const std::vector<float>& B,
                                    std::vector<float>& C,
                                    int N, int block_rows=128);
