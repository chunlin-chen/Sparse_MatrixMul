#include <random>
#include <vector>
#include <iostream>
#include <string>
#include <chrono>
#include <iomanip>
#include "matmul_sparse_cpu.hpp"
#include "matmul_base.h"
#include "matmul_sparse_csr.h"
#include "matmul_sparse_csr_fast.h"
#include "matmul_sparse_csr_pipeline.h"

using namespace std;
using namespace std::chrono;

vector<vector<int>> generateSparseMatrix_seeded(int N, double sparsity, unsigned seed) {
    mt19937 rng(seed);
    bernoulli_distribution is_zero(sparsity);
    uniform_int_distribution<int> nonzero_val(1, 10);

    vector<vector<int>> mat(N, vector<int>(N, 0));
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            if (!is_zero(rng)) mat[i][j] = nonzero_val(rng);
    return mat;
}

SparseMatrixCSR denseToSparseObject(const vector<vector<int>>& dense, int N) {
    SparseMatrixCSR mat(N, N);
    for (int i = 0; i < N; ++i) {
        vector<pair<int, double>> rowData;
        for (int j = 0; j < N; ++j) {
            if (dense[i][j] != 0.0) {
                rowData.push_back({j, dense[i][j]});
            }
        }
        mat.addRow(rowData);
    }
    return mat;
}

int main(int argc, char* argv[]) {
    // ---------- checking argc ----------
    if (argc < 3) {
        cout << "Usage: " << argv[0] << " <matrix_size> <sparsity> [seed]\n";
        cout << "Example: " << argv[0] << " 1024 0.9 42\n";
        return 1;
    }
    int N = stoi(argv[1]);
    double sparsity = stod(argv[2]);
    int seed = (argc >= 4) ? stoi(argv[3]) : 2025;

    if (N <= 0) {
        cerr << "Error: matrix_size must be > 0\n";
        return 1;
    }
    if (sparsity < 0.0 || sparsity >= 1.0) {
        cerr << "Error: sparsity must be in range [0.0, 1.0)\n";
        return 1;
    }
    cout << "\n================================================\n";
    cout << " Final Comparison Benchmark \n";
    cout << " N = " << N << ", Sparsity = " << sparsity * 100 << "%";
    cout << "\n================================================\n";

    // ---------- Generate Sparse Matrix ----------
    cout << "\n>> Generating Data...\n";
    auto matA = generateSparseMatrix_seeded(N, sparsity, static_cast<unsigned>(seed));
    auto matB = generateSparseMatrix_seeded(N, sparsity, static_cast<unsigned>(seed + 1));

    vector<float> A_f(N * N), B_f(N * N), C_f(N * N, 0.0f);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            A_f[i * N + j] = static_cast<float>(matA[i][j]);
            B_f[i * N + j] = static_cast<float>(matB[i][j]);
        }

    
    cout << "\n>> CPU Benchmarking...\n";
    double t_cpu_multi = 1e300;
    SparseMatrixCSR C_parallel(N, N); 
    {
        auto start = high_resolution_clock::now();
        SparseMatrixCSR mat_A = denseToSparseObject(matA, N);
        SparseMatrixCSR mat_B = denseToSparseObject(matB, N);
        C_parallel = mat_A.multiplySparseParallel(mat_B);
        auto end = high_resolution_clock::now();
        t_cpu_multi = duration<double, milli>(end - start).count();
        cout << left << setw(35) << "[Parallel CPU]"<<
        right << setw(12) << fixed << setprecision(2) << t_cpu_multi << " ms\n";
    }

    const int iters = 5;
    matmul_base(A_f, B_f, C_f, N);
    cout << "\n>> GPU Benchmarking...\n";
    // ---------- GPU baseline matmul ----------
    double best_ms = 1e300, sum_ms = 0.0;
    for (int t = 0; t < iters; ++t) {
        auto t0 = high_resolution_clock::now();
        matmul_base(A_f, B_f, C_f, N);
        auto t1 = high_resolution_clock::now();

        double ms = duration<double, milli>(t1 - t0).count();
        best_ms = std::min(best_ms, ms);
        sum_ms  += ms;
    }

    double avg_ms = sum_ms / iters;

    // cout << "[Baseline GPU] best over " << iters << " runs: " << best_ms << " ms\n";
    cout << left << setw(35) << "[Baseline GPU] avg over " + std::to_string(iters) + " runs:"<<
    right << setw(12) << fixed << setprecision(2) << avg_ms << " ms\n";

    /*
    // ---------- GPU sparse CSR matmul ----------
    std::vector<float> C_sparse(N * N, 0.0f);
    matmul_sparse_csr_gpu(A_f, B_f, C_sparse, N);  // warm-up

    double best_ms_csr = 1e300, sum_ms_csr = 0.0;
    for (int t = 0; t < iters; ++t) {
        auto t0_csr = high_resolution_clock::now();
        matmul_sparse_csr_gpu(A_f, B_f, C_sparse, N);
        auto t1_csr = high_resolution_clock::now();

        double ms_csr = duration<double, milli>(t1_csr - t0_csr).count();
        best_ms_csr = std::min(best_ms_csr, ms_csr);
        sum_ms_csr  += ms_csr;
    }

    double avg_ms_csr = sum_ms_csr / iters;

    // cout << "[Sparse CSR GPU] best over " << iters << " runs: " << best_ms_csr << " ms\n";
    cout << "[Sparse CSR GPU] avg  over " << iters << " runs: " << avg_ms_csr  << " ms\n";
    */

    // ==========================================
    // Fast CSR GPU (Dynamic)
    // ==========================================
    std::vector<float> C_fast(N * N, 0.0f);
    matmul_sparse_csr_fast_gpu(A_f, B_f, C_fast, N); // warm-up

    double best_ms_fast = 1e300, sum_ms_fast = 0.0;
    for (int t = 0; t < iters; ++t) {
        auto t0_fast = high_resolution_clock::now();
        matmul_sparse_csr_fast_gpu(A_f, B_f, C_fast, N);
        auto t1_fast = high_resolution_clock::now();

        double ms_fast = duration<double, milli>(t1_fast - t0_fast).count();
        best_ms_fast = std::min(best_ms_fast, ms_fast);
        sum_ms_fast  += ms_fast;
    }

    double avg_ms_fast = sum_ms_fast / iters;
    // cout << "[Fast CSR GPU]   best over " << iters << " runs: " << best_ms_fast << " ms\n";
    cout << left << setw(35) << "[CSR GPU] avg over " + std::to_string(iters) + " runs:"<<
    right << setw(12) << fixed << setprecision(2) << avg_ms_fast << " ms\n";

    // ==========================================
    // [New] CPU GPU Pipeline
    // ==========================================
    
    std::vector<float> C_pipeline(N * N, 0.0f);
    int block_rows = 128;
    matmul_sparse_csr_pipeline(A_f, B_f, C_pipeline, N, block_rows); // warm-up

    double best_ms_pipeline = 1e300, sum_ms_pipeline = 0.0;
    for (int t = 0; t < iters; ++t) {
        auto t0_pipeline = high_resolution_clock::now();
        matmul_sparse_csr_pipeline(A_f, B_f, C_pipeline, N, block_rows);
        auto t1_pipeline = high_resolution_clock::now();

        double ms_pipelien = duration<double, milli>(t1_pipeline - t0_pipeline).count();
        best_ms_pipeline = std::min(best_ms_pipeline, ms_pipelien);
        sum_ms_pipeline  += ms_pipelien;
    }

    double avg_ms_pipeline = sum_ms_pipeline / iters;
    // cout << "[CPU GPU Pipeline]   best over " << iters << " runs: " << best_ms_pipeline << " ms\n";
    cout << left << setw(35) << "[CPU GPU Pipeline] avg over " + std::to_string(iters) + " runs:"<<
    right << setw(12) << fixed << setprecision(2) << avg_ms_pipeline << " ms\n";
    
    // GPU Comparison
    cout << "\n>> GPU Speedups Comparison\n";
    cout << left << setw(35) << "  Baseline / CSR GPU:" 
        << right << setw(14) << fixed << setprecision(2) << (avg_ms / avg_ms_fast) << "x\n";
    cout << left << setw(35) << "  Baseline / CPU GPU Pipeline:" 
        << right << setw(14) << fixed << setprecision(2) << (avg_ms / avg_ms_pipeline) << "x\n";

    // CPU GPU Comparison
    string Best_GPU;
    double best_ms_gpu;
    if(avg_ms_pipeline <= avg_ms && avg_ms_pipeline <= avg_ms_fast){
        Best_GPU = "CPU GPU Pipeline";
        best_ms_gpu = avg_ms_pipeline;
    }
    else if(avg_ms_fast <= avg_ms && avg_ms_fast <= avg_ms_pipeline){
        Best_GPU = "CSR GPU";
        best_ms_gpu = avg_ms_fast;
    }
    else{
        Best_GPU = "Baseline GPU";
        best_ms_gpu = avg_ms;
    }
    cout << "\n>> Best CPU vs GPU Speedups Comparison\n";
    cout << left << setw(35) << "  Parallel CPU / " + Best_GPU + ":"
    << right << setw(14) << fixed << setprecision(2) << (t_cpu_multi / best_ms_gpu) << "x\n";

    // Check Correctness
    cout << "\n>> Correctness Check\n";
    double max_abs_diff_fast = 0.0;
    for (int i = 0; i < N * N; ++i) {
        double diff = static_cast<double>(C_f[i]) - static_cast<double>(C_fast[i]);
        max_abs_diff_fast = std::max(max_abs_diff_fast, std::abs(diff));
    }
    cout << left << setw(35) << "  Baseline GPU vs CSR GPU: ";
    if (max_abs_diff_fast < 1e-4) {
        cout << right << setw(10) << "PASS" 
            << " (diff: " << max_abs_diff_fast << ")\n";
    } else {
        cout << right << setw(10) << "FAIL" 
            << " (diff: " << max_abs_diff_fast << ")\n";
    }

    double max_abs_diff_pipeline = 0.0;
    for (int i = 0; i < N * N; ++i) {
        double diff = static_cast<double>(C_f[i]) - static_cast<double>(C_pipeline[i]);
        max_abs_diff_pipeline = std::max(max_abs_diff_pipeline, std::abs(diff));
    }
    cout << left << setw(35) << "  Baseline GPU vs CPU GPU Pipeline:";
    if (max_abs_diff_pipeline < 1e-4) {
        cout << right << setw(10) << "PASS" 
            << " (diff: " << max_abs_diff_pipeline << ")\n";
    } else {
        cout << right << setw(10) << "FAIL" 
            << " (diff: " << max_abs_diff_pipeline << ")\n";
    }
/*
    // ---------- Speedup 計算 (baseline / CSR) ----------
    double speedup_best = best_ms / best_ms_csr;
    double speedup_avg  = avg_ms  / avg_ms_csr;

    cout << "\n[Speedup (baseline / CSR)]\n";
    cout << "  best-time speedup: " << speedup_best << "x\n";
    cout << "  avg-time  speedup: " << speedup_avg  << "x\n";
    cout << "  ( >1.0 means CSR is faster than baseline )\n";

    // ========== 測試 CSR-GPU 結果是否跟 baseline 一樣 ==========
    double max_abs_diff = 0.0;
    double mse = 0.0;

    for (int i = 0; i < N * N; ++i) {
        double diff = static_cast<double>(C_f[i]) - static_cast<double>(C_sparse[i]);
        max_abs_diff = std::max(max_abs_diff, std::abs(diff));
        mse += diff * diff;
    }
    mse /= static_cast<double>(N) * static_cast<double>(N);
    double rmse = std::sqrt(mse);

    cout << "\n[Correctness check] baseline vs CSR-GPU\n";
    if (max_abs_diff == 0 ) {
        cout << "pass the test\n";
    } else {
        cout << "the difference is " << max_abs_diff << "\n";
        // cout << "  RMSE        = " << rmse << "\n";
    }
*/
    cout<<"\n";
    return 0;
}
