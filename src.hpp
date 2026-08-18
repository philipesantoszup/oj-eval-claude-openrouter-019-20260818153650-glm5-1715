#pragma once
#include "simulator.hpp"
namespace sjtu {

void Calculate(std::vector<Matrix *> keys, std::vector<Matrix *> values,
               Rater &rater, GpuSimulator &gpu_sim,
               MatrixMemoryAllocator matrix_memory_allocator) {
  assert(keys.size() == values.size());

  for (size_t i = 0; i < keys.size(); ++i) {
    auto current_query = rater.GetNextQuery();
    size_t n = i + 1;  // number of rows in query

    // Build K_concat by copying keys[0..i]
    Matrix* K_concat = matrix_memory_allocator.Allocate("K_init");
    gpu_sim.Copy(keys[0], K_concat, Position::kInGpuHbm);
    for (size_t j = 1; j <= i; ++j) {
      Matrix* new_K = matrix_memory_allocator.Allocate("K");
      gpu_sim.Concat(K_concat, keys[j], new_K, 0, Position::kInGpuHbm);
      K_concat = new_K;
    }

    // Build V_concat by copying values[0..i]
    Matrix* V_concat = matrix_memory_allocator.Allocate("V_init");
    gpu_sim.Copy(values[0], V_concat, Position::kInGpuHbm);
    for (size_t j = 1; j <= i; ++j) {
      Matrix* new_V = matrix_memory_allocator.Allocate("V");
      gpu_sim.Concat(V_concat, values[j], new_V, 0, Position::kInGpuHbm);
      V_concat = new_V;
    }

    // Move data to SRAM
    gpu_sim.MoveMatrixToSharedMem(current_query);
    gpu_sim.MoveMatrixToSharedMem(K_concat);
    gpu_sim.MoveMatrixToSharedMem(V_concat);

    // Transpose K for matmul
    gpu_sim.Transpose(K_concat, Position::kInSharedMemory);

    // Process rows and build answer incrementally
    Matrix* Answer = nullptr;
    for (size_t r = 0; r < n; ++r) {
      Matrix* Q_r = matrix_memory_allocator.Allocate("Qr");
      gpu_sim.GetRow(current_query, r, Q_r, Position::kInSharedMemory);

      Matrix* scores = matrix_memory_allocator.Allocate("sc");
      gpu_sim.MatMul(Q_r, K_concat, scores);
      gpu_sim.ReleaseMatrix(Q_r);

      Matrix* exp_s = matrix_memory_allocator.Allocate("ex");
      gpu_sim.MatExp(scores, exp_s);
      gpu_sim.ReleaseMatrix(scores);

      Matrix* sum_s = matrix_memory_allocator.Allocate("sm");
      gpu_sim.Sum(exp_s, sum_s);

      Matrix* softmax = matrix_memory_allocator.Allocate("sf");
      gpu_sim.MatDiv(exp_s, sum_s, softmax);
      gpu_sim.ReleaseMatrix(exp_s);
      gpu_sim.ReleaseMatrix(sum_s);

      Matrix* attn = matrix_memory_allocator.Allocate("at");
      gpu_sim.MatMul(softmax, V_concat, attn);
      gpu_sim.ReleaseMatrix(softmax);

      if (r == 0) {
        Answer = attn;
      } else {
        Matrix* new_ans = matrix_memory_allocator.Allocate("ans");
        gpu_sim.Concat(Answer, attn, new_ans, 0, Position::kInSharedMemory);
        gpu_sim.ReleaseMatrix(attn);
        Answer = new_ans;
      }
    }

    gpu_sim.MoveMatrixToGpuHbm(Answer);
    gpu_sim.Run(false, &matrix_memory_allocator);
    rater.CommitAnswer(*Answer);
  }
}

void Test(Rater &rater, GpuSimulator &gpu_sim,
          MatrixMemoryAllocator &matrix_memory_allocator) {
  Calculate(rater.keys_, rater.values_, rater, gpu_sim, matrix_memory_allocator);
  rater.PrintResult(gpu_sim);
}

} // namespace sjtu
