#pragma once
#include "simulator.hpp"
namespace sjtu {

void Calculate(std::vector<Matrix *> keys, std::vector<Matrix *> values,
               Rater &rater, GpuSimulator &gpu_sim,
               MatrixMemoryAllocator matrix_memory_allocator) {
  assert(keys.size() == values.size());

  for (size_t i = 0; i < keys.size(); ++i) {
    auto current_query = rater.GetNextQuery();
    size_t n = i + 1;

    // Build K_concat efficiently - release intermediate results immediately
    Matrix* K_concat = matrix_memory_allocator.Allocate("K0");
    gpu_sim.Copy(keys[0], K_concat, Position::kInGpuHbm);
    for (size_t j = 1; j <= i; ++j) {
      Matrix* new_K = matrix_memory_allocator.Allocate("K");
      gpu_sim.Concat(K_concat, keys[j], new_K, 0, Position::kInGpuHbm);
      gpu_sim.ReleaseMatrix(K_concat);
      K_concat = new_K;
    }

    // Build V_concat efficiently
    Matrix* V_concat = matrix_memory_allocator.Allocate("V0");
    gpu_sim.Copy(values[0], V_concat, Position::kInGpuHbm);
    for (size_t j = 1; j <= i; ++j) {
      Matrix* new_V = matrix_memory_allocator.Allocate("V");
      gpu_sim.Concat(V_concat, values[j], new_V, 0, Position::kInGpuHbm);
      gpu_sim.ReleaseMatrix(V_concat);
      V_concat = new_V;
    }

    // Move to SRAM
    gpu_sim.MoveMatrixToSharedMem(current_query);
    gpu_sim.MoveMatrixToSharedMem(K_concat);
    gpu_sim.MoveMatrixToSharedMem(V_concat);

    // Transpose K
    gpu_sim.Transpose(K_concat, Position::kInSharedMemory);

    // Process all rows and build answer
    Matrix* Answer = nullptr;
    for (size_t r = 0; r < n; ++r) {
      Matrix* Q_r = matrix_memory_allocator.Allocate("Q");
      gpu_sim.GetRow(current_query, r, Q_r, Position::kInSharedMemory);

      Matrix* scores = matrix_memory_allocator.Allocate("s");
      gpu_sim.MatMul(Q_r, K_concat, scores);
      gpu_sim.ReleaseMatrix(Q_r);

      Matrix* exp_s = matrix_memory_allocator.Allocate("e");
      gpu_sim.MatExp(scores, exp_s);
      gpu_sim.ReleaseMatrix(scores);

      Matrix* sum_s = matrix_memory_allocator.Allocate("m");
      gpu_sim.Sum(exp_s, sum_s);

      Matrix* soft = matrix_memory_allocator.Allocate("f");
      gpu_sim.MatDiv(exp_s, sum_s, soft);
      gpu_sim.ReleaseMatrix(exp_s);
      gpu_sim.ReleaseMatrix(sum_s);

      Matrix* attn = matrix_memory_allocator.Allocate("a");
      gpu_sim.MatMul(soft, V_concat, attn);
      gpu_sim.ReleaseMatrix(soft);

      if (r == 0) {
        Answer = attn;
      } else {
        Matrix* new_ans = matrix_memory_allocator.Allocate("A");
        gpu_sim.Concat(Answer, attn, new_ans, 0, Position::kInSharedMemory);
        gpu_sim.ReleaseMatrix(attn);
        Answer = new_ans;
      }
    }

    // Release K and V after all rows processed
    gpu_sim.ReleaseMatrix(K_concat);
    gpu_sim.ReleaseMatrix(V_concat);

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
