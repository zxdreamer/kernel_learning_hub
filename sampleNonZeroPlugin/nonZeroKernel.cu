#include "nonZeroKernel.h"

inline __device__ int32_t isZero(float const& a) {
  return a == 0.F;
}

inline __device__ int32_t isZero(half const& a) {
#if __CUDA_ARCH__ >= 530
  return a == __float2half(0.F);
#else
  return __half2float(a) == 0.F;
#endif
}

template <typename T>
__global__ void findNonZeroIndicesKernel(T const* X,
                                         int32_t* output,
                                         int32_t* count,
                                         int32_t const* K,
                                         int32_t R,
                                         int32_t C,
                                         int32_t rowOrder) {
  int32_t col = blockIdx.x * blockDim.x + threadIdx.x;

  // 1. 行主序输入，遍历行较快，列作为线程索引
  if (col < C) {
    for (int32_t row = 0; row < R; ++row) {
      if (!isZero(X[row * C + col])) {
        // 2. index代表本次输出的索引，原子操作保证线程安全
        int32_t index = atomicAdd(count, 1);
        if (output) {
          // 3. 列主序：[(r0,r1,r2,...),(c0,c1,c2,...)]
          if (rowOrder == 0) {
            output[index] = row;
            output[index + *K] = col;
          } else { // 4. 行主序：[(r0,c0),(r1,c1),(r2,c2),...]
            output[2 * index] = row;
            output[2 * index + 1] = col;
          }
        }
      }
    }
  }
}

template <typename T>
void nonZeroIndicesImpl(T const* X,
                        int32_t* output,
                        int32_t* count,
                        int32_t const* K,
                        int32_t R,
                        int32_t C,
                        bool rowOrder,
                        cudaStream_t stream) {
  constexpr int32_t kBLOCK_SIZE = 256;
  int32_t const blocksPerGrid = (C + kBLOCK_SIZE - 1) / kBLOCK_SIZE;

  findNonZeroIndicesKernel<<<blocksPerGrid, kBLOCK_SIZE, 0, stream>>>(
      X, output, count, K, R, C, static_cast<int32_t>(rowOrder));
}

// 定义float和half的偏特化模板
#define NONZERO_SPECIALIZED_IMPL(T)                                                                \
  template void nonZeroIndicesImpl<T>(T const* X,                                                  \
                                      int32_t* output,                                             \
                                      int32_t* count,                                              \
                                      int32_t const* K,                                            \
                                      int32_t R,                                                   \
                                      int32_t C,                                                   \
                                      bool rowOrder,                                               \
                                      cudaStream_t stream);

NONZERO_SPECIALIZED_IMPL(float)
NONZERO_SPECIALIZED_IMPL(half)
