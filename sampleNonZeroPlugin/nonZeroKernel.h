#ifndef SAMPLE_NONZERO_KERNEL_H
#define SAMPLE_NONZERO_KERNEL_H

#include <cuda_fp16.h>

#include <cstdint>

// X: 输入二维数据
// output: 输出数据
// count: 当前线程计算的输出数目
// K: 列序输出时的宽度
// R,C: 输入的长宽
// rowOrder: 标识输出时行序还是列序
// stream: 计算的流
template <typename T>
void nonZeroIndicesImpl(T const* X,
                        int32_t* output,
                        int32_t* count,
                        int32_t const* K,
                        int32_t R,
                        int32_t C,
                        bool rowOrder,
                        cudaStream_t stream);

#endif // SAMPLE_NONZERO_KERNEL_H
