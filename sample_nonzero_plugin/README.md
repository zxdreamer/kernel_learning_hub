# 十分钟跑起来：最小可用的 TensorRT PluginV3 模板

## Description

本示例（sample_nonzero_plugin）实现了 NonZero 运算的一个 TensorRT 插件，可配置以 行序（同一行给出一对索引）或 列序（同一列给出一对索引）的格式输出非零元素的索引。

NonZero 的含义：找出输入张量中所有非零元素的位置索引。

## How does this sample work?

本示例会创建并运行一个只包含单个 NonZeroPlugin 节点的 TensorRT 引擎，演示如何实现输出形状依赖于输入数据的自定义层，并将其加入 TensorRT 网络中。
具体来说，本示例会：
1. 实现一个 NonZero 运算的 TensorRT 插件
2. 创建网络并构建引擎
3. 运行推理并查看输出

### Preparing sample data
1. 准备输入数据
```
python data/make_mnist_pgms.py 
```
2. 准备onnx
```
python onnx_engine/build_onnx_example.py
```
目前，这两部分已经上传到github
### Build code
```
mkdir build
cd build
cmake ..
make
```
执行上述代码后，会产生plugin动态库`libnonzero_plugin.so`和可执行文件`libnonzero_plugin.so`
### Convert Model format
```
trtexec \
  --onnx=./onnx_engine/nonzero_plugin_static28x28.onnx \
  --saveEngine=./onnx_engine/nonzero.engine \
  --dynamicPlugins=./build/libnonzero_plugin.so \
  --setPluginsToSerialize=./build/libnonzero_plugin.so \
  --verbose
```
### Run bin
```
cd build
./sample_non_zero_plugin --engine=/home/zxd/code/kernel_learning_hub/sample_nonzero_plugin/onnx_engine/nonzero.engine --datadir=/home/zxd/code/kernel_learning_hub/sample_nonzero_plugin/data/mnist
```
#### Verify that the sample ran successfully. If the sample runs successfully you should see output similar to the following:
```
./sample_non_zero_plugin --engine=/home/zxd/code/kernel_learning_hub/sample_nonzero_plugin/onnx_engine/nonzero.engine --datadir=/home/zxd/code/kernel_learning_hub/sample_nonzero_plugin/data/mnist
[TEST] Start: TensorRT.sample_non_zero_plugin
[TRT][INFO] Loaded engine size: 0 MiB
[TRT][INFO] [MemUsageChange] TensorRT-managed allocation in IExecutionContext creation: CPU +0, GPU +0, now: CPU 0, GPU 0 (MiB)
filename: 7.pgm
Input:
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0
0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0
0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

mRowOrder: 1
complate enqueue....
count: 89
Output:
5 6
5 7
5 8
5 9
5 10
5 11
5 12
5 13
5 14
5 15
5 16
5 17
5 18
5 19
5 20
5 21
6 6
6 7
6 8
6 9
6 10
6 11
6 12
6 13
6 14
6 15
6 16
6 17
6 18
6 19
6 20
6 21
6 22
7 6
7 7
7 8
7 9
7 10
7 11
7 12
7 13
7 14
7 15
7 16
7 17
7 18
7 19
7 20
7 21
7 22
8 20
8 21
8 22
9 20
9 21
9 22
10 20
10 21
10 22
11 20
11 21
11 22
12 20
12 21
12 22
14 20
14 21
14 22
15 20
15 21
15 22
16 20
16 21
16 22
17 20
17 21
17 22
18 20
18 21
18 22
19 20
19 21
19 22
20 20
20 21
20 22
21 20
21 21
21 22
[TEST] PASS : TensorRT.sample_non_zero_plugin
```

### Sample `--help` options

To see the full list of available options and their descriptions, use the `-h` or `--help` command line option.

**Other documentation**
- [Introduction To NVIDIA’s TensorRT Samples](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sample-support-guide/index.html#samples)
- [Working With TensorRT Using The C++ API](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#c_topics)
- [NVIDIA’s TensorRT Documentation Library](https://docs.nvidia.com/deeplearning/sdk/tensorrt-archived/index.html)

# License

For terms and conditions for use, reproduction, and distribution, see the [TensorRT Software License Agreement](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html) documentation.


# Changelog

March 2024
This is the first version of this `README.md` file.


# Known issues

Windows users building this sample with Visual Studio with a CUDA version different from the TensorRT package will need to retarget the project to build against the installed CUDA version through the `Build Dependencies -> Build Customization` menu.

# 欢迎关注我的公众号和同名知乎 Ai Infra之道
![演示图](../公众号.png)

### 创作不易，如果感觉对你有用，请给我点一颗星，十分感谢！！！！
### 创作不易，如果感觉对你有用，请给我点一颗星，十分感谢！！！！
### 创作不易，如果感觉对你有用，请给我点一颗星，十分感谢！！！！
