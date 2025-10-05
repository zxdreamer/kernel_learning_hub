#!/usr/bin/env python3
import onnx
from onnx import helper, TensorProto

R, C = 28, 28
inp = helper.make_tensor_value_info("Input", TensorProto.FLOAT, [R, C])

plugin_node = helper.make_node(
    "NonZeroPlugin",                    # op_type，对应插件名
    inputs=["Input"],
    outputs=["Output0", "Output1"],     # 两个输出，和插件一致
    rowOrder=1,                         # 传给 PluginField("rowOrder")，0=列主，1=行主
    name="NonZeroNode"
)

out0 = helper.make_tensor_value_info("Output0", TensorProto.INT32, [2, "K"])
out1 = helper.make_tensor_value_info("Output1", TensorProto.INT32, [1])

graph = helper.make_graph(
    nodes=[plugin_node],
    name="NonZeroPluginGraph",
    inputs=[inp],
    outputs=[out0, out1],
)

model = helper.make_model(
    graph,
    opset_imports=[
        helper.make_operatorsetid("", 13),
    ]
)
onnx.save(model, "onnx_engine/nonzero_plugin_static28x28.onnx")
print("Saved to onnx_engine/nonzero_plugin_static28x28.onnx")
