# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import argparse
import struct
from pathlib import Path
from typing import List, Tuple

import numpy as np
import numpy.typing as npt
import onnx
from onnx.onnx_pb import GraphProto, ModelProto, NodeProto, TensorProto

from .onnx_model import ONNXModel
from .quant_utils import attribute_to_kwarg, load_model_with_shape_infer
import multiprocessing

from joblib import Parallel, delayed, parallel_config


def __q4_block_size() -> int:
    # happens to be 32 for now, but future quantization types
    # may have bigger block size
    return 32


def __q4_blob_size(block_size: int) -> int:
    return block_size // 2


def __q4_buf_size(rows: int, cols: int) -> int:
    block_size = __q4_block_size()
    blob_size = __q4_blob_size(block_size)
    k_blocks = (rows + block_size - 1) // block_size
    return k_blocks * cols * blob_size


def int4_block_quant(fp32weight: npt.ArrayLike, symmetric: bool) -> np.ndarray:
    """4b quantize fp32 weight to a blob"""

    if len(fp32weight.shape) != 2:
        raise ValueError("Current int4 block quantization only supports 2D tensors!")
    rows, cols = fp32weight.shape

    block_size = __q4_block_size()
    blob_size = __q4_blob_size(block_size)
    k_blocks = (rows + block_size - 1) // block_size
    padded_rows = k_blocks * block_size
    pad_len = padded_rows - rows
    if pad_len > 0:
        fp32weight = np.pad(fp32weight, ((0, pad_len), (0, 0)), "constant")

    # block wise quantization, each block comes from a single column
    packed = np.zeros((cols * k_blocks, blob_size), dtype="uint8")
    scales = np.zeros((cols * k_blocks), dtype=fp32weight.dtype)
    zero_point = np.zeros((cols * k_blocks), dtype="uint8")
    fp32weight = np.transpose(fp32weight)

    def _process_column(n):
        ncol = fp32weight[n, :]
        blks = np.split(ncol, k_blocks)
        blob_idx = n * k_blocks
        # print(f"start to process {n}")
        for blk in blks:
            packed_blob = packed[blob_idx]

            if symmetric:
                amax_idx = np.argmax(np.abs(blk))
                bmax = blk[amax_idx]
                scale = bmax / (-8)
                zp = 8
            else:
                vmin = np.min(blk)
                vmax = np.max(blk)
                vmin = min(vmin, 0.0)
                vmax = max(vmax, 0.0)
                scale = (vmax - vmin) / ((1 << 4) - 1)
                zero_point_fp = vmin
                if scale != 0.0:
                    zero_point_fp = 0.0 - vmin / scale
                zp = min(15, max(0, round(zero_point_fp)))

            reciprocal_scale = 1.0 / scale if scale != 0 else 0.0
            scales[blob_idx] = scale
            zero_point[blob_idx] = zp
            blob_idx += 1

            blk_int = np.clip(np.rint(blk * reciprocal_scale + zp), 0, 15).astype("uint8")
            for i in range(0, blob_size, 2):
                packed_blob[i // 2] = blk_int[i] | blk_int[i + 1] << 4
        # print(f"end to process {n}")

    with parallel_config(backend="threading", n_jobs=-1):
        Parallel()(delayed(_process_column)(n) for n in range(cols))
    return (
        packed.reshape((cols, k_blocks, blob_size)),
        scales.reshape((cols, k_blocks)),
        zero_point.reshape((cols, k_blocks)),
    )


class MatMulWeight4Quantizer:
    """Perform 4b quantization of constant MatMul weights"""

    ##################
    # quantization types, must be consistent with native code type
    # MLAS_BLK_QUANT_TYPE defined in mlas_q4.h

    # 32 number block, symmetric quantization, with one fp32 as scale, zero point is always 0
    BlkQ4Sym = 0

    # 32 number block, quantization, with one fp32 as scale, one uint8 zero point
    BlkQ4Zp8 = 1

    def __init__(self, model: ModelProto, quant_type: int):
        self.model = ONNXModel(model)
        self.quant_type = quant_type

    @staticmethod
    def __get_initializer(name, graph_path: List[GraphProto]) -> Tuple[TensorProto, GraphProto]:
        for gid in range(len(graph_path) - 1, -1, -1):
            graph = graph_path[gid]
            for tensor in graph.initializer:
                if tensor.name == name:
                    return tensor, graph
        return None, None

    def _q4_matmul_node_weight(self, node: NodeProto, graph_stack: List[GraphProto], symmetric) -> NodeProto:
        """If the node is MatMul with fp32 const weight, quantize the weight with int4, and return the new node"""

        if node.op_type != "MatMul":
            return node  # only care about MatMul for now

        inputB = node.input[1]  # noqa: N806
        B, Bs_graph = MatMulWeight4Quantizer.__get_initializer(inputB, graph_stack)  # noqa: N806
        if B is None:
            return node  # only care about constant weight

        # TODO!! assume B is not used by any other node
        B_array = onnx.numpy_helper.to_array(B)  # noqa: N806
        if len(B_array.shape) != 2:
            return node  # can only process 2-D matrix

        packed, scales, zero_points = int4_block_quant(B_array, symmetric)
        B_quant = onnx.numpy_helper.from_array(packed)  # noqa: N806
        B_quant.name = B.name + "_Q4"
        Bs_graph.initializer.remove(B)
        for input in Bs_graph.input:
            if input.name == inputB:
                Bs_graph.input.remove(input)
                break

        scales_tensor = onnx.numpy_helper.from_array(scales)  # noqa: N806
        scales_tensor.name = B.name + "_scales"
        Bs_graph.initializer.extend([B_quant, scales_tensor])

        input_names = [node.input[0], B_quant.name, scales_tensor.name]
        if not symmetric:
            zp_tensor = onnx.numpy_helper.from_array(zero_points)  # noqa: N806
            zp_tensor.name = B.name + "_zero_points"
            Bs_graph.initializer.extend([B_quant, zp_tensor])
            input_names.append(zp_tensor.name)

        kwargs = {}
        rows, cols = B_array.shape
        kwargs["K"] = rows
        kwargs["N"] = cols
        kwargs["bits"] = 4
        kwargs["block_size"] = 32

        matmul_q4_node = onnx.helper.make_node(
            "MatMulWithCompressWeight",
            inputs=input_names,
            outputs=[node.output[0]],
            name=node.name + "_Q4" if node.name else "",
            domain="com.microsoft",
            **kwargs,
        )
        return matmul_q4_node

    def _process_subgraph(self, graph_stack: List[GraphProto], symmetric):
        new_nodes = []
        graph = graph_stack[-1]

        for node in graph.node:
            print(node.name)
            graph_attrs = [
                attr
                for attr in node.attribute
                if attr.type == onnx.AttributeProto.GRAPH or attr.type == onnx.AttributeProto.GRAPHS
            ]
            if len(graph_attrs):
                kwargs = {}
                for attr in node.attribute:
                    if attr.type == onnx.AttributeProto.GRAPH:
                        # recursive call to take care of sub-graph
                        graph_stack.append(attr.g)
                        kv = {attr.name: self._process_subgraph(graph_stack)}
                    elif attr.type == onnx.AttributeProto.GRAPHS:
                        value = []
                        for subgraph in attr.graphs:
                            # recursive call to take care of sub-graph
                            graph_stack.append(subgraph)
                            value.extend([self._process_subgraph(graph_stack, symmetric)])
                        kv = {attr.name: value}
                    else:
                        kv = attribute_to_kwarg(attr)
                    kwargs.update(kv)
                node = onnx.helper.make_node(  # noqa: PLW2901
                    node.op_type, node.input, node.output, name=node.name, **kwargs
                )

            new_nodes.append(self._q4_matmul_node_weight(node, graph_stack, symmetric))

        graph.ClearField("node")
        graph.node.extend(new_nodes)
        graph_stack.pop()
        return graph

    def process(self, symmetric=True):
        # use a stack to keep track of sub-graphs
        graph_stack = [self.model.graph()]
        opset_import = self.model.opset_import()

        has_ms_domain = False
        for opset in opset_import:
            if opset.domain == "com.microsoft":
                has_ms_domain = True
        if not has_ms_domain:
            opset_import.extend([onnx.helper.make_opsetid("com.microsoft", 1)])

        self._process_subgraph(graph_stack, symmetric)


def parse_args():
    parser = argparse.ArgumentParser(
        description="""Blockwise int4 quantization for MatMul 2D weight matrices.

A weight matrix is partitioned into into blocks, where each block is a
continguous subset inside each column. Each block is quantized into a
set of 4b integers with a scaling factor and an optional offset.
"""
    )

    parser.add_argument("--input_model", required=True, help="Path to the input model file")
    parser.add_argument("--output_model", required=True, help="Path to the output model file")
    parser.add_argument("-e", "--use_external_data_format", required=False, action="store_true")
    parser.set_defaults(use_external_data_format=False)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    input_model_path = args.input_model
    output_model_path = args.output_model

    model = load_model_with_shape_infer(Path(input_model_path))
    quant = MatMulWeight4Quantizer(model, 0)
    quant.process()
    quant.model.save_model_to_file(output_model_path, True)
