import os
from ctypes import *
from typing import List

import numpy as np
import vart
import xir


class VartInfer():
    def __init__(self,
                 model_path='model_data/first.xmodel'):
        super(VartInfer, self).__init__()

        self.ModelInit(model_path)

    """
    obtain dpu subgrah
    """

    def get_child_subgraph_dpu(self, graph: "Graph") -> List["Subgraph"]:
        assert graph is not None, "'graph' should not be None."
        root_subgraph = graph.get_root_subgraph()
        assert (root_subgraph
                is not None), "Failed to get root subgraph of input Graph object."
        if root_subgraph.is_leaf:
            return []
        child_subgraphs = root_subgraph.toposort_child_subgraph()
        assert child_subgraphs is not None and len(child_subgraphs) > 0
        return [
            cs for cs in child_subgraphs
            if cs.has_attr("device") and cs.get_attr("device").upper() == "DPU"
        ]

    def InputOutInit(self, runner: "Runner"):
        """get tensor"""
        inputTensors = runner.get_input_tensors()
        input_ndim = tuple(inputTensors[0].dims)
        """prepare batch input/output """
        inputData = np.empty(input_ndim, dtype=np.int8, order="C")
        input_fixpos = inputTensors[0].get_attr("fix_point")
        input_scale = 2**input_fixpos

        outputDatas = []
        output_scales = []
        outputTensors = runner.get_output_tensors()
        for outputTensor in outputTensors:
            output_ndim = tuple(outputTensor.dims)
            output_fixpos = outputTensor.get_attr("fix_point")
            output_scale = 1 / (2**output_fixpos)
            output_scales.append(output_scale)
            # print(outputTensor.name, output_ndim)
            outputDatas.append(np.empty(output_ndim, dtype=np.int8, order="C"))

        return inputData, outputDatas, input_scale, output_scales, input_ndim

    def ModelInit(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError("model path is not exists.")
        g = xir.Graph.deserialize(model_path)
        self.subgraphs = self.get_child_subgraph_dpu(g)
        assert len(self.subgraphs) == 2  # only one DPU kernel
        self.dpu_runner = vart.Runner.create_runner(self.subgraphs[1], "run")

        self.inputData, self.outputDatas, self.input_scale, self.output_scales, self.input_ndim = self.InputOutInit(
            self.dpu_runner)
        # print(self.inputData, self.outputDatas, self.input_scale, self.output_scales, self.input_ndim)

    def Inference(self, input, unfold2d_input):
        """init input image to input buffer """
        img = input * self.input_scale
        img = img.astype(np.int8)

        self.inputData[0, ...] = img.reshape(self.input_ndim[1:])

        """run with batch """
        job_id = self.dpu_runner.execute_async(self.inputData, self.outputDatas)
        self.dpu_runner.wait(job_id)
        # print("inference success")

        outputs = []
        for i in range(len(self.outputDatas)):
            tmp = self.outputDatas[i].transpose(0, 3, 1, 2)
            outputs.append(np.array(tmp, dtype='float32') * self.output_scales[i])

        return outputs[0], outputs[1]
