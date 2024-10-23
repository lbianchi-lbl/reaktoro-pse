#################################################################################
# WaterTAP Copyright (c) 2020-2024, The Regents of the University of California,
# through Lawrence Berkeley National Laboratory, Oak Ridge National Laboratory,
# National Renewable Energy Laboratory, and National Energy Technology
# Laboratory (subject to receipt of any required approvals from the U.S. Dept.
# of Energy). All rights reserved.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license
# information, respectively. These files are also available online at the URL
# "https://github.com/watertap-org/reaktoro-pse/"
#################################################################################
import multiprocessing as mp

from matplotlib.pyplot import sca
from pytest import param
from reaktoro_pse.core.reaktoro_gray_box import (
    ReaktoroGrayBox,
)
from pyomo.contrib.pynumero.interfaces.external_grey_box import (
    ExternalGreyBoxBlock,
)
import numpy as np
from idaes.core.base.process_base import declare_process_block_class, ProcessBlockData


class ReaktoroBlockData:
    def __init__(self):
        self.state = None
        self.inputs = None
        self.outputs = None
        self.solver = None
        self.jacobian = None
        self.builder = None
        self.pseudo_gray_box = None


class AggregateSolverState:
    def __init__(self):
        self.hessian_type = None
        self.inputs = []
        self.input_dict = {}
        self.input_blk_indexes = []
        self.registered_blocks = None
        self.outputs = []
        self.output_blk_indexes = []
        self.jacobian_scaling_obj = []
        self.solver_functions = {}
        self.output_matrix = []
        self.jacobian_matrix = []
        self.input_windows = {}
        self.output_windows = {}

    def register_solve_function(self, block_index, solver_function):
        self.solver_functions[block_index] = solver_function

    def register_input(self, block_index, input_key, input_obj):
        self.inputs.append((block_index, input_key))
        self.input_dict[(block_index, input_key)] = input_obj
        self.input_dict[(block_index, input_key)].original_key = input_key
        self.input_blk_indexes.append(block_index)

    def register_output(self, block_index, output_key):
        self.outputs.append((block_index, output_key))
        self.output_blk_indexes.append(block_index)

        if len(self.output_blk_indexes) > 1:
            self.jacobian_matrix = np.zeros((len(self.outputs), len(self.inputs)))
            self.output_matrix = np.zeros(len(self.outputs))
            self.get_windows(block_index)

    def get_windows(self, block_idx):
        _, output_unique_sets = np.unique(self.output_blk_indexes, return_inverse=True)
        self.registered_blocks, input_unique_sets = np.unique(
            self.input_blk_indexes, return_inverse=True
        )
        input_idx = np.arange(len(self.inputs))
        output_idx = np.arange(len(self.outputs))
        output_start, output_end = min(
            output_idx[output_unique_sets == block_idx]
        ), max(output_idx[output_unique_sets == block_idx])
        input_start, input_end = min(input_idx[input_unique_sets == block_idx]), max(
            input_idx[input_unique_sets == block_idx]
        )
        self.input_windows[block_idx] = (
            input_start,
            input_end + 1,
        )  # need to offset by 1
        self.output_windows[block_idx] = (
            output_start,
            output_end + 1,
        )  # need to offset by 1

    def register_scaling_vals(self, scaling_values):
        self.jacobian_scaling_obj.append(scaling_values)

    def get_jacobian_scaling(self):
        scaling_array = None
        for obj in self.jacobian_scaling_obj:
            if scaling_array is None:
                scaling_array = obj
            else:
                np.hstack(scaling_array, obj)
        return scaling_array

    def get_params(self, block_idx, params):
        param_set = {}
        for (idx, key), item in params.items():
            if block_idx == idx:
                param_set[key] = item
        print(param_set)
        return param_set  # np.array(params)[
        #     self.input_windows[block_idx][0] : self.input_windows[block_idx][1]
        # ]

    def update_solution(self, block_idx, output, jacobian):
        print(self.input_windows[block_idx][0], self.input_windows[block_idx][1])
        self.output_matrix[
            self.output_windows[block_idx][0] : self.output_windows[block_idx][1]
        ] = output
        self.jacobian_matrix[
            self.output_windows[block_idx][0] : self.output_windows[block_idx][1],
            self.input_windows[block_idx][0] : self.input_windows[block_idx][1],
        ] = jacobian

    def solve_reaktoro_block(self, params):
        for blk in self.registered_blocks:
            param_set = self.get_params(blk, params)
            jacobian, output = self.solver_functions[blk](param_set)
            self.update_solution(blk, output, jacobian)
        return (
            self.jacobian_matrix,
            self.output_matrix,
        )


class PseudoGrayBox:
    def __init__(self):
        self.inputs = {}
        self.outputs = {}

    def register_input(self, aggregate_inputs, input_keys, block_idx):
        for input_key in input_keys:
            self.inputs[input_key] = aggregate_inputs[(block_idx, input_key)]

    def register_output(self, aggregate_outputs, output_keys, block_idx):
        for output_keys in output_keys:
            self.outputs[output_keys] = aggregate_outputs[(block_idx, output_keys)]


@declare_process_block_class("ReaktoroBlockManager")
class ReaktoroBlockManagerData(ProcessBlockData):
    def build(self):
        super().build()
        self.registered_blocks = []
        self.aggregate_solver_state = AggregateSolverState()

    def register_block(self, state, inputs, outputs, jacobian, solver, builder):
        blk = ReaktoroBlockData()
        blk.state = state
        blk.inputs = inputs
        blk.outputs = outputs
        blk.jacobian = jacobian
        blk.solver = solver
        blk.builder = builder

        self.registered_blocks.append(blk)
        return blk

    def aggregate_inputs_and_outputs(self):
        for block_idx, block in enumerate(self.registered_blocks):
            for key, obj in block.inputs.rkt_inputs.items():
                self.aggregate_solver_state.register_input(block_idx, key, obj)
            for output in block.outputs.rkt_outputs.keys():
                self.aggregate_solver_state.register_output(block_idx, output)
            self.aggregate_solver_state.register_solve_function(
                block_idx, block.solver.solve_reaktoro_block
            )

    def build_reaktoro_blocks(self):
        self.aggregate_inputs_and_outputs()
        external_model = ReaktoroGrayBox()
        external_model.configure(
            self.aggregate_solver_state,
            inputs=self.aggregate_solver_state.inputs,
            input_dict=self.aggregate_solver_state.input_dict,
            outputs=self.aggregate_solver_state.outputs,
            hessian_type="BGFS",  # TODO make it a config option
        )
        self.reaktoro_model = ExternalGreyBoxBlock(external_model=external_model)
        for block_idx, block in enumerate(self.registered_blocks):
            pseudo_gray_box_model = PseudoGrayBox()
            pseudo_gray_box_model.register_input(
                self.reaktoro_model.inputs,
                block.inputs.rkt_inputs.keys(),
                block_idx,
            )
            pseudo_gray_box_model.register_output(
                self.reaktoro_model.outputs,
                block.outputs.rkt_outputs.keys(),
                block_idx,
            )
            block.builder.build_reaktoro_block(
                gray_box_model=pseudo_gray_box_model,
                reaktoro_initialize_function=self.aggregate_solver_state.solver_functions[
                    block_idx
                ],
            )
            block.pseudo_gray_box = pseudo_gray_box_model
