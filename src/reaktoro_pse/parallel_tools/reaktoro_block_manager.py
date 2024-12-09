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

import multiprocessing
from matplotlib.pyplot import sca
from pytest import param
from reaktoro_pse.core.reaktoro_gray_box import (
    ReaktoroGrayBox,
)
from reaktoro_pse.parallel_tools.parallel_manager import ReaktoroParallelManager

from reaktoro_pse.core.reaktoro_gray_box import HessTypes

from pyomo.contrib.pynumero.interfaces.external_grey_box import (
    ExternalGreyBoxBlock,
)
import numpy as np
from idaes.core.base.process_base import declare_process_block_class, ProcessBlockData
from pyomo.common.config import ConfigValue, IsInstance, ConfigDict


class ReaktoroBlockData:
    def __init__(self):
        self.state = None
        self.inputs = None
        self.outputs = None
        self.solver = None
        self.jacobian = None
        self.builder = None
        self.pseudo_gray_box = None

    def get_configs(self):
        configs = []
        configs.append(self.state.export_config())
        configs.append(self.inputs.export_config())
        configs.append(self.outputs.export_config())
        configs.append(self.jacobian.export_config())
        configs.append(self.solver.export_config())
        return configs

    def freeze_state(self):
        self.frozen_state = self.get_configs()


class AggregateSolverState:
    def __init__(self, parallel_mode=True, maximum_number_of_parallel_solves=None):
        self.hessian_type = None
        self.inputs = []
        self.input_dict = {}
        self.input_blk_indexes = []
        self.registered_blocks = None
        self.outputs = []
        self.output_blk_indexes = []
        self.jacobian_scaling_obj = []
        self.solver_functions = {}
        self.get_solution_function = {}
        self.output_matrix = []
        self.jacobian_matrix = []
        self.input_windows = {}
        self.output_windows = {}
        self.parallel_mode = parallel_mode
        if maximum_number_of_parallel_solves is None:
            maximum_number_of_parallel_solves = multiprocessing.cpu_count() - 1
        self.maximum_number_of_parallel_solves = maximum_number_of_parallel_solves

    def register_solve_function(self, block_index, solver_function):
        self.solver_functions[block_index] = solver_function

    def register_get_function(self, block_index, get_function):
        self.get_solution_function[block_index] = get_function

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
        return param_set

    def update_solution(self, block_idx, output, jacobian):
        self.output_matrix[
            self.output_windows[block_idx][0] : self.output_windows[block_idx][1]
        ] = output
        self.jacobian_matrix[
            self.output_windows[block_idx][0] : self.output_windows[block_idx][1],
            self.input_windows[block_idx][0] : self.input_windows[block_idx][1],
        ] = jacobian

    def solve_reaktoro_block(self, params):
        if self.parallel_mode:
            return self.parallel_solver(params)
        else:
            return self.serial_solver(params)

    def parallel_solver(self, params):
        active_workers = []
        for blk in self.registered_blocks:
            param_set = self.get_params(blk, params)
            self.solver_functions[blk](param_set)
            active_workers.append(blk)
            # if we have more than max workers,
            # collect any results that are ready first
            while len(active_workers) >= self.maximum_number_of_parallel_solves:
                for blk in active_workers:
                    result = self.get_solution_function[active_workers[0]]()
                    if result is not None:
                        jacobian, output = result
                        self.update_solution(blk, output, jacobian)
                        active_workers.pop(0)  # remove first worker
                        break
        # collect any results that are still not collected
        for blk in active_workers:
            jacobian, output = self.get_solution_function[blk]()
            self.update_solution(blk, output, jacobian)

        return (
            self.jacobian_matrix,
            self.output_matrix,
        )

    def serial_solver(self, params):
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
    CONFIG = ProcessBlockData.CONFIG()
    CONFIG.declare(
        "hessian_type",
        ConfigValue(
            default="BFGS",
            domain=IsInstance((str, HessTypes)),
            description="Hessian type to use for reaktor gray box",
            doc="""Hessian type to use, some might provide better stability
                options (Jt.J, BFGS, BFGS-mod, BFGS-damp, BFGS-ipopt""",
        ),
    )
    CONFIG.declare(
        "use_parallel_mode",
        ConfigValue(
            default=True,
            domain=bool,
            description="Enables use of parallel workers",
            doc="""If true, will parallelize all reaktoro solver calls using multiprocessing""",
        ),
    )
    CONFIG.declare(
        "worker_timeout",
        ConfigValue(
            default=20,
            domain=int,
            description="Defines time in seconds for worker time out",
            doc="""This is time out for parallel workers to time out and shut down if they receive no 
            commands from main process (e.g. flowsheet)""",
        ),
    )
    CONFIG.declare(
        "maximum_number_of_parallel_solves",
        ConfigValue(
            default=None,
            domain=int,
            description="Maximum number of parallel solves",
            doc="""This will limit how many parallel solves would be run, this will not reduce number of 
            spawned processes""",
        ),
    )

    def build(self):
        super().build()
        self.registered_blocks = []
        self.aggregate_solver_state = AggregateSolverState(
            parallel_mode=self.config.use_parallel_mode,
            maximum_number_of_parallel_solves=self.config.maximum_number_of_parallel_solves,
        )
        if self.config.use_parallel_mode:
            self.parallel_manager = ReaktoroParallelManager(
                self.config.worker_timeout,
            )

    def register_block(self, state, inputs, outputs, jacobian, solver, builder):
        blk = ReaktoroBlockData()
        blk.state = state
        blk.inputs = inputs
        blk.outputs = outputs
        blk.jacobian = jacobian
        blk.solver = solver
        blk.builder = builder
        blk.freeze_state()

        self.registered_blocks.append(blk)
        return blk

    def aggregate_inputs_and_outputs(self):
        for block_idx, block in enumerate(self.registered_blocks):
            if self.config.use_parallel_mode:
                self.parallel_manager.register_block(block_idx, block)
            for key, obj in block.inputs.rkt_inputs.items():
                self.aggregate_solver_state.register_input(block_idx, key, obj)
            for output in block.outputs.rkt_outputs.keys():
                self.aggregate_solver_state.register_output(block_idx, output)
            if self.config.use_parallel_mode:
                solve_func, get_func = self.parallel_manager.get_solve_and_get_function(
                    block_idx
                )
                self.aggregate_solver_state.register_solve_function(
                    block_idx, solve_func
                )
                self.aggregate_solver_state.register_get_function(block_idx, get_func)
            else:
                self.aggregate_solver_state.register_solve_function(
                    block_idx, block.solver.solve_reaktoro_block
                )
        if self.config.use_parallel_mode:
            self.parallel_manager.start_workers()

    def build_reaktoro_blocks(self):
        self.aggregate_inputs_and_outputs()
        external_model = ReaktoroGrayBox()
        external_model.configure(
            self.aggregate_solver_state,
            inputs=self.aggregate_solver_state.inputs,
            input_dict=self.aggregate_solver_state.input_dict,
            outputs=self.aggregate_solver_state.outputs,
            hessian_type=self.config.hessian_type,
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
            if self.config.use_parallel_mode:
                init_func = self.parallel_manager.get_initialize_function(block_idx)
                disp_func = self.parallel_manager.get_display_function(block_idx)
            else:
                init_func = self.aggregate_solver_state.solver_functions[block_idx]
                disp_func = None
            block.builder.build_reaktoro_block(
                gray_box_model=pseudo_gray_box_model,
                reaktoro_initialize_function=init_func,
                display_reaktoro_state_function=disp_func,
            )
            block.pseudo_gray_box = pseudo_gray_box_model

    def terminate_workers(self):
        if self.config.use_parallel_mode:
            self.parallel_manager.terminate_workers()
