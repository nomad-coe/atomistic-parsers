#
# Copyright The NOMAD Authors.
#
# This file is part of NOMAD.
# See https://nomad-lab.eu for further info.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from typing import Any, Dict, List, Union
import numpy as np
from collections.abc import Iterable

from nomad.utils import get_logger
from nomad.metainfo import MSection, SubSection, Quantity
from nomad.parsing.file_parser import Parser
from runschema.run import Run
from runschema.system import System
from runschema.calculation import Calculation
from runschema.method import Interaction, Model
from simulationworkflowschema import MolecularDynamics


class MDParser(Parser):
    def __init__(self, **kwargs) -> None:
        self.info: Dict[str, Any] = {}
        self.cum_max_atoms: int = 2500000
        self.logger = get_logger(__name__)
        self._trajectory_steps: List[int] = []
        self._thermodynamics_steps: List[int] = []
        self._trajectory_steps_sampled: List[int] = []
        self._steps: List[int] = []
        super().__init__(**kwargs)

    @property
    def steps(self) -> List[int]:
        """
        Returns the set of trajectory and thermodynamics steps.
        """
        if not self._steps:
            self._steps = list(set(self.trajectory_steps + self.thermodynamics_steps))
            self._steps.sort()
        return self._steps

    @property
    def trajectory_steps(self) -> List[int]:
        """
        Returns the sampled trajectory steps.
        """
        if not self._trajectory_steps_sampled:
            self._trajectory_steps_sampled = [
                step
                for n, step in enumerate(self._trajectory_steps)
                if n % self.archive_sampling_rate == 0
            ]
        return self._trajectory_steps_sampled

    @trajectory_steps.setter
    def trajectory_steps(self, value: List[int]):
        self._trajectory_steps = list(set(value))
        self._trajectory_steps.sort()
        self.info['n_frames'] = len(self._trajectory_steps)
        self._trajectory_steps_sampled = []

    @property
    def thermodynamics_steps(self) -> List[int]:
        """
        Returns the thermodynamics steps.
        """
        # TODO is it necessary to sample thermodynamics steps
        return self._thermodynamics_steps

    @thermodynamics_steps.setter
    def thermodynamics_steps(self, value: List[int]):
        self._thermodynamics_steps = list(set(value))
        self._thermodynamics_steps.sort()

    @property
    def n_atoms(self) -> int:
        return np.amax(self.info.get('n_atoms', [0]))

    @n_atoms.setter
    def n_atoms(self, value: Union[Iterable, int]):
        self.info['n_atoms'] = [value] if not isinstance(value, Iterable) else value

    @property
    def archive_sampling_rate(self) -> int:
        """
        Returns the sampling rate of saved thermodynamics data and trajectory.
        """
        if self.info.get('archive_sampling_rate') is None:
            n_frames = self.info.get('n_frames', len(self._trajectory_steps))
            n_atoms = np.amax(self.n_atoms)
            if not n_atoms or not n_frames:
                self.info['archive_sampling_rate'] = 1
            else:
                cum_atoms = n_atoms * n_frames
                self.info['archive_sampling_rate'] = (
                    1
                    if cum_atoms <= self.cum_max_atoms
                    else -(-cum_atoms // self.cum_max_atoms)
                )
        return self.info.get('archive_sampling_rate')

    def parse(self, *args, **kwargs):
        self.info = {}
        self.trajectory_steps = []
        self.thermodynamics_steps = []
        self._steps = []
        self._trajectory_steps_sampled = []
        super().parse(*args, **kwargs)

    def parse_trajectory_step(self, data: Dict[str, Any]) -> None:
        """
        Create a system section and write the provided data.
        """
        if self.archive is None:
            return

        if (step := data.get('step')) is not None and step not in self.trajectory_steps:
            return

        if self.archive.run:
            sec_run = self.archive.run[-1]
        else:
            sec_run = Run()
            self.archive.run.append(sec_run)

        sec_system = System()
        sec_run.system.append(sec_system)
        self.parse_section(data, sec_system)

    def parse_thermodynamics_step(self, data: Dict[str, Any]) -> None:
        """
        Create a calculation section and write the provided data.
        """
        if self.archive is None:
            return

        if (
            step := data.get('step')
        ) is not None and step not in self.thermodynamics_steps:
            return

        if self.archive.run:
            sec_run = self.archive.run[-1]
        else:
            sec_run = Run()
            self.archive.run.append(sec_run)
        sec_calc = Calculation()
        sec_run.calculation.append(sec_calc)

        self.parse_section(data, sec_calc)
        try:
            system_ref_index = self.trajectory_steps.index(sec_calc.step)
            sec_calc.system_ref = sec_run.system[system_ref_index]
        except Exception:
            pass

    def parse_md_workflow(self, data: Dict[str, Any]) -> None:
        """
        Create an md workflow section and write the provided data.
        """
        if self.archive is None:
            return

        sec_workflow = MolecularDynamics()
        self.parse_section(data, sec_workflow)
        self.archive.workflow2 = sec_workflow

    def parse_interactions(self, interactions: List[Dict], sec_model: MSection) -> None:
        if not interactions:
            return

        def write_interaction_values(values):
            sec_interaction = Interaction()
            sec_model.contributions.append(sec_interaction)
            sec_interaction.type = current_type
            sec_interaction.n_atoms = max(
                [len(v) for v in values.get('atom_indices', [[0]])]
            )
            for key, val in values.items():
                quantity_def = sec_interaction.m_def.all_quantities.get(key)
                if quantity_def:
                    try:
                        sec_interaction.m_set(quantity_def, val)
                    except Exception:
                        self.logger.error('Error setting metadata.', data={'key': key})

        interactions.sort(key=lambda x: x.get('type'))
        current_type = interactions[0].get('type')
        interaction_values: Dict[str, Any] = {}
        for interaction in interactions:
            interaction_type = interaction.get('type')
            if current_type and current_type != interaction_type:
                write_interaction_values(interaction_values)
                current_type = interaction_type
                interaction_values = {}
            interaction_values.setdefault('n_interactions', 0)
            interaction_values['n_interactions'] += 1
            for key, val in interaction.items():
                if key == 'type':
                    continue
                interaction_values.setdefault(key, [])
                interaction_values[key].append(val)
        if interaction_values:
            write_interaction_values(interaction_values)

    def parse_interactions_by_type(
        self, interactions_by_type: List[Dict], sec_model: Model
    ) -> None:
        for interaction_type_dict in interactions_by_type:
            sec_interaction = Interaction()
            sec_model.contributions.append(sec_interaction)
            self.parse_section(interaction_type_dict, sec_interaction)
        # TODO Shift Gromacs and Lammps parsers to use this function as well if possible
