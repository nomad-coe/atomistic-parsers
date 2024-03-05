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

from typing import Any, Dict, List
import numpy as np
import inspect
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
        self.info["n_frames"] = len(self._trajectory_steps)
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
        return np.amax(self.info.get("n_atoms", [0]))

    @n_atoms.setter
    def n_atoms(self, value):
        self.info["n_atoms"] = [value] if not isinstance(value, Iterable) else value

    @property
    def archive_sampling_rate(self) -> int:
        """
        Returns the sampling rate of saved thermodynamics data and trajectory.
        """
        if self.info.get("archive_sampling_rate") is None:
            n_frames = self.info.get("n_frames", len(self._trajectory_steps))
            n_atoms = np.amax(self.n_atoms)
            if not n_atoms or not n_frames:
                self.info["archive_sampling_rate"] = 1
            else:
                cum_atoms = n_atoms * n_frames
                self.info["archive_sampling_rate"] = (
                    1
                    if cum_atoms <= self.cum_max_atoms
                    else -(-cum_atoms // self.cum_max_atoms)
                )
        return self.info.get("archive_sampling_rate")

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

        if (step := data.get("step")) is not None and step not in self.trajectory_steps:
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
            step := data.get("step")
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
        interaction_dict = {}
        for interaction_key in Interaction.m_def.all_quantities.keys():
            interaction_dict[interaction_key] = np.array(
                [interaction.get(interaction_key) for interaction in interactions],
                dtype=object,
            )
        interaction_dict = {key: val for key, val in interaction_dict.items()}
        interaction_types = (
            np.unique(interaction_dict["type"])
            if interaction_dict.get("type") is not None
            else []
        )
        for interaction_type in interaction_types:
            sec_interaction = Interaction()
            sec_model.contributions.append(sec_interaction)
            interaction_indices = np.where(
                interaction_dict["type"] == interaction_type
            )[0]
            sec_interaction.type = interaction_type
            sec_interaction.n_interactions = len(interaction_indices)
            sec_interaction.n_atoms
            for key, val in interaction_dict.items():
                if key == "type":
                    continue
                interaction_vals = val[interaction_indices]
                if type(interaction_vals[0]).__name__ == "ndarray":
                    interaction_vals = np.array(
                        [vals.tolist() for vals in interaction_vals], dtype=object
                    )
                if interaction_vals.all() is None:
                    continue
                if key == "parameters":
                    interaction_vals = interaction_vals.tolist()
                elif key == "n_atoms":
                    interaction_vals = interaction_vals[0]
                if hasattr(sec_interaction, key):
                    sec_interaction.m_set(
                        sec_interaction.m_get_quantity_definition(key), interaction_vals
                    )

            if not sec_interaction.n_atoms:
                sec_interaction.n_atoms = (
                    len(sec_interaction.get("atom_indices")[0])
                    if sec_interaction.get("atom_indices") is not None
                    else None
                )

    def parse_interactions_by_type(
        self, interactions_by_type: List[Dict], sec_model: Model
    ) -> None:
        for interaction_type_dict in interactions_by_type:
            sec_interaction = Interaction()
            sec_model.contributions.append(sec_interaction)
            self.parse_section(interaction_type_dict, sec_interaction)
        # TODO Shift Gromacs and Lammps parsers to use this function as well if possible
