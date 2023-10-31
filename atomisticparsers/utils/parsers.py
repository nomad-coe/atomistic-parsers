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
from collections.abc import Iterable

from nomad.utils import get_logger
from nomad.datamodel import EntryArchive
from nomad.metainfo import MSection, SubSection
from nomad.datamodel.metainfo.simulation.run import Run
from nomad.datamodel.metainfo.simulation.system import System
from nomad.datamodel.metainfo.simulation.calculation import Calculation
from nomad.datamodel.metainfo.simulation.workflow import MolecularDynamics
from nomad.datamodel.metainfo.simulation.method import Interaction


# TODO put this in nomad.parsing
class AtomisticParser:
    def __init__(self, **kwargs) -> None:
        self._info: Dict[str, Any] = {}
        self._archive: EntryArchive = kwargs.get('archive')
        for key, val in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, val)

    @property
    def archive(self) -> EntryArchive:
        return self._archive

    @archive.setter
    def archive(self, value: EntryArchive):
        self._info = {}
        self._archive = value

    def get(self, key: str, default: Any = None) -> Any:
        '''
        Returns variables stored in _info, if not the provided default.
        '''
        return self._info.get(key, default)

    def parse_section(self, data: Dict[str, Any], root: MSection) -> None:
        '''
        Write the quantities in data into an archive section.
        '''
        for key, val in data.items():
            if not hasattr(root, key):
                continue

            if isinstance((section := getattr(root.m_def.section_cls, key)), SubSection):
                for val_n in [val] if isinstance(val, dict) else val:
                    self.parse_section(val_n, root.m_create(section.sub_section.section_cls, section))
                continue

            root.m_set(root.m_get_quantity_definition(key), val)


class MDParser(AtomisticParser):
    def __init__(self, **kwargs) -> None:
        self.cum_max_atoms: int = 2500000
        self.logger = get_logger(__name__)
        self._trajectory_steps: List[int] = []
        self._thermodynamics_steps: List[int] = []
        self._trajectory_steps_sampled: List[int] = []
        self._steps: List[int] = []
        super().__init__(**kwargs)

    @property
    def steps(self) -> List[int]:
        '''
        Returns the set of trajectory and thermodynamics steps.
        '''
        if not self._steps:
            self._steps = list(set(self.trajectory_steps + self.thermodynamics_steps))
            self._steps.sort()
        return self._steps

    @property
    def trajectory_steps(self) -> List[int]:
        '''
        Returns the sampled trajectory steps.
        '''
        if not self._trajectory_steps_sampled:
            self._trajectory_steps_sampled = [
                step for n, step in enumerate(self._trajectory_steps) if n % self.archive_sampling_rate == 0]
        return self._trajectory_steps_sampled

    @trajectory_steps.setter
    def trajectory_steps(self, value: List[int]):
        self._trajectory_steps = list(set(value))
        self._trajectory_steps.sort()
        self._info['n_frames'] = len(self._trajectory_steps)
        self._trajectory_steps_sampled = []

    @property
    def thermodynamics_steps(self) -> List[int]:
        '''
        Returns the thermodynamics steps.
        '''
        # TODO is it necessary to sample thermodynamics steps
        return self._thermodynamics_steps

    @thermodynamics_steps.setter
    def thermodynamics_steps(self, value: List[int]):
        self._thermodynamics_steps = list(set(value))
        self._thermodynamics_steps.sort()

    @property
    def n_atoms(self) -> int:
        return np.amax(self._info.get('n_atoms', [0]))

    @n_atoms.setter
    def n_atoms(self, value):
        self._info['n_atoms'] = [value] if not isinstance(value, Iterable) else value

    @property
    def archive_sampling_rate(self) -> int:
        '''
        Returns the sampling rate of saved thermodynamics data and trajectory.
        '''
        if self.get('archive_sampling_rate') is None:
            n_frames = self.get('n_frames', len(self._trajectory_steps))
            n_atoms = np.amax(self.n_atoms)
            if not n_atoms or not n_frames:
                self._info['archive_sampling_rate'] = 1
            else:
                cum_atoms = n_atoms * n_frames
                self._info['archive_sampling_rate'] = 1 if cum_atoms <= self.cum_max_atoms else -(-cum_atoms // self.cum_max_atoms)
        return self.get('archive_sampling_rate')

    @property
    def archive(self) -> EntryArchive:
        return self._archive

    @archive.setter
    def archive(self, value: EntryArchive):
        self._info = {}
        self.trajectory_steps = []
        self.thermodynamics_steps = []
        self._steps = []
        self._trajectory_steps_sampled = []
        self._archive = value

    def parse_trajectory_step(self, data: Dict[str, Any]) -> None:
        '''
        Create a system section and write the provided data.
        '''
        if self.archive is None:
            return

        if (step := data.get('step')) is not None and step not in self.trajectory_steps:
            return

        sec_run = self.archive.run[-1] if self.archive.run else self.archive.m_create(Run)

        self.parse_section(data, sec_run.m_create(System))

    def parse_thermodynamics_step(self, data: Dict[str, Any]) -> None:
        '''
        Create a calculation section and write the provided data.
        '''
        if self.archive is None:
            return

        if (step := data.get('step')) is not None and step not in self.thermodynamics_steps:
            return

        sec_run = self.archive.run[-1] if self.archive.run else self.archive.m_create(Run)
        sec_calc = sec_run.m_create(Calculation)

        self.parse_section(data, sec_calc)
        try:
            system_ref_index = self.trajectory_steps.index(sec_calc.step)
            sec_calc.system_ref = sec_run.system[system_ref_index]
        except Exception:
            pass

    def parse_md_workflow(self, data: Dict[str, Any]) -> None:
        '''
        Create an md workflow section and write the provided data.
        '''
        if self.archive is None:
            return

        sec_workflow = MolecularDynamics()
        self.parse_section(data, sec_workflow)
        self.archive.workflow2 = sec_workflow

    def parse_interactions(self, interactions: Dict[str, Any], sec_model: MSection) -> None:

        interaction_key_list = Interaction.__dict__.keys()
        interaction_dict = {}
        interaction_keys_remove = ['__module__', '__doc__', 'm_def']
        interaction_key_list = [key for key in interaction_key_list if key not in interaction_keys_remove]
        for interaction_key in interaction_key_list:
            interaction_dict[interaction_key] = np.array([interaction.get(interaction_key) for interaction in interactions], dtype=object)
        interaction_dict = {key: val for key, val in interaction_dict.items()}
        interaction_types = np.unique(interaction_dict['type']) if interaction_dict.get('type') is not None else []
        for interaction_type in interaction_types:
            sec_interaction = sec_model.m_create(Interaction)
            interaction_indices = np.where(interaction_dict['type'] == interaction_type)[0]
            sec_interaction.type = interaction_type
            sec_interaction.n_inter = len(interaction_indices)
            sec_interaction.n_atoms
            for key, val in interaction_dict.items():
                if key == 'type':
                    continue
                interaction_vals = val[interaction_indices]
                if type(interaction_vals[0]).__name__ == 'ndarray':
                    interaction_vals = np.array([vals.tolist() for vals in interaction_vals], dtype=object)
                if interaction_vals.all() is None:
                    continue
                if key == 'parameters':
                    interaction_vals = interaction_vals.tolist()
                elif key == 'n_atoms':
                    interaction_vals = interaction_vals[0]
                try:
                    setattr(sec_interaction, key, interaction_vals)
                except Exception:
                    self.logger.warning(
                        'Some issue trying to store ' + key + 'in Interactions section.'
                        ' Possibly a data type problem. Ignoring these values.')

            if not sec_interaction.get('n_atoms'):
                sec_interaction.n_atoms = len(sec_interaction.get('atom_indices')[0]) if sec_interaction.get('atom_indices') is not None else None
