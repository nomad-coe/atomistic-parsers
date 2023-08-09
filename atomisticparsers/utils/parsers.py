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
from collections import Iterable

from nomad.utils import get_logger
from nomad.datamodel import EntryArchive
from nomad.datamodel.metainfo.simulation.run import Run
from nomad.datamodel.metainfo.simulation.system import System
from nomad.datamodel.metainfo.simulation.calculation import Calculation
from nomad.metainfo import MSection, SubSection


class MDParser:
    def __init__(self, **kwargs) -> None:
        self._info: Dict[str, Any] = {}
        self._archive: EntryArchive = kwargs.get('archive')
        self.thermodynamics_quantities: List[str] = ['pressure', 'temperature', 'time']
        self.cum_max_atoms: int = 2500000
        self.logger = get_logger(__name__)
        self._trajectory_steps = []
        self._thermodynamics_steps = []
        self._trajectory_steps_sampled = []
        self._thermodynamics_steps_sampled = []
        self._steps = []
        self._steps_sampled = []
        for key, val in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, val)

    @property
    def steps(self):
        '''
        Returns the sampled steps. Steps are the set of trajectory and thermodynamics steps.
        '''
        if not self._steps_sampled:
            self._steps.sort()
            self._info['n_frames'] = len(self._steps)
            self._steps_sampled = list(self._steps)
            if self.frame_rate > 1:
                self._steps_sampled = [step for n, step in enumerate(self._steps) if n % self.frame_rate == 0]
        return self._steps_sampled

    @property
    def trajectory_steps(self):
        '''
        Returns the sampled trajectory steps.
        '''
        if not self._trajectory_steps_sampled:
            self._trajectory_steps_sampled = [
                step for step in self._trajectory_steps if step in self.steps]
        return self._trajectory_steps_sampled

    @trajectory_steps.setter
    def trajectory_steps(self, value):
        self._trajectory_steps = value
        self._trajectory_steps.sort()
        self._trajectory_steps_sampled = []
        self._steps = list(set(self._trajectory_steps + self._thermodynamics_steps))

    @property
    def thermodynamics_steps(self):
        '''
        Returns the sampled thermodynamics steps.
        '''
        if not self._thermodynamics_steps_sampled:
            self._thermodynamics_steps_sampled = [
                step for step in self._thermodynamics_steps if step in self.steps]
        return self._thermodynamics_steps_sampled

    @thermodynamics_steps.setter
    def thermodynamics_steps(self, value):
        self._thermodynamics_steps = value
        self._thermodynamics_steps.sort()
        self._thermodynamics_steps_sampled = []
        self._steps = list(set(self._trajectory_steps + self._thermodynamics_steps))

    @property
    def n_atoms(self):
        return np.amax(self._info.get('n_atoms', [0]))

    @n_atoms.setter
    def n_atoms(self, value):
        self._info['n_atoms'] = [value] if not isinstance(value, Iterable) else value

    @property
    def frame_rate(self) -> int:
        '''
        Returns the sampling rate of saved thermodynamics data and trajectory.
        '''
        if self.get('frame_rate') is None:
            n_frames = self.get('n_frames', len(self._steps))
            n_atoms = np.amax(self.n_atoms)
            if n_atoms == 0 or n_frames == 0:
                self._info['frame_rate'] = 1
            else:
                cum_atoms = n_atoms * n_frames
                self._info['frame_rate'] = 1 if cum_atoms <= self.cum_max_atoms else -(-cum_atoms // self.cum_max_atoms)
        return self.get('frame_rate')

    @property
    def archive(self):
        return self._archive

    @archive.setter
    def archive(self, value):
        self._info = {}
        self.trajectory_steps = []
        self.thermodynamics_steps = []
        self._steps = []
        self._trajectory_steps_sampled = []
        self._thermodynamics_steps_sampled = []
        self._steps_sampled = []
        self._archive = value

    def get(self, key: str, default: Any = None) -> Any:
        return self._info.get(key, default)

    def parse_section(self, data: Dict[str, Any], root: MSection):
        for key, val in data.items():
            if not hasattr(root, key):
                continue

            if isinstance((section := getattr(root.m_def.section_cls, key)), SubSection):
                for val_n in [val] if isinstance(val, dict) else val:
                    self.parse_section(val_n, root.m_create(section.sub_section.section_cls, section))
                continue

            root.m_set(root.m_get_quantity_definition(key), val)

    def parse_trajectory_step(self, data: Dict[str, Any]) -> None:
        if data.get('step') not in self.trajectory_steps:
            return

        sec_run = self.archive.run[-1] if self.archive.run else self.archive.m_create(Run)

        self.parse_section(data, sec_run.m_create(System))

    def parse_thermodynamics_step(self, data: Dict[str, Any]) -> None:
        if data.get('step') not in self.thermodynamics_steps:
            return

        sec_run = self.archive.run[-1] if self.archive.run else self.archive.m_create(Run)
        sec_calc = sec_run.m_create(Calculation)

        self.parse_section(data, sec_calc)
        try:
            system_ref_index = self.trajectory_steps.index(sec_calc.step)
            sec_calc.system_ref = sec_run.system[system_ref_index]
        except Exception:
            pass
