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

import os
import logging
import numpy as np

from nomad.units import ureg
from nomad.datamodel import EntryArchive
from nomad.parsing.file_parser import TextParser, Quantity
from nomad.datamodel.metainfo.simulation.run import (
    Run, Program
)
from nomad.datamodel.metainfo.simulation.method import (
    Method
)
from nomad.datamodel.metainfo.simulation.system import (
    System, Atoms
)
from nomad.datamodel.metainfo.simulation.calculation import (
    Calculation, Energy, EnergyEntry
)
from nomad.datamodel.metainfo.workflow import Workflow
from nomad.datamodel.metainfo.simulation import workflow as workflow2
from atomisticparsers.utils import MDAnalysisParser


MOL = 6.022140857e+23
re_f = r'[-+]?\d+\.\d*(?:[Ee][-+]\d+)?'
re_n = r'[\n\r]'


class ConfigParser(TextParser):
    def init_quantities(self):
        self._quantities = [
            Quantity(
                'parameter',
                rf'([\w\-]+) +(.+){re_n}',
                repeats=True, str_operation=lambda x: x.split(' ', 1))
        ]

    def get_parameters(self):
        return {parameter[0]: parameter[1] for parameter in self.get('parameter', [])}


class MainfileParser(TextParser):
    def init_quantities(self):
        self._quantities = [
            Quantity(
                'version_arch', r'Info\: NAMD ([\d\.]+) for (.+)',
                str_operation=lambda x: x.split(' ', 1), convert=False
            ),
            Quantity('config_file', r'Info\: Configuration file is (\S+)', dtype=str),
            Quantity(
                'simulation_parameters',
                r'Info\: SIMULATION PARAMETERS\:([\s\S]+?)Info\: SUMMARY',
                sub_parser=TextParser(quantities=[
                    Quantity(
                        'parameter',
                        rf'Info\: ([A-Z][A-Z ]+) +([\d\.e\-\+]+)',
                        str_operation=lambda x: [v.strip() for v in x.rsplit(' ', 1)], repeats=True),
                    Quantity(
                        'cell',
                        r'PERIODIC CELL BASIS \d +(.+)',
                        repeats=True, dtype=np.dtype(np.float64)
                    ),
                    Quantity('output_file', r'Info\: OUTPUT FILENAME +(\S+)', dtype=str),
                    Quantity('coordinate_file', r'Info\: COORDINATE PDB +(\S+)', dtype=str),
                    Quantity('structure_file', r'Info\: STRUCTURE FILE +(\S+)', dtype=str),
                    Quantity('parameter_file', r'Info\: PARAMETERS +(\S+)', dtype=str, repeats=True)
                ])
            ),
            Quantity(
                'step',
                rf'ENERGY\: +(\d+ +{re_f}.+)', repeats=True, dtype=np.dtype(np.float64)
            ),
            Quantity(
                'property_names',
                r'ETITLE\: +(.+)',
                str_operation=lambda x: x.lower().strip().split()),
            Quantity(
                'coordinates_write_step',
                r'WRITING COORDINATES TO OUTPUT FILE AT STEP (\d+)',
                repeats=True, dtype=np.int32)
        ]

    def get_parameters(self):
        return {p[0]: p[1] for p in self.get('simulation_parameters', {}).get('parameter', [])}


class NAMDParser:
    def __init__(self) -> None:
        self.mainfile_parser = MainfileParser()
        self.config_parser = ConfigParser()
        self.traj_parser = MDAnalysisParser()
        self._frame_rate = None
        # max cumulative number of atoms for all parsed trajectories to calculate sampling rate
        self._cum_max_atoms = 1000000
        self._metainfo_map = {
            'bond': 'energy_contribution_bond',
            'angle': 'energy_contribution_angle',
            'dihed': 'energy_contribution_dihedral',
            'imprp': 'energy_contribution_improper',
            'elect': 'energy_electronic',
            'vdw': 'energy_van_der_waals',
            'boundary': 'energy_contribution_boundary',
            'misc': 'energy_contribution_miscellaneous',
            'total': 'energy_total',
            'temp': 'temperature',
            'total3': 'energy_x_namd_total3',
            'tempavg': 'x_namd_temperature_average',
            'pressure': 'pressure',
            'gpressure': 'x_namd_gpressure',
            'volume': 'x_namd_volume',
            'pressavg': 'x_namd_pressure_average',
            'gpressavg': 'x_namd_gpressure_average'
        }

    @property
    def frame_rate(self):
        '''
        Sampling rate of the for trajectories calculated as the ratio of the cumulative
        number of atoms in all frames and the maximum allowable atoms in the trajectories
        to be parsed given by _cum_max_atoms.
        '''
        if self._frame_rate is None:
            n_atoms = self.traj_parser.get('n_atoms', 0)
            n_frames = self.traj_parser.get('n_frames', 0)
            if n_atoms == 0 or n_frames == 0:
                self._frame_rate = 1
            else:
                cum_atoms = n_atoms * n_frames
                self._frame_rate = 1 if cum_atoms <= self._cum_max_atoms else cum_atoms // self._cum_max_atoms
        return self._frame_rate

    def init_parser(self):
        '''
        Initializes the mainfile and auxilliary file parsers.
        '''
        self.mainfile_parser.mainfile = self.filepath
        self.mainfile_parser.logger = self.logger
        self.config_parser.logger = self.logger
        self.traj_parser.logger = self.logger
        self._frame_rate = None

    def parse(self, filepath: str, archive: EntryArchive, logger=None):
        '''
        Main parsing function. Populates the archive with the quantities parsed from the
        mainfile parser and auxilliary file parsers.
        '''
        self.filepath = os.path.abspath(filepath)
        self.maindir = os.path.dirname(filepath)
        self.archive = archive
        self.logger = logger if logger is not None else logging

        self.init_parser()
        sec_run = archive.m_create(Run)
        version_arch = self.mainfile_parser.get('version_arch', [None, None])
        sec_run.program = Program(name='namd', version=version_arch[0])
        sec_run.program.x_namd_build_osarch = version_arch[1]

        # read the config file and simulation parameters
        self.config_parser.mainfile = os.path.join(
            self.maindir, os.path.basename(self.mainfile_parser.get('config_file', '')))
        sec_method = sec_run.m_create(Method)
        sec_method.x_namd_input_parameters = self.config_parser.get_parameters()
        sec_method.x_namd_simulation_parameters = self.mainfile_parser.get_parameters()

        def parse_system(index=0):
            if self.traj_parser.mainfile is None:
                return

            sec_system = sec_run.m_create(System)
            sec_system.atoms = Atoms(
                labels=self.traj_parser.get_atom_labels(index), positions=self.traj_parser.get_positions(index),
                velocities=self.traj_parser.get_velocities(index))
            lattice_vectors = self.traj_parser.get_lattice_vectors(index)
            if lattice_vectors is None:
                # get if from simulation parameters
                lattice_vectors = self.mainfile_parser.get('simulation_parameters', {}).get('cell')
                lattice_vectors = lattice_vectors * ureg.angstrom if lattice_vectors is not None else lattice_vectors
            sec_system.atoms.lattice_vectors = lattice_vectors
            return sec_system

        # input structure
        parameters = self.mainfile_parser.get('simulation_parameters', {})
        self.traj_parser.mainfile = os.path.join(self.maindir, parameters.get('coordinate_file'))
        initial_system = parse_system()
        # energy unit is kcal / mol
        energy_unit = ureg.J * 4184.0 * self.traj_parser.get('n_atoms') / MOL

        # trajectories
        # TODO other formats
        output_file = parameters.get('output_file')
        self.traj_parser.mainfile = os.path.join(self.maindir, f'{output_file}.coor')
        self.traj_parser.options = dict(format='coor')
        self.traj_parser.auxilliary_files = []

        # output properties at each step
        property_names = self.mainfile_parser.get('property_names', [])
        # saved trajectories
        saved_trajectories = self.mainfile_parser.get('coordinates_write_step')

        for step_n, step in enumerate(self.mainfile_parser.get('step', [])):
            # parse only calculation if step coincides with frame rate sampling
            if (step_n % self.frame_rate) > 0:
                continue

            sec_calc = sec_run.m_create(Calculation)
            sec_energy = sec_calc.m_create(Energy)
            for index, name in enumerate(property_names):
                metainfo_name = self._metainfo_map.get(name)
                if metainfo_name is None:
                    continue
                value = step[index]
                if metainfo_name.startswith('energy_contribution_'):
                    metainfo_name = metainfo_name.replace('energy_contribution_', '')
                    sec_energy.contributions.append(EnergyEntry(kind=metainfo_name, value=value * energy_unit))
                elif metainfo_name.startswith('energy_'):
                    metainfo_name = metainfo_name.replace('energy_', '')
                    setattr(sec_energy, metainfo_name, EnergyEntry(value=value * energy_unit))
                    if metainfo_name == 'total':
                        # include potential and kinetic terms
                        for key in ['kinetic', 'potential']:
                            try:
                                setattr(sec_energy.total, key, step[property_names.index(key)] * energy_unit)
                            except Exception:
                                pass
                elif 'pressure' in metainfo_name:
                    setattr(sec_calc, metainfo_name, value * ureg.bar)
                elif 'temperature' in metainfo_name:
                    setattr(sec_calc, metainfo_name, value * ureg.kelvin)
                elif 'volume' in metainfo_name:
                    setattr(sec_calc, metainfo_name, value * ureg.angstrom ** 3)
                # TODO how about forces

            # parse system if coordinates are written to file
            time_step = step[0]
            if time_step in saved_trajectories:
                frame = saved_trajectories.index(time_step)
                sec_calc.system_ref = parse_system(frame)

        sec_run.calculation[0].system_ref = initial_system

        # workflow
        sec_workflow = self.archive.m_create(Workflow)
        sec_workflow.type = 'molecular_dynamics'
        self.archive.workflow2 = workflow2.MolecularDynamics()
