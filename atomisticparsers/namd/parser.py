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
from typing import Dict, Any

from nomad.units import ureg
from nomad.datamodel import EntryArchive
from nomad.parsing.file_parser import TextParser, Quantity
from runschema.run import Run, Program
from runschema.method import Method
from simulationworkflowschema import MolecularDynamics
from atomisticparsers.utils import MDAnalysisParser, MDParser
from .metainfo import namd  # pylint: disable=unused-import


MOL = 6.022140857e23
re_f = r'[-+]?\d+\.\d*(?:[Ee][-+]\d+)?'
re_n = r'[\n\r]'


class ConfigParser(TextParser):
    def init_quantities(self):
        self._quantities = [
            Quantity(
                'parameter',
                rf'([\w\-]+) +(.+){re_n}',
                repeats=True,
                str_operation=lambda x: x.split(' ', 1),
            )
        ]

    def get_parameters(self):
        return {parameter[0]: parameter[1] for parameter in self.get('parameter', [])}


class MainfileParser(TextParser):
    def init_quantities(self):
        self._quantities = [
            Quantity(
                'version_arch',
                r'Info\: NAMD ([\d\.]+) for (.+)',
                str_operation=lambda x: x.split(' ', 1),
                convert=False,
            ),
            Quantity('config_file', r'Info\: Configuration file is (\S+)', dtype=str),
            Quantity(
                'simulation_parameters',
                r'Info\: SIMULATION PARAMETERS\:([\s\S]+?)Info\: SUMMARY',
                sub_parser=TextParser(
                    quantities=[
                        Quantity(
                            'parameter',
                            rf'Info\: ([A-Z][A-Z ]+) +([\d\.e\-\+]+)',
                            str_operation=lambda x: [
                                v.strip() for v in x.rsplit(' ', 1)
                            ],
                            repeats=True,
                        ),
                        Quantity(
                            'cell',
                            r'PERIODIC CELL BASIS \d +(.+)',
                            repeats=True,
                            dtype=np.dtype(np.float64),
                        ),
                        Quantity(
                            'output_file', r'Info\: OUTPUT FILENAME +(\S+)', dtype=str
                        ),
                        Quantity(
                            'coordinate_file',
                            r'Info\: COORDINATE PDB +(\S+)',
                            dtype=str,
                        ),
                        Quantity(
                            'structure_file', r'Info\: STRUCTURE FILE +(\S+)', dtype=str
                        ),
                        Quantity(
                            'parameter_file',
                            r'Info\: PARAMETERS +(\S+)',
                            dtype=str,
                            repeats=True,
                        ),
                    ]
                ),
            ),
            Quantity(
                'step',
                rf'ENERGY\: +(\d+ +{re_f}.+)',
                repeats=True,
                dtype=np.dtype(np.float64),
            ),
            Quantity(
                'timing',
                r'TIMING\: *(\d+) *CPU\: *[\d\.]+, +[\d\.]+/step +Wall\: *([\d\.]+), *([\d\.]+)/step',
                repeats=True,
                dtype=np.dtype(np.float64),
            ),
            Quantity('total_time', r'WallClock: +([\d\.]+)', dtype=float),
            Quantity(
                'property_names',
                r'ETITLE\: +(.+)',
                str_operation=lambda x: x.lower().strip().split(),
            ),
            Quantity(
                'coordinates_write_step',
                r'WRITING COORDINATES TO OUTPUT FILE AT STEP (\d+)',
                repeats=True,
                dtype=np.int32,
            ),
        ]

    def get_parameters(self):
        return {
            p[0]: p[1]
            for p in self.get('simulation_parameters', {}).get('parameter', [])
        }


class NAMDParser(MDParser):
    def __init__(self) -> None:
        self.mainfile_parser = MainfileParser()
        self.config_parser = ConfigParser()
        self.traj_parser = MDAnalysisParser()
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
            'gpressavg': 'x_namd_gpressure_average',
        }
        super().__init__()

    def write_to_archive(self) -> None:
        """
        Main parsing function. Populates the archive with the quantities parsed from the
        mainfile parser and auxilliary file parsers.
        """
        self.mainfile_parser.mainfile = self.mainfile
        self.mainfile_parser.logger = self.logger
        self.config_parser.logger = self.logger
        self.traj_parser.logger = self.logger
        self.maindir = os.path.dirname(self.mainfile)

        sec_run = Run()
        self.archive.run.append(sec_run)
        version_arch = self.mainfile_parser.get('version_arch', [None, None])
        sec_run.program = Program(name='namd', version=version_arch[0])
        sec_run.program.x_namd_build_osarch = version_arch[1]

        # read the config file and simulation parameters
        self.config_parser.mainfile = os.path.join(
            self.maindir, os.path.basename(self.mainfile_parser.get('config_file', ''))
        )
        sec_method = Method()
        sec_run.method.append(sec_method)
        sec_method.x_namd_input_parameters = self.config_parser.get_parameters()
        sec_method.x_namd_simulation_parameters = self.mainfile_parser.get_parameters()

        def get_system_data(index):
            if self.traj_parser.mainfile is None:
                return {}

            labels = self.traj_parser.get_atom_labels(index)
            positions = self.traj_parser.get_positions(index)
            velocities = self.traj_parser.get_velocities(index)
            lattice_vectors = self.traj_parser.get_lattice_vectors(index)
            if lattice_vectors is None:
                # get if from simulation parameters
                lattice_vectors = self.mainfile_parser.get(
                    'simulation_parameters', {}
                ).get('cell')
                lattice_vectors = (
                    lattice_vectors * ureg.angstrom
                    if lattice_vectors is not None
                    else lattice_vectors
                )
            return dict(
                atoms=dict(
                    labels=labels,
                    positions=positions,
                    velocities=velocities,
                    lattice_vectors=lattice_vectors,
                )
            )

        # input structure
        parameters = self.mainfile_parser.get('simulation_parameters', {})
        self.traj_parser.mainfile = os.path.join(
            self.maindir, parameters.get('coordinate_file')
        )

        # initial_system
        self.parse_trajectory_step(get_system_data(0))

        # energy unit is kcal / mol
        n_atoms = self.traj_parser.get('n_atoms')
        energy_unit = ureg.J * 4184.0 * n_atoms / MOL

        # trajectories
        # TODO other formats
        output_file = parameters.get('output_file')
        self.traj_parser.mainfile = os.path.join(self.maindir, f'{output_file}.coor')
        self.traj_parser.options = dict(format='coor')
        self.traj_parser.auxilliary_files = []

        # output properties at each step
        property_names = self.mainfile_parser.get('property_names', [])
        # saved trajectories
        saved_trajectories = self.mainfile_parser.get('coordinates_write_step', [])
        # md data
        steps_data = self.mainfile_parser.get('step', [])
        # set up md parser
        self.n_atoms = n_atoms
        self.trajectory_steps = [0] + saved_trajectories
        self.thermodynamics_steps = [int(step[0]) for step in steps_data]

        timings = self.mainfile_parser.get('timing', [])
        timing_step = sec_method.x_namd_simulation_parameters.get(
            'TIMING OUTPUT STEPS', 1
        )
        time_per_step = self.mainfile_parser.get('total_time', 0.0) / len(steps_data)

        for step in self.trajectory_steps:
            if not step or self.traj_parser.mainfile is None:
                continue

            index = saved_trajectories.index(step)
            self.parse_trajectory_step(get_system_data(index))

        for step_data in steps_data:
            step = int(step_data[0])
            if step not in self.thermodynamics_steps:
                continue

            energy: Dict[str, Any] = {}
            thermo_data = {'step': step, 'energy': energy}
            for index, name in enumerate(property_names):
                metainfo_name = self._metainfo_map.get(name)
                if metainfo_name is None:
                    continue
                value = step_data[index]
                if metainfo_name.startswith('energy_contribution_'):
                    energy.setdefault('contributions', [])
                    metainfo_name = metainfo_name.replace('energy_contribution_', '')
                    energy['contributions'].append(
                        dict(kind=metainfo_name, value=value * energy_unit)
                    )
                elif metainfo_name.startswith('energy_'):
                    metainfo_name = metainfo_name.replace('energy_', '')
                    energy[metainfo_name] = dict(value=value * energy_unit)
                    if metainfo_name == 'total':
                        # include potential and kinetic terms
                        for key in ['kinetic', 'potential']:
                            try:
                                energy['total'][key] = (
                                    step_data[property_names.index(key)] * energy_unit
                                )
                            except Exception:
                                pass
                elif 'pressure' in metainfo_name:
                    thermo_data[metainfo_name] = value * ureg.bar
                elif 'temperature' in metainfo_name:
                    thermo_data[metainfo_name] = value * ureg.kelvin
                elif 'volume' in metainfo_name:
                    thermo_data[metainfo_name] = value * ureg.angstrom**3
                # TODO forces
            # timing
            index_time = step // timing_step
            if index_time < len(timings):
                thermo_data['time_calculation'] = timings[index_time][2]
                thermo_data['time_physical'] = (
                    0 if index_time == 0 else timings[index_time - 1][1]
                ) + timings[index_time][2] * (step % timing_step + 1)
            elif time_per_step:
                thermo_data['time_calculation'] = time_per_step
                thermo_data['time_physical'] = time_per_step * step
            self.parse_thermodynamics_step(thermo_data)

        # workflow
        self.archive.workflow2 = MolecularDynamics()

        self.traj_parser.close()
