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
import numpy as np
import fnmatch
from typing import List
import re

from nomad.parsing.file_parser.text_parser import Quantity, TextParser
from nomad.units import ureg
from runschema.run import Run, Program
from runschema.system import System, Atoms
from runschema.method import Method, ForceField, Model, AtomParameters
from runschema.calculation import (
    Calculation,
    Energy,
    EnergyEntry,
    VibrationalFrequencies,
)
from simulationworkflowschema import GeometryOptimization, GeometryOptimizationMethod
from atomisticparsers.utils import MDAnalysisParser, MDParser
from .metainfo.tinker import x_tinker_section_control_parameters

re_n = r'[\n\r]'
re_f = r'[-+]?\d+\.\d*(?:[DdEe][-+]\d+)?'
mol = (1 * ureg.mol).to('particle').magnitude


class KeyParser(TextParser):
    def __init__(self):
        super().__init__()

    def init_quantities(self):
        def to_key_val(val_in):
            val = val_in.strip().split()
            if len(val) == 1:
                return [val[0], True]
            elif len(val) == 2:
                val[1] = float(val[1]) if re.match(re_f, val[1]) else val[1]
                return val
            else:
                return [val[0], val[1:]]

        self._quantities = [
            Quantity(
                'key_val',
                rf'([a-z\-]+) *([^#]*?) *{re_n}',
                str_operation=to_key_val,
                repeats=True,
                convert=False,
            )
        ]


class RunParser(TextParser):
    def __init__(self):
        super().__init__()

    def init_quantities(self):
        def to_argument(val_in):
            val_in = val_in.strip().split()
            argument = dict()
            if '-k' in val_in:
                argument['key'] = val_in.pop(val_in.index('-k') + 1)
                val_in.remove('-k')
            argument['name'] = val_in.pop(0)
            argument['parameters'] = val_in
            return argument

        self._quantities = [
            Quantity(
                'molecular_dynamics',
                r'dynamic +(\S+ +\-*k* *\S+.+)',
                repeats=True,
                str_operation=to_argument,
            ),
            Quantity(
                'geometry_optimization',
                r'minimize +(\S+ +\-*k* *\S+.+)',
                repeats=True,
                str_operation=to_argument,
            ),
            Quantity(
                'single_point',
                r'vibrate +(\S+ +\-*k* *\S+.+)',
                repeats=True,
                str_operation=to_argument,
            ),
        ]


class OutParser(TextParser):
    def __init__(self):
        super().__init__()

    def init_quantities(self):
        iteration_quantity = Quantity(
            'iteration',
            rf'Iter.+([\s\S]+?){re_n} *{re_n}',
            sub_parser=TextParser(
                quantities=[
                    Quantity(
                        'step',
                        rf' +\d+ +({re_f} +{re_f} +[\d\.DE\+\- ]+)',
                        str_operation=lambda x: [
                            float(v)
                            for v in x.replace('D+', 'E+').replace('d+', 'e+').split()
                        ],
                        repeats=True,
                        dtype=np.dtype(np.float64),
                    )
                ]
            ),
        )

        calculation_quantities = [
            Quantity('program_version', r'Version ([\d\.]+)', dtype=str),
            Quantity(
                'vibrate',
                r'(Eigenvalues of the Hessian Matrix[\s\S]+?)\Z',
                sub_parser=TextParser(
                    quantities=[
                        Quantity(
                            'eigenvalues',
                            r'Eigenvalues of the Hessian Matrix :\s+([\d\-\s\.]+)',
                            dtype=np.dtype(np.float64),
                        ),
                        Quantity(
                            'frequencies',
                            r'Vibrational Frequencies \(cm\-1\) :([\d\-\s\.]+)',
                            dtype=np.dtype(np.float64),
                        ),
                    ]
                ),
            ),
            Quantity(
                'minimize',
                r'(.+?Optimization :[\s\S]+?Final Gradient Norm.+)',
                sub_parser=TextParser(
                    quantities=[
                        iteration_quantity,
                        Quantity(
                            'method', r' +(.+) Optimization', flatten=False, dtype=str
                        ),
                        Quantity(
                            'x_tiner_final_function_value',
                            rf'Final Function Value : +({re_f})',
                            dtype=np.float64,
                        ),
                        Quantity(
                            'x_tinker_final_rms_gradient',
                            rf'Final RMS Gradient : +({re_f})',
                            dtype=np.float64,
                        ),
                        Quantity(
                            'x_tinker_final_gradient_norm',
                            rf'Final Gradient Norm : +({re_f})',
                            dtype=np.float64,
                        ),
                    ]
                ),
            ),
            Quantity(
                'dynamic',
                r'(Molecular Dynamics[\s\S]+?)(?:\#\#\#\#\#|\Z)',
                sub_parser=TextParser(
                    quantities=[
                        iteration_quantity,
                        Quantity(
                            'instantaneous_values',
                            r'(Instantaneous Values for Frame Saved at[\s\S]+?Coordinate File.+)',
                            repeats=True,
                            sub_parser=TextParser(
                                quantities=[
                                    Quantity(
                                        'step', r'(\d+) Dynamics Steps', dtype=np.int32
                                    ),
                                    Quantity(
                                        'time',
                                        rf'Current Time +({re_f}) Picosecond',
                                        dtype=np.float64,
                                        unit=ureg.ps,
                                    ),
                                    Quantity(
                                        'potential',
                                        rf'Current Potential +({re_f}) Kcal/mole',
                                        dtype=np.float64,
                                        unit=ureg.J * 4184.0 / mol,
                                    ),
                                    Quantity(
                                        'kinetic',
                                        rf'Current Kinetic +({re_f}) Kcal/mole',
                                        dtype=np.float64,
                                        unit=ureg.J * 4184.0 / mol,
                                    ),
                                    Quantity(
                                        'lattice_lengths',
                                        rf'Lattice Lengths +({re_f} +{re_f} +{re_f})',
                                        dtype=np.dtype(np.float64),
                                    ),
                                    Quantity(
                                        'lattice_angles',
                                        rf'Lattice Angles +({re_f} +{re_f} +{re_f})',
                                        dtype=np.dtype(np.float64),
                                    ),
                                    Quantity(
                                        'frame', r'Frame Number +(\d+)', dtype=np.int32
                                    ),
                                    Quantity(
                                        'coordinate_file',
                                        r'Coordinate File +(\S+)',
                                        dtype=str,
                                    ),
                                ]
                            ),
                        ),
                        Quantity(
                            'average_values',
                            r'(Average Values for the Last[\s\S]+?Density.+)',
                            repeats=True,
                            sub_parser=TextParser(
                                quantities=[
                                    Quantity(
                                        'step', r'(\d+) Dynamics Steps', dtype=np.int32
                                    ),
                                    Quantity(
                                        'time',
                                        rf'Simulation Time +({re_f}) Picosecond',
                                        dtype=np.float64,
                                        unit=ureg.ps,
                                    ),
                                    Quantity(
                                        'energy_total',
                                        rf'Total Energy +({re_f}) Kcal/mole',
                                        dtype=np.float64,
                                        unit=ureg.J * 4184.0 / mol,
                                    ),
                                    Quantity(
                                        'potential',
                                        rf'Potential Energy +({re_f}) Kcal/mole',
                                        dtype=np.float64,
                                        unit=ureg.J * 4184.0 / mol,
                                    ),
                                    Quantity(
                                        'kinetic',
                                        rf'Kinetic Energy +({re_f}) Kcal/mole',
                                        dtype=np.float64,
                                        unit=ureg.J * 4184.0 / mol,
                                    ),
                                    Quantity(
                                        'temperature',
                                        rf'Temperature +({re_f}) Kelvin',
                                        dtype=np.float64,
                                        unit=ureg.kelvin,
                                    ),
                                    Quantity(
                                        'pressure',
                                        rf'Pressure +({re_f}) Atmosphere',
                                        dtype=np.float64,
                                        unit=ureg.atm,
                                    ),
                                    Quantity(
                                        'density',
                                        rf'Density +({re_f}) Grams/cc',
                                        dtype=np.float64,
                                        unit=ureg.g / ureg.cm**3,
                                    ),
                                ]
                            ),
                        ),
                    ]
                ),
            ),
        ]

        self._quantities = [
            Quantity(
                'run',
                r'Software Tools for Molecular Design([\s\S]+?)(?:TINKER  \-\-\-|\Z)',
                repeats=True,
                sub_parser=TextParser(quantities=calculation_quantities),
            )
        ]


class TinkerParser(MDParser):
    def __init__(self):
        self.out_parser = OutParser()
        self.traj_parser = MDAnalysisParser()
        self.key_parser = KeyParser()
        self.run_parser = RunParser()
        self._run_types = {
            'vibrate': 'single_point',
            'minimize': 'geometry_optimization',
            'dynamic': 'molecular_dynamics',
        }
        super().__init__()

    def _get_tinker_file(self, ext):
        if ext not in self._files:
            return
        current = self._files[ext]['current']
        # advance to the next file
        files = self._files[ext]['files']
        try:
            self._files[ext]['current'] = files[files.index(current) + 1]
        except Exception:
            # the last file has been reached in this case
            pass
        return os.path.join(self.maindir, current)

    def parse_system(self, index, filename):
        self.traj_parser.mainfile = filename
        self.traj_parser.options = dict(
            topology_format='ARC' if filename.endswith('.arc') else 'TXYZ'
        )

        if self.traj_parser.universe is None:
            return

        sec_system = System()
        self.archive.run[-1].system.append(sec_system)
        trajectory = self.traj_parser.universe.trajectory[index]
        sec_system.atoms = Atoms(
            positions=trajectory.positions * ureg.angstrom,
            labels=[atom.name for atom in list(self.traj_parser.universe.atoms)],
        )
        if trajectory.triclinic_dimensions is not None:
            sec_system.atoms.lattice_vectors = (
                trajectory.triclinic_dimensions * ureg.angstrom
            )
            sec_system.atoms.periodic = [True, True, True]
        if trajectory.has_velocities:
            sec_system.atoms.velocities = trajectory.velocities * (
                ureg.angstrom / ureg.ps
            )

        return sec_system

    def parse_method(self):
        sec_method = Method()
        self.archive.run[-1].method.append(sec_method)

        parameters = self.archive.run[-1].x_tinker_control_parameters
        if parameters.get('parameters') is not None:
            sec_method.force_field = ForceField(
                model=[Model(name=parameters['parameters'])]
            )
            # TODO read the prm file

        property_map = {
            'name': 'label',
            'type': 'x_tinker_atom_type',
            'resid': 'x_tinker_atom_resid',
        }
        if self.traj_parser.universe is not None:
            for atom in list(self.traj_parser.universe.atoms):
                sec_atom = AtomParameters()
                sec_method.atom_parameters.append(sec_atom)
                for key in ['charge', 'mass', 'name', 'type', 'resid']:
                    if hasattr(atom, key):
                        setattr(
                            sec_atom, property_map.get(key, key), getattr(atom, key)
                        )

        # TODO add interaction parameters

    def resolve_workflow_type(self, run):
        for program in run.keys():
            if program in self._run_types and run.get(program) is not None:
                return self._run_types[program]

    def parse_workflow(self, program, run):
        def resolve_ensemble_type():
            parameters = self.archive.run[-1].x_tinker_control_parameters
            if parameters is None:
                return

            thermostat, barostat = (
                parameters.get('thermostat', ''),
                parameters.get('barostat', ''),
            )
            thermostats = ['berendsen', 'andersen', 'bussi']
            # TODO verify this
            if barostat.lower() in ['berendsen', 'montecarlo']:
                ensemble_type = 'NPT'
            elif not barostat and thermostat in thermostats:
                ensemble_type = 'NVT'
            elif not barostat and not thermostat:
                ensemble_type = 'NVE'
            else:
                ensemble_type = None

            return ensemble_type

        # TODO handle multiple workflow sections
        parameters = list(program.get('parameters', []))
        # so we cover the case when optional parameters are missing
        parameters.extend([None] * 6)
        workflow_type = self.resolve_workflow_type(run)
        if workflow_type == 'molecular_dynamics':
            control_parameters = self.archive.run[-1].x_tinker_control_parameters
            # TODO verify this! I am sure it is wrong but tinker documentation does not specify clearly
            ensemble_types = ['NVE', 'NVT', 'NPT', None, None]
            ensemble = (
                ensemble_types[int(parameters[3]) - 1]
                if parameters[3] is not None
                else resolve_ensemble_type()
            )
            self.parse_md_workflow(
                dict(
                    method=dict(
                        thermodynamic_ensemble=ensemble,
                        integration_timestep=parameters[1] * ureg.fs
                        if parameters[1]
                        else parameters[1],
                    ),
                    x_tinker_barostat_tau=control_parameters.get('tau-pressure'),
                    x_tinker_barostat_type=control_parameters.get('barostat'),
                    x_tinker_integrator_type=control_parameters.get('integrator'),
                    x_tinker_number_of_steps_requested=parameters[0],
                    x_tinker_integrator_dt=parameters[1] * ureg.ps
                    if parameters[1]
                    else parameters,
                    x_tinker_thermostat_target_temperature=parameters[4] * ureg.kelvin
                    if parameters[4]
                    else parameters[4],
                    x_tinker_barostat_target_pressure=parameters[5] * ureg.atmosphere
                    if parameters[5]
                    else parameters[5],
                    x_tinker_thermostat_tau=control_parameters.get('tau-temperature'),
                    x_tinker_thermostat_type=control_parameters.get('thermostat'),
                )
            )

        elif workflow_type == 'geometry_optimization':
            workflow = GeometryOptimization(method=GeometryOptimizationMethod())
            workflow.method.method = run.minimize.get('method')
            workflow.x_tinker_convergence_tolerance_rms_gradient = parameters[0]
            for key, val in run.minimize.items():
                if key.startswith('x_tinker'):
                    setattr(workflow, key, val)

            self.archive.workflow2 = workflow

    def write_to_archive(self) -> None:
        self.out_parser.mainfile = self.mainfile
        self.out_parser.logger = self.logger
        self.maindir = os.path.dirname(self.mainfile)
        self._base_name = os.path.basename(self.mainfile).rsplit('.', 1)[0]
        files = os.listdir(self.maindir)
        # required to track the current files
        self._files = dict()
        for ext in ['xyz', 'arc', 'key']:
            matches = fnmatch.filter(files, f'{self._base_name}.{ext}*')
            if not matches:
                matches = fnmatch.filter(files, f'tinker.{ext}*')
            matches.sort(
                key=lambda x: int(x.rsplit('_', 1)[-1]) if x[-1].isdecimal() else 0
            )
            self._files[ext] = dict(
                files=matches, current=matches[0] if matches else ''
            )

        self.maindir = os.path.dirname(self.mainfile)

        workflows: List[str] = []

        def get_reference_filename(program):
            # resolves the filename as provided in the cli command for the program
            filename = program.get('name', '')
            if '.xyz' not in filename:
                filename = self._files['xyz']['current']
            # succeding files start from the index of the file for geometry_opt and md
            filename = filename if '.xyz' in filename else f'{filename}.xyz'
            if workflows[-1] in ['geometry_optimization', 'molecular_dynamics']:
                files = self._files['xyz']['files']
                try:
                    self._files['xyz']['current'] = files[files.index(filename) + 1]
                except Exception:
                    pass
            return os.path.join(self.maindir, filename)

        # necesarry to extract program parameters from the cli command in basename.run
        self.run_parser.mainfile = os.path.join(self.maindir, f'{self._base_name}.run')

        # TODO put runs in separate archives
        for run in self.out_parser.get('run', []):
            sec_run = Run()
            self.archive.run.append(sec_run)
            sec_run.program = Program(name='tinker', version=run.get('program_version'))

            workflow_type = self.resolve_workflow_type(run)
            workflows.append(workflow_type)
            # get parameters of the program from the cli command, the key file and the
            # initial structure file can also be specied as an argument so we need to
            # extract the information here
            # program can be executed for an arbitrary number of times so we need to
            # resolve the index of the appropriate command
            n_workflow = (
                len([workflow for workflow in workflows if workflow == workflow_type])
                - 1
            )
            program = self.run_parser.get(workflow_type, [])
            program = program[n_workflow] if len(program) > n_workflow else dict()
            if run.vibrate is not None:
                # reference structure
                sec_system = self.parse_system(0, get_reference_filename(program))
                sec_scc = Calculation()
                sec_run.calculation.append(sec_scc)
                sec_vibrations = VibrationalFrequencies()
                sec_scc.vibrational_frequencies.append(sec_vibrations)
                sec_vibrations.value = [
                    run.vibrate.frequencies[n]
                    for n in range(len(run.vibrate.get('frequencies', [])))
                    if n % 2 == 1
                ] * (1 / ureg.cm)
                sec_vibrations.x_tinker_eigenvalues = [
                    run.vibrate.eigenvalues[n]
                    for n in range(len(run.vibrate.get('eigenvalues', [])))
                    if n % 2 == 1
                ]
                sec_scc.system_ref = sec_system

            if run.minimize is not None:
                # initial structure
                initial_system = self.parse_system(0, get_reference_filename(program))

                # optimized structure
                sec_system = self.parse_system(0, self._get_tinker_file('xyz'))

                for n, step in enumerate(
                    run.minimize.get('iteration', {}).get('step', [])
                ):
                    sec_scc = Calculation()
                    sec_run.calculation.append(sec_scc)
                    sec_scc.energy = Energy(
                        total=EnergyEntry(
                            value=step[0]
                            * len(sec_system.atoms.positions)
                            * ureg.J
                            * 4184.0
                            / mol
                        )
                    )
                    if n == 0:
                        sec_scc.system_ref = initial_system
                # only the optimized structure is printed, corresponding to the last scc
                sec_scc.system_ref = sec_system

            if run.dynamic is not None:
                self.traj_parser.mainfile = self._get_tinker_file('arc')
                self.traj_parser.options = dict(topology_format='ARC')
                average_values = {
                    value.step: value for value in run.dynamic.get('average_values', [])
                }
                # set up md parser
                # TODO this is not efficient as we need to load traj file to know if it exists
                traj_steps, thermo_steps, n_atoms, filenames = [], [], [], []
                for nframe, value in enumerate(
                    run.dynamic.get('instantaneous_values', [])
                ):
                    filename = os.path.join(
                        self.maindir, value.get('coordinate_file', '')
                    )
                    index = nframe if filename.endswith('.arc') else 0
                    filenames.append((filename, index))
                    thermo_steps.append(value.step)
                    self.traj_parser.mainfile = filename
                    self.traj_parser.options = dict(
                        topology_format='ARC' if filename.endswith('.arc') else 'TXYZ'
                    )
                    trajectory = self.traj_parser.universe.trajectory[index]
                    if trajectory is not None:
                        traj_steps.append(value.step)
                        n_atoms.append(len(trajectory))
                self.n_atoms = max(n_atoms)
                self.trajectory_steps = traj_steps
                self.thermodynamics_steps = thermo_steps

                for n_frame, step in enumerate(self.thermodynamics_steps):
                    self.traj_parser.mainfile = filenames[n_frame][0]
                    trajectory = self.traj_parser.universe.trajectory[
                        filenames[n_frame][1]
                    ]
                    if step in self.trajectory_steps:
                        # sampled trajectory
                        positions = trajectory.positions * ureg.angstrom
                        labels = [
                            atom.name for atom in list(self.traj_parser.universe.atoms)
                        ]
                        lattice_vectors, periodic, velocities = None, None, None
                        if trajectory.triclinic_dimensions is not None:
                            lattice_vectors = (
                                trajectory.triclinic_dimensions * ureg.angstrom
                            )
                            periodic = [True, True, True]
                        if trajectory.has_velocities:
                            velocities = trajectory.velocities * (
                                ureg.angstrom / ureg.ps
                            )
                        self.parse_trajectory_step(
                            dict(
                                atoms=dict(
                                    positions=positions,
                                    labels=labels,
                                    lattice_vectors=lattice_vectors,
                                    periodic=periodic,
                                    velocities=velocities,
                                )
                            )
                        )
                    # thermo properties
                    instantaneous = run.dynamic.instantaneous_values[n_frame]
                    n_atoms_step = n_atoms[traj_steps.index(step)]
                    energy = dict(
                        total=dict(
                            value=(instantaneous.potential + instantaneous.kinetic)
                            * n_atoms_step,
                            kinetic=instantaneous.kinetic * n_atoms_step,
                            potential=instantaneous.potential * n_atoms_step,
                        )
                    )
                    data = dict(energy=energy, step=instantaneous.step)
                    average = average_values.get(instantaneous.step)
                    if average:
                        data['temperature'] = average.temperature
                        data['pressure'] = average.pressure
                    if trajectory is not None and trajectory.has_forces:
                        data['forces'] = dict(
                            total=dict(
                                value=trajectory.forces * (ureg.kJ / ureg.angstrom)
                            )
                        )
                    self.parse_thermodynamics_step(data)

            # TODO add support for other tinker programs

            # control parameters
            # a key file can be specified optionally via the cli we assign this instead to get the parameters
            self.key_parser.mainfile = os.path.join(
                self.maindir, program.get('key', '')
            )
            if self.key_parser.mainfile is None:
                # if the key file is not specified in cli, read from either basename.key
                # or tinker.key
                self.key_parser.mainfile = self._get_tinker_file('key')

            parameters = {
                key.lower(): val for key, val in self.key_parser.get('key_val', [])
            }
            sec_run.x_tinker_control_parameters = parameters
            # TODO should this be removed and only have a dictionary of control parameters
            sec_control = x_tinker_section_control_parameters()
            sec_run.x_tinker_section_control_parameters.append(sec_control)
            for key, val in parameters.items():
                key = key.replace('-', '_')
                setattr(
                    sec_control,
                    f'x_tinker_inout_control_{key}',
                    val if isinstance(val, bool) else str(val),
                )

            self.parse_method()

            self.parse_workflow(program, run)

            self.traj_parser.close()
