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
from ase.io.trajectory import Trajectory

from nomad.units import ureg
from nomad.parsing.file_parser import FileParser
from nomad.datamodel.metainfo.simulation.run import Run, Program
from nomad.datamodel.metainfo.simulation.method import (
    Method, ForceField, Model)
from nomad.datamodel.metainfo.simulation.system import (
    System, Atoms, Constraint)
from nomad.datamodel.metainfo.simulation.calculation import (
    Calculation, Energy, EnergyEntry, Forces, ForcesEntry)
from nomad.datamodel.metainfo.workflow import (
    Workflow, GeometryOptimization, MolecularDynamics)
from nomad.datamodel.metainfo.simulation.workflow import (
    GeometryOptimization as GeometryOptimization2, GeometryOptimizationMethod,
    MolecularDynamics as MolecularDynamics2, MolecularDynamicsMethod
)


class TrajParser(FileParser):
    def __init__(self):
        super().__init__()

    @property
    def traj(self):
        if self._file_handler is None:
            self._file_handler = Trajectory(self.mainfile, 'r')
            # check if traj file is really asap
            if 'calculator' in self._file_handler.backend.keys():
                if self._file_handler.backend.calculator.name != 'emt':  # pylint: disable=E1101
                    self.logger.error('Trajectory is not ASAP.')
                    self._file_handler = None
        return self._file_handler

    def get_version(self):
        if hasattr(self.traj, 'ase_version') and self.traj.ase_version:
            return self.traj.ase_version
        else:
            return '3.x.x'


class AsapParser:
    def __init__(self):
        self.traj_parser = TrajParser()

    def init_parser(self):
        self.traj_parser.mainfile = self.filepath
        self.traj_parser.logger = self.logger

    def parse_system(self, traj):
        sec_system = self.archive.run[0].m_create(System)

        sec_atoms = sec_system.m_create(Atoms)
        sec_atoms.lattice_vectors = traj.get_cell() * ureg.angstrom
        sec_atoms.labels = traj.get_chemical_symbols()
        sec_atoms.positions = traj.get_positions() * ureg.angstrom
        sec_atoms.periodic = traj.get_pbc()
        if traj.get_velocities() is not None:
            sec_atoms.velocities = traj.get_velocities() * (ureg.angstrom / ureg.fs)

        def get_constraint_name(constraint):
            def index():
                d = constraint['kwargs'].get('direction')
                return ((d / np.linalg.norm(d)) ** 2).argsort()[2]

            name = constraint.get('name')
            if name == 'FixedPlane':
                return ['fix_yz', 'fix_xz', 'fix_xy'][index()]
            elif name == 'FixedLine':
                return ['fix_x', 'fix_y', 'fix_z'][index()]
            elif name == 'FixAtoms':
                return 'fix_xyz'
            else:
                return name

        for constraint in traj.constraints:
            sec_constraint = sec_system.m_create(Constraint)
            as_dict = constraint.todict()
            indices = as_dict['kwargs'].get('a', as_dict['kwargs'].get('indices'))
            sec_constraint.indices = np.asarray(indices)
            sec_constraint.kind = get_constraint_name(as_dict)

    def parse_scc(self, traj):
        sec_scc = self.archive.run[0].m_create(Calculation)

        try:
            sec_energy = sec_scc.m_create(Energy)
            sec_energy.total = EnergyEntry(value=traj.get_total_energy() * ureg.eV)
        except Exception:
            pass

        try:
            sec_forces = sec_scc.m_create(Forces)
            sec_forces.total = ForcesEntry(
                value=traj.get_forces() * ureg.eV / ureg.angstrom,
                value_raw=traj.get_forces(apply_constraint=False) * ureg.eV / ureg.angstrom)
        except Exception:
            pass

    def parse_method(self):
        traj = self.traj_parser.traj
        sec_method = self.archive.run[0].m_create(Method)

        if traj[0].calc is not None:
            sec_method.force_field = ForceField(model=[Model(name=traj[0].calc.name)])

        description = traj.description if hasattr(traj, 'description') else dict()
        if not description:
            return

        sec_workflow = self.archive.m_create(Workflow)

        workflow = None
        calc_type = description.get('type')
        if calc_type == 'optimization':
            sec_workflow.type = 'geometry_optimization'
            sec_geometry_opt = sec_workflow.m_create(GeometryOptimization)
            sec_geometry_opt.method = description.get('optimizer', '').lower()
            sec_geometry_opt.x_asap_maxstep = description.get('maxstep', 0)
            workflow = GeometryOptimization2(method=GeometryOptimizationMethod())
            workflow.method.method = description.get('optimizer', '').lower()
        elif calc_type == 'molecular-dynamics':
            sec_workflow.type = 'molecular_dynamics'
            sec_md = sec_workflow.m_create(MolecularDynamics)
            sec_md.x_asap_timestep = description.get('timestep', 0)
            sec_md.x_asap_temperature = description.get('temperature', 0)
            workflow = MolecularDynamics2(method=MolecularDynamicsMethod())
            md_type = description.get('md-type', '')
            if 'Langevin' in md_type:
                sec_md.ensemble_type = 'NVT'
                sec_md.x_asap_langevin_friction = description.get('friction', 0)
                workflow.method.thermodynamic_ensemble = 'NVT'
            elif 'NVT' in md_type:
                sec_md.ensemble_type = 'NVT'
                workflow.method.thermodynamic_ensemble = 'NVT'
            elif 'Verlet' in md_type:
                sec_md.ensemble_type = 'NVE'
                workflow.method.thermodynamic_ensemble = 'NVE'
            elif 'NPT' in md_type:
                sec_md.ensemble_type = 'NPT'
                workflow.method.thermodynamic_ensemble = 'NPT'
        self.archive.workflow2 = workflow

    def parse(self, filepath, archive, logger):
        self.filepath = os.path.abspath(filepath)
        self.archive = archive
        self.maindir = os.path.dirname(self.filepath)
        self.logger = logger if logger is not None else logging

        self.init_parser()

        if self.traj_parser.traj is None:
            return

        sec_run = self.archive.m_create(Run)
        sec_run. program = Program(name='ASAP', version=self.traj_parser.get_version())

        # TODO do we build the topology and method for each frame
        self.parse_method()
        for traj in self.traj_parser.traj:
            self.parse_system(traj)
            self.parse_scc(traj)
            # add references to scc
            sec_scc = sec_run.calculation[-1]
            sec_scc.method_ref = sec_run.method[-1]
            sec_scc.system_ref = sec_run.system[-1]
