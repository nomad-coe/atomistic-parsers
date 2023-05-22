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
import logging
# import gsd
# try:
#     import gsd.fl as gsdfl
# except Exception:
#     logging.warn('Required module gsd.fl not found.')
#     gsdfl = False
try:
    import gsd.hoomd as gsdhoomd
except Exception:
    logging.warn('Required module gsd.hoomd not found.')
    gsdhoomd = False

# from nomad.units import ureg
from nomad.parsing.file_parser import FileParser
from nomad.datamodel.metainfo.simulation.run import Run, Program  #, TimeRun
from nomad.datamodel.metainfo.simulation.method import (
    Method, ForceField, Model, Interaction, AtomParameters
)
from nomad.datamodel.metainfo.simulation.system import (
    System, Atoms, AtomsGroup
)
from nomad.datamodel.metainfo.simulation.calculation import (
    Calculation
)
from nomad.datamodel.metainfo.workflow import (
    Workflow, MolecularDynamics
)
from nomad.datamodel.metainfo.simulation import workflow as workflow2
# from .metainfo.hoomdblue import x_gromacs_section_control_parameters, x_gromacs_section_input_output_files
from nomad.atomutils import get_molecules_from_bond_list, is_same_molecule, get_composition

MOL = 6.022140857e+23


class HoomdblueGsdParser(FileParser):
    def __init__(self):
        super().__init__(None)
        self._nomad_to_hoomdblue_map = {}
        self._nomad_to_hoomdblue_map['system'] = {}
        self._nomad_to_hoomdblue_map['system']['atoms'] = {
            # 'lattice_vectors': 'configuration.box',
            'positions': 'particles.position',
            'x_hoomdblue_orientation': 'particles.orientation',
            # 'x_hoomdblue_typeid': 'particles.typeid',
            'velocities': 'particles.velocity',
            'x_hoomdblue_angmom': 'particles.angmom',
            'x_hoomdblue_image': 'particles.image',
        }

        self._nomad_to_hoomdblue_map['calculation'] = {
            'step': 'configuration.step',
        }

        self._nomad_to_hoomdblue_map['method'] = {}
        self._nomad_to_hoomdblue_map['method']['atom_parameters'] = {
            # 'x_hoomdblue_types': 'particles.types',
            'mass': 'particles.mass',
            'charge': 'particles.charge',
            'x_hoomdblue_diameter': 'particles.diameter',
            'x_hoomdblue_body': 'particles.body',
            'x_hoomdblue_moment_inertia': 'particles.moment_inertia',
            'x_hoomdblue_type_shapes': 'particles.type_shapes'
        }
        self._hoomdblue_interaction_keys = ['bonds', 'angles', 'dihedrals', 'impropers', 'constraints', 'pairs']

    def _attr_getter(self, source, path, default):
        '''
        Extracts attribute from object based on path, and returns default if not defined.
        '''
        section_segments = path.split('.')
        for section in section_segments:
            try:
                value = getattr(source, section)
                source = value[-1] if isinstance(value, list) else value
            except Exception:
                return
        source = source if source is not None else default
        return source

    @property
    def filegsd(self):
        if self._file_handler is None:
            try:
                self._file_handler = gsdhoomd.open(self.mainfile, 'rb')
            except Exception:
                self.logger.error('Error reading gsd file.')

        return self._file_handler


class HoomdblueParser:
    def __init__(self):
        self.gsd_parser = HoomdblueGsdParser()
        self._frame_rate = None
        # max cumulative number of atoms for all parsed trajectories to calculate sampling rate
        self._cum_max_atoms = 2500000

    @property
    def frame_rate(self):
        if self._frame_rate is None:
            if self.gsd_parser:
                n_frames = 0
                n_atoms = 0
                for frame in enumerate(self.gsd_parser.filegsd):
                    particles = getattr(frame, 'particles', None)
                    n_atoms = getattr(particles, 'N', 0)
                    n_atoms += n_atoms
                    n_frames += 1
                if n_atoms == 0 or n_frames == 0:
                    self._frame_rate = 1
                else:
                    cum_atoms = n_atoms * n_frames
                    self._frame_rate = 1 if cum_atoms <= self._cum_max_atoms else cum_atoms // self._cum_max_atoms
        return self._frame_rate

    def parse_calculation(self, frame):
        sec_run = self.archive.run[-1]
        sec_scc = sec_run.m_create(Calculation)

        calculation_map = self.gsd_parser._nomad_to_hoomdblue_map['calculation']
        for nomad_attr, hoomdblue_sec in calculation_map.items():
            hoomdblue_attr = self.gsd_parser._attr_getter(frame, hoomdblue_sec, None)
            if hoomdblue_attr:
                if nomad_attr == 'step':
                    hoomdblue_attr = int(hoomdblue_attr)
                setattr(sec_scc, nomad_attr, hoomdblue_attr)

    def parse_system(self, frame):
        sec_run = self.archive.run[-1]

        particles = getattr(frame, 'particles', None)
        n_atoms = getattr(particles, 'N', 0)
        if n_atoms == 0:
            step = self.gsd_parser._attr_getter(frame, 'configuration.step', None)
            self.logger.error('Error parsing gsd method: There seems to be no atoms in frame' + str(step))

        sec_system = sec_run.m_create(System)
        sec_atoms = sec_system.m_create(Atoms)

        # get lattice vectors
        hoomdblue_box = self.gsd_parser._attr_getter(frame, 'configuration.box', None)
        if hoomdblue_box is not None:
            lattice_vectors = [
                [hoomdblue_box[0], hoomdblue_box[3] * hoomdblue_box[4] * hoomdblue_box[1], hoomdblue_box[3] * hoomdblue_box[5] * hoomdblue_box[2]],
                [0, hoomdblue_box[1], hoomdblue_box[4] * hoomdblue_box[5] * hoomdblue_box[2]],
                [0, 0, hoomdblue_box[2]]]
            setattr(sec_atoms, 'lattice_vectors', lattice_vectors)

        atom_attributes_map = self.gsd_parser._nomad_to_hoomdblue_map['system']['atoms']
        for nomad_attr, hoomdblue_sec in atom_attributes_map.items():
            hoomdblue_attr = self.gsd_parser._attr_getter(frame, hoomdblue_sec, None)
            if hoomdblue_attr is not None:
                setattr(sec_atoms, nomad_attr, hoomdblue_attr)

        particle_types = getattr(particles, 'types', None)
        particles_typeid = getattr(particles, 'typeid', None)
        bond_list = self.gsd_parser._attr_getter(frame, 'bonds.group', None)
        if bond_list is not None:
            molecules = get_molecules_from_bond_list(n_atoms, bond_list, particle_types, particles_typeid)

            # create groups of molecules
            mol_groups = []
            mol_groups.append({})
            mol_groups[0]['molecules'] = []
            mol_groups[0]['molecules'].append(molecules[0])
            mol_groups[0]['formula'] = get_composition(molecules[0]['names'])
            for mol in molecules[1:]:
                flag_mol_group_exists = False
                for i_mol_group in range(len(mol_groups)):
                    if is_same_molecule(mol, mol_groups[i_mol_group]['molecules'][0]):
                        mol_groups[i_mol_group]['molecules'].append(mol)
                        flag_mol_group_exists = True
                        break
                if not flag_mol_group_exists:
                    mol_groups.append({})
                    mol_groups[-1]['molecules'] = []
                    mol_groups[-1]['molecules'].append(mol)
                    mol_groups[-1]['formula'] = get_composition(mol['names'])

            for i_mol_group, mol_group in enumerate(mol_groups):
                sec_mol_group = sec_system.m_create(AtomsGroup)
                sec_mol_group.type = 'molecule_group'
                sec_mol_group.index = i_mol_group
                sec_mol_group.atom_indices = np.concatenate([mol['indices'] for mol in mol_group['molecules']])
                sec_mol_group.n_atoms = len(sec_mol_group.atom_indices)
                sec_mol_group.is_molecule = False
                sec_mol_group.label = f'group_{i_mol_group}'
                n_mol = len(mol_group['molecules'])
                formula = f'{i_mol_group}({n_mol})'
                sec_mol_group.composition_formula = formula

                for i_molecule, molecule in enumerate(mol_group['molecules']):
                    sec_molecule = sec_mol_group.m_create(AtomsGroup)
                    sec_molecule.index = i_molecule
                    sec_molecule.atom_indices = molecule['indices']
                    sec_molecule.n_atoms = len(sec_molecule.atom_indices)
                    sec_molecule.label = str(i_mol_group)
                    sec_molecule.type = 'molecule'
                    sec_molecule.is_molecule = True
                    sec_molecule.composition_formula = get_composition(molecule['names'])

    def parse_method(self, frame):

        sec_method = self.archive.run[-1].m_create(Method)
        sec_force_field = sec_method.m_create(ForceField)
        sec_model = sec_force_field.m_create(Model)

        particles = getattr(frame, 'particles', None)
        n_atoms = getattr(particles, 'N', 0)
        if n_atoms == 0:
            step = self.gsd_parser._attr_getter(frame, 'configuration.step', None)
            self.logger.warning(rf'Error parsing gsd method: There seems to be no atoms in frame {step}')
            return

        particle_types = getattr(particles, 'types', None)
        particles_typeid = getattr(particles, 'typeid', None)
        if particle_types:
            particle_labels = [particle_types[particles_typeid[i_atom]] for i_atom in range(n_atoms)] if particles_typeid.any() else None

        atom_parameters_map = self.gsd_parser._nomad_to_hoomdblue_map['method']['atom_parameters']
        for n in range(n_atoms):
            sec_atom = sec_method.m_create(AtomParameters)
            for nomad_attr, hoomdblue_sec in atom_parameters_map.items():
                hoomdblue_attr = self.gsd_parser._attr_getter(frame, hoomdblue_sec, [None] * (n + 1))
                hoomdblue_attr = hoomdblue_attr[n] if len(hoomdblue_attr) > n else None
                if hoomdblue_attr is not None:
                    setattr(sec_atom, nomad_attr, hoomdblue_attr)
            sec_atom.label = particle_labels[n]

        # Get the interactions
        atom_types = np.array(getattr(particles, 'types', []))
        atom_typeid = np.array(getattr(particles, 'typeid', []))
        atom_labels = np.array(atom_types)[atom_typeid]
        for interaction_key in self.gsd_parser._hoomdblue_interaction_keys:
            hoomdblue_sec = self.gsd_parser._attr_getter(frame, interaction_key, None)
            if not hoomdblue_sec:
                continue
            sec_interaction = sec_model.m_create(Interaction)
            sec_interaction.type = interaction_key
            sec_interaction.n_inter = getattr(hoomdblue_sec, 'N', None)
            sec_interaction.n_atoms = getattr(hoomdblue_sec, 'M', None)
            inter_type_names = getattr(hoomdblue_sec, 'types', None)
            inter_atom_indices = getattr(hoomdblue_sec, 'group', None)
            typeid = getattr(hoomdblue_sec, 'typeid', [])
            inter_types = np.unique(typeid)
            for inter in inter_types:
                inter_group = np.where(typeid == inter)[0]
                n_inter = len(inter_group)
                if n_inter < 1:
                    continue
                sec_inter_group = sec_interaction.m_create(Interaction)
                sec_inter_group.n_inter = n_inter
                sec_inter_group.n_atoms = sec_interaction.n_atoms
                sec_inter_group.name = inter_type_names[inter]
                sec_inter_group.atom_indices = inter_atom_indices[inter_group]
                sec_inter_group.atom_labels = atom_labels[sec_inter_group.atom_indices]

    def parse_workflow(self):

        sec_workflow = self.archive.m_create(Workflow)
        sec_workflow.type = 'molecular_dynamics'
        __ = sec_workflow.m_create(MolecularDynamics)
        workflow = workflow2.MolecularDynamics(
            method=workflow2.MolecularDynamicsMethod(
                thermostat_parameters=workflow2.ThermostatParameters(),
                barostat_parameters=workflow2.BarostatParameters()
            ), results=workflow2.MolecularDynamicsResults()
        )
        self.archive.workflow2 = workflow

    def init_parser(self):

        self.gsd_parser.mainfile = self.filepath

    def parse(self, filepath, archive, logger):

        self.filepath = os.path.abspath(filepath)
        self.archive = archive
        self.logger = logging.getLogger(__name__) if logger is None else logger
        self._maindir = os.path.dirname(self.filepath)
        self._hoomdblue_files = os.listdir(self._maindir)
        self._basename = os.path.basename(filepath).rsplit('.', 1)[0]

        self.init_parser()

        if self.gsd_parser is None:
            return

        sec_run = self.archive.m_create(Run)
        sec_run.program = Program(name='HOOMDBLUE', version='unknown')

        for i_frame, frame in enumerate(self.gsd_parser.filegsd):

            self.parse_method(frame)

            if (i_frame % self.frame_rate) == 0:
                self.parse_system(frame)

            self.parse_calculation(frame)

        self.parse_workflow()
