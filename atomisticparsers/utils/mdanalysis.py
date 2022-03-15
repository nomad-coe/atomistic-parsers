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

try:
    import MDAnalysis
    from MDAnalysis.topology.guessers import guess_atom_element
except Exception:
    MDAnalysis = None

from nomad.units import ureg
from nomad.parsing.file_parser import FileParser

MOL = 6.022140857e+23


class MDAnalysisParser(FileParser):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._args = args
        self._kwargs = kwargs

    @property
    def auxilliary_files(self):
        return self._args

    @auxilliary_files.setter
    def auxilliary_files(self, value):
        self._file_handler = None
        self._args = [value] if isinstance(value, str) else value

    @property
    def options(self):
        return self._kwargs

    @options.setter
    def options(self, value):
        self._file_handler = None
        self._kwargs = value

    @property
    def universe(self):
        if self._file_handler is None:
            try:
                self._file_handler = MDAnalysis.Universe(self.mainfile, *self.auxilliary_files, **self.options)
            except Exception as e:
                self.logger.error('Error creating MDAnalysis universe.', exc_info=e)
        return self._file_handler

    def parse(self, quantity_key: str = None, **kwargs):
        if self._results is None:
            self._results = dict()

        atoms = list(self.universe.atoms)

        name_map = {'mass': 'masses'}
        unit_map = {'mass': ureg.amu, 'charge': ureg.elementary_charge}
        self._results['atom_info'] = dict()
        for key in ['name', 'charge', 'mass', 'resid', 'resname', 'molnum', 'moltype', 'type']:
            try:
                value = [getattr(atom, key) for atom in atoms]
            except Exception:
                continue
            value = value * unit_map.get(key, 1) if value is not None else value
            self._results['atom_info'][name_map.get(key, f'{key}s')] = value

        # if atom name is not identified, set it to 'X'
        if self._results['atom_info'].get('names') is None:
            self._results['atom_info']['names'] = ['X'] * self.universe.atoms.n_atoms
        self._results['n_atoms'] = self.universe.atoms.n_atoms
        self._results['n_frames'] = len(self.universe.trajectory)
        # self._results['atom_labels'] = [
        #     guess_atom_element(name) for name in self._results['atom_info'].get('names', [])]

    @property
    def with_trajectory(self):
        '''
        True if trajectory is present.
        '''
        return self.universe.trajectory is not None and len(self.universe.trajectory) > 0

    def get_frame(self, frame_index):
        '''
        Returns the frame in the trajectory with index frame_index.
        '''
        try:
            return self.universe.trajectory[frame_index]
        except Exception as e:
            self.logger.warning('Error accessing frame.', exc_info=e)
            raise e

    def get_n_atoms(self, frame_index):
        '''
        Returns the number of atoms of the frame with index frame_index.
        '''
        frame = self.get_frame(frame_index)
        return len(frame) if frame is not None else None

    def get_atom_labels(self, frame_index):
        '''
        Returns the number of atoms of the frame with index frame_index.
        '''
        # MDAnalysis assumes no change in atom configuration
        return [guess_atom_element(name) for name in self.get('atom_info', {}).get('names', [])]

    def get_time_step(self, frame_index):
        '''
        Returns the integer time step of the frame with index frame_index.
        '''
        frame = self.get_frame(frame_index)
        return int(frame.time / frame.dt) if frame is not None else None

    def get_lattice_vectors(self, frame_index):
        '''
        Returns the lattice vectors of the frame with index frame_index.
        '''
        lattice_vectors = self.get_frame(frame_index).triclinic_dimensions
        return lattice_vectors * ureg.angstrom if lattice_vectors is not None else None

    def get_pbc(self, frame_index):
        '''
        Returns the lattice periodicity of the frame with index frame_index.
        '''
        lattice_vectors = self.get_lattice_vectors(frame_index)
        return [True] * 3 if lattice_vectors is not None else [False] * 3

    def get_positions(self, frame_index):
        '''
        Returns the positions of the atoms of the frame with index frame_index.
        '''
        frame = self.get_frame(frame_index)
        return frame.positions * ureg.angstrom if frame.has_positions else None

    def get_velocities(self, frame_index):
        '''
        Returns the velocities of the atoms of the frame with index frame_index.
        '''
        frame = self.get_frame(frame_index)
        return frame.velocities * ureg.angstrom / ureg.ps if frame.has_velocities else None

    def get_forces(self, frame_index):
        '''
        Returns the forces on the atoms of the frame with index frame_index.
        '''
        frame = self.get_frame(frame_index)
        return frame.forces * ureg.kJ / (MOL * ureg.angstrom) if frame.has_forces else None

    def get_interactions(self):
        interactions = self.get('interactions', None)

        if interactions is not None:
            return interactions

        interaction_types = ['angles', 'bonds', 'dihedrals', 'impropers']
        interactions = []
        for interaction_type in interaction_types:
            try:
                interaction = getattr(self.universe, interaction_type)
            except Exception:
                continue

            for i in range(len(interaction)):
                interactions.append(
                    dict(
                        atom_labels=list(interaction[i].type), parameters=float(interaction[i].value()),
                        atom_indices=interaction[i].indices, type=interaction[i].btype
                    )
                )

        self._results['interactions'] = interactions

        return interactions
