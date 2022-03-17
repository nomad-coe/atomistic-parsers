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

import numpy as np
try:
    import MDAnalysis
    import MDAnalysis.analysis.rdf as MDA_RDF
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
        self._atomsgroup_info = None

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
    def atomsgroup_info(self):
        if self._atomsgroup_info is None:
            atomsgroup_info = dict()
            atomsgroup_info['moltypes'] = self.get_moltypes()
            atomsgroup_info['molnums'] = self.get_molnums()
            atomsgroup_info['segids'] = getattr(self.universe.atoms, 'segids', None)
            atomsgroup_info['segindices'] = getattr(self.universe.atoms, 'segindices', None)
            atomsgroup_info['resnames'] = self.get_resnames()
            atomsgroup_info['resids'] = getattr(self.universe.atoms, 'resids', None)
            atomsgroup_info['elements'] = self.get_elements()
            atomsgroup_info['names'] = self.get_names()
            self._atomsgroup_info = atomsgroup_info
        return self._atomsgroup_info

    def get_moltypes(self):
        if hasattr(self.universe.atoms, 'moltypes'):
            return self.universe.atoms.moltypes
        elif hasattr(self.universe.atoms, 'fragments'):
            atoms_fragtypes = self.get_fragtypes()
            return atoms_fragtypes
        else:
            return

    def get_molnums(self):
        if hasattr(self.universe.atoms, 'molnums'):
            return self.universe.atoms.molnums
        elif hasattr(self.universe.atoms, 'fragindices'):
            return self.universe.atoms.fragindices
        else:
            return

    def get_resnames(self):
        if hasattr(self.universe.atoms, 'resnames'):
            return self.universe.atoms.resnames
        elif hasattr(self.universe.atoms, 'resids'):
            return self.universe.atoms.resids.astype(str)
        else:
            return

    def get_elements(self):
        if hasattr(self.universe.atoms, 'elements'):
            return self.universe.atoms.elements
        elif hasattr(self.universe.atoms, 'types'):
            return self.universe.atoms.types
        else:
            return

    def get_names(self):
        if hasattr(self.universe.atoms, 'names'):
            return self.universe.atoms.names
        elif hasattr(self.universe.atoms, 'types'):
            return self.universe.atoms.types
        else:
            return

    def get_fragtypes(self):
        atoms_fragtypes = np.empty(self.universe.atoms.types.shape, dtype=str)
        ctr_fragtype = 0
        atoms_fragtypes[self.universe.atoms.fragments[0]._ix] = ctr_fragtype
        frag_unique_atomtypes = []
        frag_unique_atomtypes.append(self.universe.atoms.types[self.universe.atoms.fragments[0]._ix])
        ctr_fragtype += 1
        for i_frag in range(1, self.universe.atoms.n_fragments):
            types_i_frag = self.universe.atoms.types[self.universe.atoms.fragments[i_frag]._ix]
            flag_fragtype_exists = False
            for j_frag in range(len(frag_unique_atomtypes) - 1, -1, -1):
                types_j_frag = frag_unique_atomtypes[j_frag]
                if len(types_i_frag) != len(types_j_frag):
                    continue
                elif np.all(types_i_frag == types_j_frag):
                    atoms_fragtypes[self.universe.atoms.fragments[i_frag]._ix] = j_frag
                    flag_fragtype_exists = True
            if not flag_fragtype_exists:
                atoms_fragtypes[self.universe.atoms.fragments[i_frag]._ix] = ctr_fragtype
                frag_unique_atomtypes.append(self.universe.atoms.types[self.universe.atoms.fragments[i_frag]._ix])
                ctr_fragtype += 1
        return atoms_fragtypes

    class BeadGroup(object):
        # see https://github.com/MDAnalysis/mdanalysis/issues/1891#issuecomment-387138110
        # by @richardjgowers with performance improvements
        def __init__(self, atoms, compound="fragments"):
            """Initialize with an AtomGroup instance.
            Will split based on keyword 'compounds' (residues or fragments).
            """
            self._atoms = atoms
            self.compound = compound
            self._nbeads = len(getattr(self._atoms, self.compound))
            # for caching
            self._cache = {}
            self._cache["positions"] = None
            self.__last_frame = None

        def __len__(self):
            return self._nbeads

        @property
        def positions(self):
            # cache positions for current frame
            if self.universe.trajectory.frame != self.__last_frame:
                self._cache["positions"] = self._atoms.center_of_mass(
                    unwrap=True, compound=self.compound)
                self.__last_frame = self.universe.trajectory.frame
            return self._cache["positions"]

        @property
        @MDAnalysis.lib.util.cached("universe")
        def universe(self):
            return self._atoms.universe

    class replace_with_COM:
        """Replace special atom index `atom_indices` in each fragment with COM of the fragment."""
        def __init__(self, molecule, selection_atom_indices):
            self.molecule = molecule
            self.com_atoms = selection_atom_indices

            # sanity check
            assert self.get_com().shape == self.com_atoms.positions.shape

        def get_com(self):
            return self.molecule.center_of_mass(unwrap=True, compound="fragments")

        def __call__(self, ts):
            self.com_atoms.positions = self.get_com()
            return ts

    def _calc_molecular_RDF(self):
        moltypes = np.unique(self.atomsgroup_info['moltypes'])
        BeadGroups = {}
        for moltype in moltypes:
            if hasattr(self.universe.atoms, 'moltypes'):
                AGs_by_moltype = self.universe.select_atoms('moltype ' + moltype)
            else:  # this is easier than adding something to the universe
                selection = 'index ' + ''.join(str(i) + ' '
                             for i in np.where(self.atomsgroup_info['moltypes'] == moltype)[0])
                AGs_by_moltype = self.universe.select_atoms(selection)
            BeadGroups[moltype] = self.BeadGroup(AGs_by_moltype, compound="fragments")

        min_box_dimension = np.min(self.universe.trajectory[0].dimensions[:3])
        max_rdf_dist = min_box_dimension / 2
        n_bins = 150
        n_smooth = 6
        RDF_types = []
        RDF_variables_name = []
        RDF_bins = []
        RDF_values = []
        for i, moltype_i in enumerate(moltypes):
            for j, moltype_j in enumerate(moltypes):
                if j > i:
                    continue
                elif i == j and BeadGroups[moltype_i].positions.shape[0] == 1:  # skip if only 1 mol in group
                    continue

                if i == j:
                    exclusion_block = (1, 1)  # remove self-distance
                else:
                    exclusion_block = None
                pair_type = moltype_i + '-' + moltype_j
                RDF = MDA_RDF.InterRDF(BeadGroups[moltype_i], BeadGroups[moltype_j],
                                       range=(0, max_rdf_dist),
                                       exclusion_block=exclusion_block, nbins=n_bins).run()
                RDF_types.append(pair_type)
                RDF_variables_name.append(['distance'])
                RDF_bins.append([RDF.results.bins[int(n_smooth / 2):-int(n_smooth / 2)]])
                RDF_values.append(np.convolve(RDF.results.rdf,
                                              np.ones((n_smooth,)) / n_smooth,
                                              mode='same')[int(n_smooth / 2):-int(n_smooth / 2)])
        RDF_results = {}
        RDF_results['types'] = np.array(RDF_types)
        RDF_results['n_smooth'] = n_smooth
        RDF_results['variables_name'] = np.array(RDF_variables_name)
        RDF_results['bins'] = np.array(RDF_bins)
        RDF_results['values'] = np.array(RDF_values)
        return RDF_results

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
