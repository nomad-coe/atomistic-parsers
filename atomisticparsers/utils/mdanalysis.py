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
from typing import Any, Dict
from nptyping import NDArray
from collections import namedtuple
from array import array
from scipy import sparse
from scipy.stats import linregress

from nomad.units import ureg
from nomad.parsing.file_parser import FileParser
from nomad.atomutils import (
    BeadGroup,
    shifted_correlation_average
)

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

    @property
    def bead_groups(self):
        atoms_moltypes = self.get('atoms_info', {}).get('moltypes', [])
        moltypes = np.unique(atoms_moltypes)
        bead_groups = {}
        compound = 'fragments'
        for moltype in moltypes:
            if hasattr(self.universe.atoms, 'moltypes'):
                ags_by_moltype = self.universe.select_atoms('moltype ' + moltype)
            else:  # this is easier than adding something to the universe
                selection = ' '.join([str(i) for i in np.where(atoms_moltypes == moltype)[0]])
                selection = f'index {selection}'
                ags_by_moltype = self.universe.select_atoms(selection)
            ags_by_moltype = ags_by_moltype[ags_by_moltype.masses > abs(1e-2)]  # remove any virtual/massless sites (needed for, e.g., 4-bead water models)
            bead_groups[moltype] = BeadGroup(ags_by_moltype, compound=compound)

        return bead_groups

    def parse(self, quantity_key: str = None, **kwargs):
        if self._results is None:
            self._results: Dict[str, Any] = dict()

        atoms = list(self.universe.atoms)

        name_map = {'mass': 'masses'}
        unit_map = {'mass': ureg.amu, 'charge': ureg.elementary_charge}
        self._results['atoms_info'] = dict()
        for key in ['name', 'charge', 'mass', 'resid', 'resname', 'molnum', 'moltype', 'type', 'segid', 'element']:
            try:
                value = [getattr(atom, key) for atom in atoms]
            except Exception:
                continue
            value = value * unit_map.get(key, 1) if value is not None else value
            self._results['atoms_info'][name_map.get(key, f'{key}s')] = value

        # if atom name is not identified, set it to 'X'
        if self._results['atoms_info'].get('names') is None:
            self._results['atoms_info']['names'] = ['X'] * self.universe.atoms.n_atoms
        self._results['n_atoms'] = self.universe.atoms.n_atoms
        self._results['n_frames'] = len(self.universe.trajectory)

        # make substitutions based on available atom info
        if self._results['atoms_info'].get('moltypes') is None:
            if hasattr(self.universe.atoms, 'fragments'):
                self._results['atoms_info']['moltypes'] = self.get_fragtypes()

        if self._results['atoms_info'].get('molnums') is None:
            try:
                value = getattr(self.universe.atoms, 'fragindices')
                self._results['atoms_info']['molnums'] = value
            except Exception:
                pass

        if self._results['atoms_info'].get('resnames') is None:
            try:
                self._results['atoms_info']['resnames'] = self._results['atoms_info']['resids']
            except Exception:
                pass

        if self._results['atoms_info'].get('names') is None:
            try:
                self._results['atoms_info']['names'] = self._results['atoms_info']['types']
            except Exception:
                pass

        if self._results['atoms_info'].get('elements') is None:
            try:
                self._results['atoms_info']['elements'] = self._results['atoms_info']['names']
            except Exception:
                pass

    def get_fragtypes(self):
        # TODO put description otherwise, make private or put under parse method
        '''
        '''
        atoms_fragtypes = np.empty(self.universe.atoms.types.shape, dtype=str)
        ctr_fragtype = 0
        atoms_fragtypes[self.universe.atoms.fragments[0].ix] = ctr_fragtype
        frag_unique_atomtypes = [self.universe.atoms.types[self.universe.atoms.fragments[0].ix]]
        ctr_fragtype += 1
        for i_frag in range(1, self.universe.atoms.n_fragments):
            types_i_frag = self.universe.atoms.types[self.universe.atoms.fragments[i_frag].ix]
            flag_fragtype_exists = False
            for j_frag in range(len(frag_unique_atomtypes) - 1, -1, -1):
                types_j_frag = frag_unique_atomtypes[j_frag]
                if len(types_i_frag) != len(types_j_frag):
                    continue
                elif np.all(types_i_frag == types_j_frag):
                    atoms_fragtypes[self.universe.atoms.fragments[i_frag].ix] = j_frag
                    flag_fragtype_exists = True
            if not flag_fragtype_exists:
                atoms_fragtypes[self.universe.atoms.fragments[i_frag].ix] = ctr_fragtype
                frag_unique_atomtypes.append(self.universe.atoms.types[self.universe.atoms.fragments[i_frag].ix])
                ctr_fragtype += 1
        return atoms_fragtypes

    def calc_molecular_rdf(self, n_traj_split=10, n_prune=1, interval_indices=None, n_bins=200, n_smooth=2):
        '''
        Calculates the radial distribution functions between for each unique pair of
        molecule types as a function of their center of mass distance.

        interval_indices: 2D array specifying the groups of the n_traj_split intervals to be averaged
        '''
        def get_rdf_avg(rdf_results_tmp, rdf_results, interval_indices, n_frames_split):
            split_weights = n_frames_split[np.array(interval_indices)] / np.sum(n_frames_split[np.array(interval_indices)])
            assert abs(np.sum(split_weights) - 1.0) < 1e-6
            rdf_values_avg = split_weights[0] * rdf_results_tmp['value'][interval_indices[0]]
            for i_interval, interval in enumerate(interval_indices[1:]):
                assert (rdf_results_tmp['types'][interval] == rdf_results_tmp['types'][interval - 1])
                assert (rdf_results_tmp['variables_name'][interval] == rdf_results_tmp['variables_name'][interval - 1])
                assert (rdf_results_tmp['bins'][interval] == rdf_results_tmp['bins'][interval - 1]).all()
                rdf_values_avg += split_weights[i_interval + 1] * rdf_results_tmp['value'][interval]
            rdf_results['types'].append(rdf_results_tmp['types'][interval_indices[0]])
            rdf_results['variables_name'].append(rdf_results_tmp['variables_name'][interval_indices[0]])
            rdf_results['bins'].append(rdf_results_tmp['bins'][interval_indices[0]])
            rdf_results['value'].append(rdf_values_avg)
            rdf_results['frame_start'].append(int(rdf_results_tmp['frame_start'][interval_indices[0]]))
            rdf_results['frame_end'].append(int(rdf_results_tmp['frame_end'][interval_indices[-1]]))

        if self.universe is None:
            return
        if self.universe.trajectory[0].dimensions is None:
            return

        n_frames = self.universe.trajectory.n_frames
        if n_frames < n_traj_split:
            n_traj_split = 1
            frames_start = np.array([0])
            frames_end = np.array([n_frames])
            n_frames_split = np.array([n_frames])
            interval_indices = [[0]]
        else:
            run_len = int(n_frames / n_traj_split)
            frames_start = np.arange(n_traj_split) * run_len
            frames_end = frames_start + run_len
            frames_end[-1] = n_frames
            n_frames_split = frames_end - frames_start
            assert np.sum(n_frames_split) == n_frames
            if not interval_indices:
                interval_indices = [[i] for i in range(n_traj_split)]

        bead_groups = self.bead_groups
        atoms_moltypes = self.get('atoms_info', {}).get('moltypes', [])
        moltypes = np.unique(atoms_moltypes)
        if bead_groups is None:
            return {}

        min_box_dimension = np.min(self.universe.trajectory[0].dimensions[:3])
        max_rdf_dist = min_box_dimension / 2

        rdf_results = {}
        rdf_results['n_smooth'] = n_smooth
        rdf_results['types'] = []
        rdf_results['variables_name'] = []
        rdf_results['bins'] = []
        rdf_results['value'] = []
        rdf_results['frame_start'] = []
        rdf_results['frame_end'] = []
        for i, moltype_i in enumerate(moltypes):
            for j, moltype_j in enumerate(moltypes):
                if j > i:
                    continue
                elif i == j and bead_groups[moltype_i].positions.shape[0] == 1:  # skip if only 1 mol in group
                    continue

                if i == j:
                    exclusion_block = (1, 1)  # remove self-distance
                else:
                    exclusion_block = None
                pair_type = moltype_i + '-' + moltype_j
                rdf_results_tmp = {}
                rdf_results_tmp['types'] = []
                rdf_results_tmp['variables_name'] = []
                rdf_results_tmp['bins'] = []
                rdf_results_tmp['value'] = []
                rdf_results_tmp['frame_start'] = []
                rdf_results_tmp['frame_end'] = []
                for i_interval in range(n_traj_split):
                    rdf_results_tmp['types'].append(pair_type)
                    rdf_results_tmp['variables_name'].append(['distance'])
                    rdf = MDA_RDF.InterRDF(bead_groups[moltype_i], bead_groups[moltype_j],
                                           range=(0, max_rdf_dist), exclusion_block=exclusion_block,
                                           nbins=n_bins).run(frames_start[i_interval], frames_end[i_interval], n_prune)
                    rdf_results_tmp['frame_start'].append(frames_start[i_interval])
                    rdf_results_tmp['frame_end'].append(frames_end[i_interval])

                    rdf_results_tmp['bins'].append(rdf.results.bins[int(n_smooth / 2):-int(n_smooth / 2)] * ureg.angstrom)
                    rdf_results_tmp['value'].append(np.convolve(
                        rdf.results.rdf, np.ones((n_smooth,)) / n_smooth,
                        mode='same')[int(n_smooth / 2):-int(n_smooth / 2)])

                for interval_group in interval_indices:
                    get_rdf_avg(rdf_results_tmp, rdf_results, interval_group, n_frames_split)

        return rdf_results

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
        return [guess_atom_element(name) for name in self.get('atoms_info', {}).get('names', [])]

    def get_time(self, frame_index):
        '''
        Returns the elapsed simulated physical time since the start of the simulation for index frame_index.
        '''
        frame = self.get_frame(frame_index)
        return frame.time * ureg.picosecond if frame is not None else None

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

    def __calc_diffusion_constant(self, times: NDArray, values: NDArray, dim: int = 3):
        '''
        Determines the diffusion constant from a fit of the mean squared displacement
        vs. time according to the Einstein relation.
        '''
        linear_model = linregress(times, values)
        slope = linear_model.slope
        error = linear_model.rvalue
        return slope * 1 / (2 * dim), error

    def calc_molecular_mean_squared_displacements(self):
        '''
        Calculates the mean squared displacement for the center of mass of each
        molecule type.
        '''

        def mean_squared_displacement(start: np.ndarray, current: np.ndarray):
            """
            Calculates mean square displacement between current and initial (start) coordinates.
            """
            vec = start - current
            return (vec ** 2).sum(axis=1).mean()

        if self.universe is None:
            return
        if self.universe.trajectory[0].dimensions is None:
            return

        atoms_moltypes = self.get('atoms_info', {}).get('moltypes', [])
        moltypes = np.unique(atoms_moltypes)
        dt = self.universe.trajectory.dt
        n_frames = self.universe.trajectory.n_frames
        times = np.arange(n_frames) * dt

        if n_frames < 50:
            return

        bead_groups = self.bead_groups
        msd_results = {}
        msd_results['value'] = []
        msd_results['times'] = []
        msd_results['diffusion_constant'] = []
        msd_results['error_diffusion_constant'] = []
        for moltype in moltypes:
            positions = self.get_nojump_positions(bead_groups[moltype])
            results = shifted_correlation_average(mean_squared_displacement, times, positions)
            msd_results['value'].append(results[1])
            msd_results['times'].append(results[0])
            diffusion_constant, error = self.__calc_diffusion_constant(*results)
            msd_results['diffusion_constant'].append(diffusion_constant)
            msd_results['error_diffusion_constant'].append(error)

        msd_results['types'] = moltypes
        msd_results['times'] = np.array(msd_results['times']) * ureg.picosecond
        msd_results['value'] = np.array(msd_results['value']) * ureg.angstrom**2
        msd_results['diffusion_constant'] = (np.array(
            msd_results['diffusion_constant']) * ureg.angstrom**2 / ureg.picosecond)
        msd_results['error_diffusion_constant'] = np.array(msd_results['error_diffusion_constant'])

        return msd_results

    def parse_jumps(self, selection):
        __ = self.universe.trajectory[0]
        prev = np.array(selection.positions)
        box = self.universe.trajectory[0].dimensions[:3]
        sparse_data = namedtuple('SparseData', ['data', 'row', 'col'])
        jump_data = (
            sparse_data(data=array('b'), row=array('l'), col=array('l')),
            sparse_data(data=array('b'), row=array('l'), col=array('l')),
            sparse_data(data=array('b'), row=array('l'), col=array('l'))
        )

        for i_frame, _ in enumerate(self.universe.trajectory[1:]):
            curr = np.array(selection.positions)
            delta = ((curr - prev) / box).round().astype(np.int8)
            prev = np.array(curr)
            for d in range(3):
                col, = np.where(delta[:, d] != 0)
                jump_data[d].col.extend(col)
                jump_data[d].row.extend([i_frame] * len(col))
                jump_data[d].data.extend(delta[col, d])

        return jump_data

    def generate_nojump_matrices(self, selection):
        jump_data = self.parse_jumps(selection)
        N = len(self.universe.trajectory)
        M = selection.positions.shape[0]

        nojump_matrices = tuple(
            sparse.csr_matrix((np.array(m.data), (m.row, m.col)), shape=(N, M)) for m in jump_data
        )
        return nojump_matrices

    def get_nojump_positions(self, selection):
        nojump_matrices = self.generate_nojump_matrices(selection)
        box = self.universe.trajectory[0].dimensions[:3]

        nojump_positions = []
        for i_frame, __ in enumerate(self.universe.trajectory):
            delta = np.array(np.vstack(
                [m[:i_frame, :].sum(axis=0) for m in nojump_matrices]
            ).T) * box
            nojump_positions.append(selection.positions - delta)

        return np.array(nojump_positions)
