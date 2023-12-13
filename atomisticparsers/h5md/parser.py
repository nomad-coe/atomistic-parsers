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
import h5py

from nomad.datamodel import EntryArchive
from nomad.metainfo.util import MEnum
from nomad.parsing.file_parser import FileParser
from nomad.datamodel.metainfo.simulation.run import Run, Program  # TimeRun
from nomad.datamodel.metainfo.simulation.method import (
    Method, ForceField, Model, AtomParameters, ForceCalculations, NeighborSearching
)
from nomad.datamodel.metainfo.simulation.system import (
    AtomsGroup
)
from nomad.datamodel.metainfo.simulation.calculation import (
    Calculation, Energy, EnergyEntry, BaseCalculation
)
from nomad.datamodel.metainfo.simulation.workflow import (
    EnsembleProperty, EnsemblePropertyValues, CorrelationFunction, CorrelationFunctionValues
)
from atomisticparsers.utils import MDParser, MOL
from .metainfo.h5md import ParamEntry, CalcEntry, Author
from nomad.units import ureg


class HDF5Parser(FileParser):
    def __init__(self):
        super().__init__(None)

    @property
    def filehdf5(self):
        if self._file_handler is None:
            try:
                self._file_handler = h5py.File(self.mainfile, 'r')
            except Exception:
                self.logger.error('Error reading hdf5 file.')

        return self._file_handler

    def apply_unit(self, quantity, unit: str, unit_factor: float):

        if quantity is None:
            return
        if unit:
            unit = ureg(unit)
            unit *= unit_factor
            quantity *= unit

        return quantity

    def decode_bytes(self, dataset):
        if dataset is None:
            return None
        elif isinstance(dataset, np.ndarray):
            if dataset.size == 0:
                return None
            dataset = [val.decode("utf-8") for val in dataset] if isinstance(dataset[0], bytes) else dataset
            dataset = [val.__bool__() for val in dataset] if isinstance(dataset[0], bool) else dataset
        elif type(dataset).__name__ == 'bool_':  # TODO fix error when using isinstance() here
            dataset = dataset.__bool__()
        else:
            dataset = dataset.decode("utf-8") if isinstance(dataset, bytes) else dataset
        return dataset

    def get_attribute(self, group, attribute: str = None, path: str = None, default=None):
        '''
        Extracts attribute from group object based on path, and returns default if not defined.
        '''
        if path:
            section_segments = path.split('.')
            for section in section_segments:
                try:
                    value = group.get(section)
                    group = value
                except Exception:
                    return
        value = group.attrs.get(attribute)
        value = self.decode_bytes(value) if value is not None else default

        return value if value is not None else default

    def get_group_dataset(self, group, path: str, default=None):
        '''
        Extracts group or dataset from group object based on path, and returns default if not defined.
        '''
        section_segments = path.split('.')
        for section in section_segments:
            try:
                value = group.get(section)
                unit = self.get_attribute(group, 'unit', path=section)
                unit_factor = self.get_attribute(group, 'unit_factor', path=section, default=1.0)
                group = value
            except Exception:
                return

        if value is None:
            value = default
        elif isinstance(value, h5py.Dataset):
            value = value[()]
            value = self.apply_unit(value, unit, unit_factor)
        value = self.decode_bytes(value)

        return value if value is not None else default

    def parse(self, quantity_key: str = None, **kwargs):
        pass


class H5MDParser(MDParser):
    def __init__(self):
        super().__init__()
        self._data_parser = HDF5Parser()
        self._n_frames = None  # TODO use trajectory_steps from MDParser?
        self._n_atoms = None
        self._frame_rate = None
        self._atom_parameters = None
        self._system_info = None
        self._observable_info = None
        self._parameter_info = None
        self._h5md_groups = None
        self._h5md_particle_group_all = None
        self._h5md_positions_group_all = None
        self._h5md_positions_value_all = None

        self._nomad_to_particles_group_map = {
            'positions': 'position',
            'velocities': 'velocity',
            'forces': 'force',
            'labels': 'species_label',
            'label': 'force_field_label',
            'mass': 'mass',
            'charge': 'charge',
        }

        self._nomad_to_box_group_map = {
            'lattice_vectors': 'edges',
            'periodic': 'boundary',
            'dimension': 'dimension'
        }

    @property
    def h5md_groups(self):
        if self._h5md_groups is None:
            if not self._data_parser.filehdf5:
                return {}
            groups = ['h5md', 'particles', 'observables', 'connectivity', 'parameters']
            self._h5md_groups = {group: self._data_parser.get_group_dataset(self._data_parser.filehdf5, group) for group in groups}
        return self._h5md_groups

    @property
    def h5md_particle_group_all(self):
        if self._h5md_particle_group_all is None:
            group_particles = self.h5md_groups.get('particles')
            if not group_particles:
                return
            self._h5md_particle_group_all = self._data_parser.get_group_dataset(group_particles, 'all')
        return self._h5md_particle_group_all

    @property
    def h5md_positions_group_all(self):
        if self._h5md_positions_group_all is None:
            if self.h5md_particle_group_all is None:
                return
            self._h5md_positions_group_all = self._data_parser.get_group_dataset(self.h5md_particle_group_all, 'position')
        return self._h5md_positions_group_all

    @property
    def h5md_positions_value_all(self):
        if self._h5md_positions_value_all is None:
            if self.h5md_positions_group_all is None:
                return
            self._h5md_positions_value_all = self._data_parser.get_group_dataset(self.h5md_positions_group_all, 'value')
        return self._h5md_positions_value_all

    @property
    def n_frames(self):
        if self._n_frames is None:
            if self.h5md_positions_value_all is None:
                return
            self._n_frames = len(self.h5md_positions_value_all) if self.h5md_positions_value_all is not None else None
        return self._n_frames

    @property
    def n_atoms(self):
        if self._n_atoms is None:
            if self.h5md_positions_value_all is None:
                return
            self._n_atoms = [len(pos) for pos in self.h5md_positions_value_all] if self.h5md_positions_value_all is not None else None
        return self._n_atoms

    @property
    def frame_rate(self):  # TODO extend to non-fixed n_atoms
        if self._frame_rate is None:
            if self._n_atoms == 0 or self._n_frames == 0:
                self._frame_rate = 1
            else:
                cum_atoms = self._n_atoms * self._n_frames
                self._frame_rate = 1 if cum_atoms <= self._cum_max_atoms else cum_atoms // self._cum_max_atoms
        return self._frame_rate

    @property
    def atom_parameters(self):
        if self._atom_parameters is None:
            if not self.h5md_particle_group_all:
                return {}
            if self.n_atoms is None:
                return {}
            self._atom_parameters = {}
            n_atoms = self.n_atoms[0]  # TODO Extend to non-static n_atoms
            particles = self.h5md_particle_group_all  # TODO Extend to arbitrary particle groups

            atom_parameter_keys = ['label', 'mass', 'charge']
            for key in atom_parameter_keys:
                value = self._data_parser.get_group_dataset(particles, self._nomad_to_particles_group_map[key])
                if value is not None:
                    self._atom_parameters[key] = value
                else:
                    continue
                if isinstance(self._atom_parameters[key], h5py.Group):
                    self.logger.warning('Time-dependent atom parameters currently not supported. Atom parameter values will not be stored.')
                    continue
                elif len(self._atom_parameters[key]) != n_atoms:
                    self.logger.warning('Inconsistent length of some atom parameters. Atom parameter values will not be stored.')
                    continue
        return self._atom_parameters

    @property
    def system_info(self):
        if self._system_info is None:
            particles_group = self.h5md_particle_group_all
            positions_group = self.h5md_positions_group_all
            positions_value = self.h5md_positions_value_all
            if not particles_group:
                return
            self._system_info = {'system': {}, 'calculation': {}}
            n_frames = self.n_frames

            if positions_value is None:  # For now we require that positions are present in the H5MD file to store other particle attributes
                self.logger.warning('No positions available in H5MD file. Other particle attributes will not be stored')
                return
            self._system_info['system']['positions'] = positions_value
            self._system_info['system']['n_atoms'] = self.n_atoms
            # get the times and steps based on the positions
            self._system_info['system']['steps'] = self._data_parser.get_group_dataset(positions_group, 'step')
            self._system_info['system']['times'] = self._data_parser.get_group_dataset(positions_group, 'time')

            # get the remaining system particle quantities
            system_keys = {'labels': 'system', 'velocities': 'system', 'forces': 'calculation'}
            for key, sec_key in system_keys.items():
                value = self._data_parser.get_group_dataset(particles_group, self._nomad_to_particles_group_map[key])
                if value is not None:
                    self._system_info[sec_key][key] = value
                else:
                    continue

                if isinstance(self._system_info[sec_key][key], h5py.Group):
                    self._system_info[sec_key][key] = self._data_parser.get_group_dataset(self._system_info[sec_key][key], 'value')
                    if self._system_info[sec_key][key] is None:
                        continue
                    elif len(self._system_info[sec_key][key]) != n_frames:  # TODO Should really check that the stored times for these quantities are exact, not just the same length
                        self.logger.warning('Distinct trajectory lengths of particle attributes not supported. These attributes will not be stored.')
                        continue
                else:
                    self._system_info[sec_key][key] = [self._system_info[sec_key][key]] * n_frames

            # TODO Should we extend this to pick up additional attributes in the particles group? Or require that we follow the H5MD schema strictly?

            # get the system box quantities
            box = self._data_parser.get_group_dataset(particles_group, 'box')
            if box is None:
                return

            box_attributes = {'dimension': 'system', 'periodic': 'system'}
            for box_key, sec_key in box_attributes.items():
                value = self._data_parser.get_attribute(box, self._nomad_to_box_group_map[box_key], path=None)
                if value is not None:
                    self._system_info[sec_key][box_key] = [value] * n_frames

            box_keys = {'lattice_vectors': 'system'}
            for box_key, sec_key in box_keys.items():
                value = self._data_parser.get_group_dataset(box, self._nomad_to_box_group_map[box_key])
                if value is not None:
                    self._system_info[sec_key][box_key] = value
                else:
                    continue

                if isinstance(self._system_info[sec_key][box_key], h5py.Group):
                    self._system_info[sec_key][box_key] = self._data_parser.get_group_dataset(self._system_info[sec_key][box_key], 'value')
                    if self._system_info[sec_key][box_key] is None:
                        continue
                    elif len(self._system_info[sec_key][box_key]) != n_frames:  # TODO Should really check that the stored times for these quantities are exact, not just the same length
                        self.logger.warning('Distinct trajectory lengths of box vectors and positions is not supported. These values will not be stored.')
                        continue
                else:
                    self._system_info[sec_key][box_key] = [self._system_info[sec_key][box_key]] * n_frames
        return self._system_info

    @property
    def observable_info(self):
        if self._observable_info is None:
            observables_group = self.h5md_groups.get('observables')
            if observables_group is None:
                return

            self._observable_info = {
                'configurational': {},
                'ensemble_average': {},
                'correlation_function': {}
            }

            def get_observable_paths(observable_group, current_path, paths):
                for obs_key in observable_group.keys():
                    path = obs_key + '.'
                    observable = self._data_parser.get_group_dataset(observable_group, obs_key)
                    observable_type = self._data_parser.get_group_dataset(observable_group, obs_key).attrs.get('type')
                    if not observable_type:
                        paths = get_observable_paths(observable, current_path + path, paths)
                    else:
                        paths.append(current_path + path[:-1])

                return paths

            observable_paths = get_observable_paths(observables_group, current_path='', paths=[])

            for path in observable_paths:
                observable = self._data_parser.get_group_dataset(observables_group, path)
                observable_type = self._data_parser.get_group_dataset(observables_group, path).attrs.get('type')
                observable_name = path.split('.')[0]
                observable_label = '-'.join(path.split('.')[1:]) if len(path.split('.')) > 1 else ''
                if observable_name not in self._observable_info[observable_type].keys():
                    self._observable_info[observable_type][observable_name] = {}
                self._observable_info[observable_type][observable_name][observable_label] = {}
                for key in observable.keys():
                    observable_attribute = self._data_parser.get_group_dataset(observable, key)
                    if isinstance(observable_attribute, h5py.Group):
                        self.logger.warning('Group structures within individual observables not supported. These values will not be stored.')
                        continue
                    self._observable_info[observable_type][observable_name][observable_label][key] = observable_attribute
        return self._observable_info

    def get_atomsgroup_fromh5md(self, nomad_sec, h5md_sec_particlesgroup):
        for i_key, key in enumerate(h5md_sec_particlesgroup.keys()):
            particles_group = {group_key: self._data_parser.get_group_dataset(h5md_sec_particlesgroup[key], group_key) for group_key in h5md_sec_particlesgroup[key].keys()}
            sec_atomsgroup = nomad_sec.m_create(AtomsGroup)
            sec_atomsgroup.type = particles_group.pop('type', None)
            sec_atomsgroup.index = i_key
            sec_atomsgroup.atom_indices = particles_group.pop('indices', None)
            sec_atomsgroup.n_atoms = len(sec_atomsgroup.atom_indices) if sec_atomsgroup.atom_indices is not None else None
            sec_atomsgroup.is_molecule = particles_group.pop('is_molecule', None)
            sec_atomsgroup.label = particles_group.pop('label', None)
            sec_atomsgroup.composition_formula = particles_group.pop('formula', None)
            particles_subgroup = particles_group.pop('particles_group', None)
            # set the remaining attributes
            for particles_group_key in particles_group.keys():
                val = particles_group.get(particles_group_key)
                units = val.units if hasattr(val, 'units') else None
                if units is not None:
                    val = val.magnitude
                sec_atomsgroup.x_h5md_parameters.append(ParamEntry(kind=particles_group_key, value=val, unit=units))
            # get the next atomsgroup
            if particles_subgroup:
                self.get_atomsgroup_fromh5md(sec_atomsgroup, particles_subgroup)

    def check_metainfo_for_key_and_Enum(self, metainfo_class, key, val):
        if key in metainfo_class.__dict__.keys():
            quant = metainfo_class.__dict__.get(key)
            if quant.get('type') is not None:
                if isinstance(quant.type, MEnum):
                    if val in quant.type._list:
                        return key
                    else:
                        return 'x_h5md_' + key
                else:
                    return key
            else:
                # TODO remove the variables in the warning below
                self.logger.warning(key + 'in ' + metainfo_class + ' is not a Quantity or does not have an associated type.')  # Not sure if this can ever happen
                return key
        else:
            return 'x_h5md_' + key

    @property
    def parameter_info(self):
        if self._parameter_info is None:
            parameters_group = self.h5md_groups.get('parameters')
            if parameters_group is None:
                return
            self._parameter_info = {
                'force_calculations': {},
                'workflow': {}
            }

            def get_parameters(parameter_group):
                param_dict = {}
                for key, val in parameter_group.items():
                    if isinstance(val, h5py.Group):
                        param_dict[key] = get_parameters(val)
                    else:
                        param_dict[key] = self._data_parser.get_group_dataset(parameter_group, key)
                        if isinstance(param_dict[key], str):
                            if key == 'thermodynamic_ensemble':
                                param_dict[key] = param_dict[key].upper()  # TODO change enums to lower case and adjust Gromacs and Lammps code accordingly
                            else:
                                param_dict[key] = param_dict[key].lower()
                        elif isinstance(param_dict[key], (int, np.int32, np.int64)):
                            param_dict[key] = param_dict[key].item()

                return param_dict

            force_calculations_group = self._data_parser.get_group_dataset(parameters_group, 'force_calculations')
            if force_calculations_group is not None:
                self._parameter_info['force_calculations'] = get_parameters(force_calculations_group)
            workflow_group = self._data_parser.get_group_dataset(parameters_group, 'workflow')
            if workflow_group is not None:
                self._parameter_info['workflow'] = get_parameters(workflow_group)

        return self._parameter_info

    def parse_calculation(self):
        sec_run = self.archive.run[-1]
        calculation_info = self.observable_info.get('configurational')
        system_info = self._system_info.get('calculation')  # note: it is currently ensured in parse_system() that these have the same length as the system_map
        if not calculation_info:  # TODO should still create entries for system time link in this case
            return
        time_step = None  # TODO GET TIME STEP FROM PARAMS SECTION
        time_unit = ureg.picosecond
        def format_times(times):
            return np.around(times.magnitude * ureg.convert(1.0, times.units, time_unit), 5)

        system_map = {}
        system_map_key = ''
        if self._system_time_map:
            system_map_key = 'time'
            for time, i_sys in self._system_time_map.items():
                system_map[time] = {'system': i_sys}
        elif self._system_step_map:
            system_map_key = 'step'
            for step, i_sys in self._system_step_map.items():
                system_map[step] = {'system': i_sys}
        else:
            self.logger.warning('No step or time available for system data. Cannot make calculation to system references.')
            system_map_key = 'time'

        for observable_type, observable_dict in calculation_info.items():
            for key, observable in observable_dict.items():
                map_key = observable_type + '-' + key if key else observable_type
                if system_map_key == 'time':
                    times = observable.get('time')
                    if times is not None:
                        times = format_times(times)  # TODO What happens if no units are given?
                        for i_time, time in enumerate(times):
                            map_entry = system_map.get(time)
                            if map_entry:
                                map_entry[map_key] = i_time
                            else:
                                system_map[time] = {map_key: i_time}
                    else:
                        self.logger.warning('No time information available for some observables. Cannot store these values.')
                elif system_map_key == 'step':
                    steps = observable.get('step')
                    if steps:
                        steps = np.around(steps)
                        for i_step, step in enumerate(steps):
                            map_entry = system_map.get(step)
                            if map_entry:
                                map_entry[map_key] = i_step
                            else:
                                system_map[time] = {map_key: i_step}
                else:
                    self.logger.error('system_map_key not assigned correctly.')

        for frame in sorted(system_map):
            data = {
                'method_ref': sec_run.method[-1] if sec_run.method else None,
                'energy': {},
            }
            data_h5md = {
                'x_h5md_custom_calculations': [],
                'x_h5md_energy_contributions': []
            }
            if system_map_key == 'time':
                data['time'] = frame * time_unit
                if time_step:
                    data['step'] = int((frame / time_step).magnitude)
            elif system_map_key == 'step':
                data['step'] = frame
                if time_step:
                    data['time'] = data['step'] * time_step

            system_index = system_map[frame]['system']
            if system_index is not None:
                for key, val in system_info.items():
                    if key == 'forces':
                        data[key] = dict(total=dict(value=val[system_index]))
                    else:
                        if key in BaseCalculation.__dict__.keys():
                            data[key] = val[system_index]
                        else:
                            unit = None
                            if hasattr(val, 'units'):
                                unit = val.units
                                val = val.magnitude
                            data_h5md['x_h5md_custom_calculations'].append(CalcEntry(kind=key, value=val, unit=unit))

            for observable_type, observable_dict in calculation_info.items():
                for key, observable in observable_dict.items():
                    map_key = observable_type + '-' + key if key else observable_type
                    obs_index = system_map[frame].get(map_key)
                    if obs_index:
                        val = observable.get('value', [None] * (obs_index + 1))[obs_index]
                        if 'energ' in observable_type:  # TODO check for energies or energy when matching name
                            if key in Energy.__dict__.keys():
                                data['energy'][key] = dict(value=val)
                            else:
                                data_h5md['x_h5md_energy_contributions'].append(EnergyEntry(kind=map_key, value=val))
                        else:
                            if key == '':
                                key = observable_type
                            else:
                                key = map_key

                            if key in BaseCalculation.__dict__.keys():
                                data[key] = val
                            else:
                                unit = None
                                if hasattr(val, 'units'):
                                    unit = val.units
                                    val = val.magnitude
                                data_h5md['x_h5md_custom_calculations'].append(CalcEntry(kind=map_key, value=val, unit=unit))

            self.parse_thermodynamics_step(data)
            sec_calc = sec_run.calculation[-1]
            if format_times(sec_calc.time) != frame:  # TODO check this comparison
                sec_calc = sec_run.m_create(Calculation)
                sec_calc.time = frame * time_unit
            for calc_entry in data_h5md['x_h5md_custom_calculations']:
                sec_calc.x_h5md_custom_calculations.append(calc_entry)
            sec_energy = sec_calc.energy
            if not sec_energy:
                sec_energy = sec_calc.m_create(Energy)
            for energy_entry in data_h5md['x_h5md_energy_contributions']:
                sec_energy.x_h5md_energy_contributions.append(energy_entry)

    def parse_system(self):
        sec_run = self.archive.run[-1]

        system_info = self.system_info.get('system')
        if not system_info:
            self.logger.error('No particle information found in H5MD file.')
            return

        n_frames = len(system_info.get('times', []))
        self._system_time_map = {}
        self._system_step_map = {}
        for frame in range(n_frames):
            # if (n % self.frame_rate) > 0:
            #     continue

            atoms_dict = {}
            for key in system_info.keys():
                if key == 'times':
                    time = system_info.get('times', [None] * (frame + 1))[frame]
                    if time is not None:
                        self._system_time_map[round(ureg.convert(
                            time.magnitude, time.units, ureg.picosecond), 5)] = len(self._system_time_map)
                elif key == 'steps':
                    step = system_info.get('steps', [None] * (frame + 1))[frame]
                    if step is not None:
                        self._system_step_map[round(step)] = len(self._system_step_map)
                else:
                    atoms_dict[key] = system_info.get(key, [None] * (frame + 1))[frame]

            if frame == 0:  # TODO extend to time-dependent bond lists
                connectivity = self.h5md_groups['connectivity']
                atoms_dict['bond_list'] = self._data_parser.get_group_dataset(connectivity, 'bonds')

            self.parse_trajectory_step({
                'atoms': atoms_dict
            })

            if frame == 0:  # TODO extend to time-dependent topologies
                topology = self._data_parser.get_group_dataset(connectivity, 'particles_group', None)
                if topology:
                    self.get_atomsgroup_fromh5md(sec_run.system[frame], topology)

    def parse_method(self):

        sec_method = Method()
        self.archive.run[-1].method.append(sec_method)
        sec_force_field = sec_method.m_create(ForceField)  # TODO @landinesa how do we remove m_create's here?
        sec_model = sec_force_field.m_create(Model)

        # get the atom parameters
        n_atoms = self.n_atoms[0]  # TODO Extend to non-static n_atoms
        for n in range(n_atoms):
            # sec_atom = sec_method.m_create(AtomParameters)
            sec_atom = AtomParameters()
            sec_method.atom_parameters.append(sec_atom)

            for key in self.atom_parameters.keys():
                sec_atom.m_set(sec_atom.m_get_quantity_definition(key), self.atom_parameters[key][n])

        # Get the interactions
        connectivity = self.h5md_groups.get('connectivity')
        if not connectivity:
            return

        atom_labels = self.atom_parameters.get('label')
        interaction_keys = ['bonds', 'angles', 'dihedrals', 'impropers']
        interactions_by_type = []
        for interaction_key in interaction_keys:
            interaction_list = self._data_parser.get_group_dataset(connectivity, interaction_key)
            if interaction_list is None:
                continue
            elif isinstance(interaction_list, h5py.Group):
                self.logger.warning('Time-dependent interactions currently not supported. These values will not be stored')
                continue

            interaction_type_dict = {
                'type': interaction_key,
                'n_interactions': len(interaction_list),
                'n_atoms': len(interaction_list[0]),
                'atom_indices': interaction_list,
                'atom_labels': np.array(atom_labels)[interaction_list] if atom_labels is not None else None
            }
            interactions_by_type.append(interaction_type_dict)
        self.parse_interactions_by_type(interactions_by_type, sec_model)

        # Get the force calculation parameters
        force_calculation_parameters = self.parameter_info.get('force_calculations')
        if force_calculation_parameters is None:
            return

        sec_force_calculations = sec_force_field.m_create(ForceCalculations)
        sec_neighbor_searching = sec_force_calculations.m_create(NeighborSearching)

        for key, val in force_calculation_parameters.items():
            if not isinstance(val, dict):
                key = self.check_metainfo_for_key_and_Enum(ForceCalculations, key, val)
                sec_force_calculations.m_set(sec_force_calculations.m_get_quantity_definition(key), val)
            else:
                if key == 'neighbor_searching':
                    for neigh_key, neigh_val in val.items():
                        neigh_key = self.check_metainfo_for_key_and_Enum(NeighborSearching, neigh_key, neigh_val)
                        # setattr(sec_neighbor_searching, neigh_key, neigh_val)
                        sec_neighbor_searching.m_set(sec_neighbor_searching.m_get_quantity_definition(neigh_key), neigh_val)
                else:
                    self.logger.warning('Unknown parameters in force calculations section. These will not be stored.')

    def parse_workflow(self):

        workflow_parameters = self.parameter_info.get('workflow').get('molecular_dynamics')
        # TODO should store parameters that do not match the enum vals as x_h5MD params, not sure how with MDParser??
        if workflow_parameters is None:
            return

        def get_workflow_results(property_type_dict, observables, workflow_results):

            def populate_property_dict(property_dict, val_name, val, flag_known_property=False):
                if val is None:
                    return
                value_unit = val.units if hasattr(val, 'units') else None
                value_magnitude = val.magnitude if hasattr(val, 'units') else val
                if flag_known_property:
                    property_dict[val_name] = value_magnitude * value_unit if value_unit else value_magnitude
                else:
                    property_dict[f'{val_name}_unit'] = str(value_unit) if value_unit else None
                    property_dict[f'{val_name}_magnitude'] = value_magnitude

            property_key = property_type_dict['property_type_key']
            property_value_key = property_type_dict['property_type_value_key']
            for observable_type, observable_dict in observables.items():
                flag_known_property = False
                if observable_type in property_type_dict['properties_known']:
                    property_key = observable_type
                    property_value_key = property_type_dict['properties_known'][observable_type]
                    flag_known_property = True
                workflow_results[property_key] = []
                property_dict = {property_value_key: []}
                property_dict['label'] = observable_type
                for key, observable in observable_dict.items():
                    property_values_dict = {'label': key}
                    for quant_name, val in observable.items():
                        if quant_name == 'val':
                            continue
                        if quant_name == 'bins':
                            continue
                        if quant_name in property_type_dict['property_keys_list']:
                            property_dict[quant_name] = val
                        if quant_name in property_type_dict['property_value_keys_list']:
                            property_values_dict[quant_name] = val
                        # TODO Still need to add custom values here.

                    val = observable.get('value')
                    populate_property_dict(property_values_dict, 'value', val, flag_known_property=flag_known_property)
                    bins = observable.get('bins')
                    populate_property_dict(property_values_dict, 'bins', bins, flag_known_property=flag_known_property)
                    property_dict[property_value_key].append(property_values_dict)
                workflow_results[property_key].append(property_dict)

        workflow_results = {}
        ensemble_average_observables = self.observable_info.get('ensemble_average')
        ensemble_property_dict = {
            'property_type_key': 'ensemble_properties',
            'property_type_value_key': 'ensemble_property_values',
            'properties_known': {'radial_distribution_functions': 'radial_distribution_function_values'},
            'property_keys_list': EnsembleProperty.m_def.all_quantities.keys(),
            'property_value_keys_list': EnsemblePropertyValues.m_def.all_quantities.keys()
        }
        get_workflow_results(ensemble_property_dict, ensemble_average_observables, workflow_results)
        correlation_function_observables = self.observable_info.get('correlation_function')
        correlation_function_dict = {
            'property_type_key': 'correlation_functions',
            'property_type_value_key': 'correlation_function_values',
            'properties_known': {'mean_squared_displacements': 'mean_squared_displacement_values'},
            'property_keys_list': CorrelationFunction.m_def.all_quantities.keys(),
            'property_value_keys_list': CorrelationFunctionValues.m_def.all_quantities.keys()
        }
        get_workflow_results(correlation_function_dict, correlation_function_observables, workflow_results)
        self.parse_md_workflow(dict(method=workflow_parameters, results=workflow_results))

    def init_parser(self):

        self._data_parser.mainfile = self.filepath

    def parse(self, filepath, archive: EntryArchive, logger):

        self.filepath = os.path.abspath(filepath)
        self.archive = archive
        self.logger = logging.getLogger(__name__) if logger is None else logger
        self._maindir = os.path.dirname(self.filepath)
        self._h5md_files = os.listdir(self._maindir)
        self._basename = os.path.basename(filepath).rsplit('.', 1)[0]

        self.init_parser()

        if self._data_parser.filehdf5 is None:
            self.logger.warning('hdf5 file missing in H5MD Parser.')
            return

        sec_run = Run()
        self.archive.run.append(sec_run)

        group_h5md = self.h5md_groups.get('h5md')
        if group_h5md:
            program_name = self._data_parser.get_attribute(group_h5md, 'name', path='program')
            program_version = self._data_parser.get_attribute(group_h5md, 'version', path='program')
            sec_run.program = Program(name=program_name, version=program_version)
            h5md_version = self._data_parser.get_attribute(group_h5md, 'version', path=None)
            sec_run.x_h5md_version = h5md_version
            group_author = self._data_parser.get_group_dataset(group_h5md, 'author')
            h5md_author_name = self._data_parser.get_attribute(group_author, 'name', path=None)
            h5md_author_email = self._data_parser.get_attribute(group_author, 'email', path=None)
            sec_run.x_h5md_author = Author(name=h5md_author_name, email=h5md_author_email)
            group_creator = self._data_parser.get_group_dataset(group_h5md, 'creator')
            h5md_creator_name = self._data_parser.get_attribute(group_creator, 'name', path=None)
            h5md_creator_version = self._data_parser.get_attribute(group_creator, 'version', path=None)
            sec_run.x_h5md_creator = Program(name=h5md_creator_name, version=h5md_creator_version)
        else:
            self.logger.warning('"h5md" group missing in (H5MD)hdf5 file. Program and author metadata will be missing!')

        self.parse_method()

        self.parse_system()

        self.parse_calculation()

        self.parse_workflow()
