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

from typing import List, Dict, Tuple, Any, Union, Iterable, cast, Callable, TYPE_CHECKING
from h5py import Group

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

    def get_value(self, group, path: str, default=None):
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

    # def parse(self, key, **kwargs):
    #     source = kwargs.get('source', self.filehdf5)
    #     path = kwargs.get('path')
    #     attribute = kwargs.get('attribute')
    #     if attribute is not None:
    #         value = self.hdf5_attr_getter(source, path, attribute)
    #     else:
    #         value = self.hdf5_getter(source, path)
    #     # i do not know how to construct the full path here relative to root
    #     full_path = f'...{path}.{attribute}'
    #     self._results[full_path] = value


class H5MDParser(MDParser):
    def __init__(self):
        super().__init__()
        self._data_parser = HDF5Parser()
        self._n_frames = None
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
        self._time_unit = ureg.picosecond
        self._h5md_group = None
        self._particles_group = None
        self._observables_group = None
        self._connectivity_group = None
        self._parameters_group = None

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

    def format_times(self, times):
        return np.around(times.magnitude * ureg.convert(1.0, times.units, self._time_unit), 5)

    @property
    def frame_rate(self):  # TODO extend to non-fixed n_atoms
        if self._frame_rate is None:
            if self._n_atoms == 0 or self._n_frames == 0:
                self._frame_rate = 1
            else:
                cum_atoms = self._n_atoms * self._n_frames
                self._frame_rate = 1 if cum_atoms <= self._cum_max_atoms else cum_atoms // self._cum_max_atoms
        return self._frame_rate

    def parse_atom_parameters(self):
        if not self._h5md_particle_group_all:
            return {}
        if self._n_atoms is None:
            return {}
        self._atom_parameters = {}
        n_atoms = self._n_atoms[0]  # TODO Extend to non-static n_atoms
        particles = self._h5md_particle_group_all  # TODO Extend to arbitrary particle groups

        atom_parameter_keys = ['label', 'mass', 'charge']
        for key in atom_parameter_keys:
            value = self._data_parser.get_value(particles, self._nomad_to_particles_group_map[key])
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

    def parse_system_info(self):
        self._system_info = {'system': {}, 'calculation': {}}
        particles_group = self._h5md_particle_group_all
        positions_group = self._h5md_positions_group_all
        positions_value = self._h5md_positions_value_all
        n_frames = self._n_frames
        if particles_group is None or positions_value is None or positions_value is None: # For now we require that positions are present in the H5MD file to store other particle attributes
            self.logger.warning('No positions available in H5MD file. Other particle attributes will not be stored')
            return self._system_info

        def get_value(value, steps):
            if value is None:
                return value
            if isinstance(value, h5py.Group):
                group = value
                value = self._data_parser.get_value(value, 'value')
                attr_steps = self._data_parser.get_value(group, 'step')
                if value is None or attr_steps is None:
                    self.logger.warning('Missing values or steps in particle attributes. These attributes will not be stored.')
                    return None
                elif attr_steps.sort() != steps.sort() if attr_steps is not None else False:
                    self.logger.warning('Distinct trajectory lengths of particle attributes not supported. These attributes will not be stored.')
                    return None
                else:
                    return value
            else:
                return [value] * n_frames

        # get the steps based on the positions
        steps = self._data_parser.get_value(positions_group, 'step')
        if steps is None:
            self.logger.warning('No step information available in H5MD file. System information cannot be parsed.')
            return self._system_info
        self.trajectory_steps = steps

        # get the rest of the particle quantities
        values_dict = {'system': {}, 'calculation': {}}
        times = self._data_parser.get_value(positions_group, 'time')
        values_dict['system']['time'] = times
        values_dict['calculation']['time'] = times
        values_dict['system']['positions'] = positions_value
        values_dict['system']['n_atoms'] = self._n_atoms
        system_keys = {'labels': 'system', 'velocities': 'system', 'forces': 'calculation'}
        for key, sec_key in system_keys.items():
            value = self._data_parser.get_value(particles_group, self._nomad_to_particles_group_map[key])
            values_dict[sec_key][key] = get_value(value, steps)

        # get the box quantities
        box = self._data_parser.get_value(particles_group, 'box')
        if box is not None:
            box_attributes = {'dimension': 'system', 'periodic': 'system'}
            for box_key, sec_key in box_attributes.items():
                value = self._data_parser.get_attribute(box, self._nomad_to_box_group_map[box_key], path=None)
                values_dict[sec_key][box_key] = [value] * n_frames if value is not None else None
            box_keys = {'lattice_vectors': 'system'}
            for box_key, sec_key in box_keys.items():
                value = self._data_parser.get_value(box, self._nomad_to_box_group_map[box_key])
                values_dict[sec_key][box_key] = get_value(value, steps)

        # populate the dictionary
        for i_step, step in enumerate(steps):
            self._system_info['system'][step] = {key: val[i_step] for key,val in values_dict['system'].items()}
            self._system_info['calculation'][step] = {key: val[i_step] for key,val in values_dict['calculation'].items()}

    def parse_observable_info(self):
        self._observable_info = {
            'configurational': {},
            'ensemble_average': {},
            'correlation_function': {}
        }
        thermodynamics_steps = []
        if self._observables_group is None:
            return self._observable_info

        def get_observable_paths(observable_group: Dict, current_path: str) -> List:
            paths = []
            for obs_key in observable_group.keys():
                path = obs_key + '.'
                observable = self._data_parser.get_value(observable_group, obs_key)
                observable_type = self._data_parser.get_value(observable_group, obs_key).attrs.get('type')
                if not observable_type:
                    paths.extend(get_observable_paths(observable, f'{current_path}{path}'))
                else:
                    paths.append(current_path + path[:-1])

            return paths

        observable_paths = get_observable_paths(self._observables_group, current_path='')
        for path in observable_paths:
            observable = self._data_parser.get_value(self._observables_group, path)
            observable_type = self._data_parser.get_value(self._observables_group, path).attrs.get('type')
            observable_name = path.split('.')[0]
            observable_label = '-'.join(path.split('.')[1:])
            if observable_type == 'configurational':
                steps = self._data_parser.get_value(observable, 'step')
                if steps is None:
                    self.logger.warning('Missing step information in some observables. These will not be stored.')
                    continue
                thermodynamics_steps = list(set(steps) | set(thermodynamics_steps))
                times = self._data_parser.get_value(observable, 'time')
                values = self._data_parser.get_value(observable, 'value')
                if isinstance(values, h5py.Group):
                    self.logger.warning('Group structures within individual observables not supported. These will not be stored.')
                    continue
                for i_step, step in enumerate(steps):
                    if not self._observable_info[observable_type].get(step):
                        self._observable_info[observable_type][step] = {}
                        self._observable_info[observable_type][step]['time'] = times[i_step]
                    observable_key = f'{observable_name}-{observable_label}' if observable_label else f'{observable_name}'
                    self._observable_info[observable_type][step][observable_key] = values[i_step]
            else:
                if observable_name not in self._observable_info[observable_type].keys():
                    self._observable_info[observable_type][observable_name] = {}
                self._observable_info[observable_type][observable_name][observable_label] = {}
                for key in observable.keys():
                    observable_attribute = self._data_parser.get_value(observable, key)
                    if isinstance(observable_attribute, h5py.Group):
                        self.logger.warning('Group structures within individual observables not supported. These will not be stored.')
                        continue
                    self._observable_info[observable_type][observable_name][observable_label][key] = observable_attribute

            self.thermodynamics_steps = thermodynamics_steps

    def parse_atomsgroup(self, nomad_sec, h5md_sec_particlesgroup: Group):
        for i_key, key in enumerate(h5md_sec_particlesgroup.keys()):
            particles_group = {group_key: self._data_parser.get_value(h5md_sec_particlesgroup[key], group_key) for group_key in h5md_sec_particlesgroup[key].keys()}
            sec_atomsgroup = AtomsGroup()
            nomad_sec.atoms_group.append(sec_atomsgroup)
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
                val = val.magnitude if units is not None else val
                sec_atomsgroup.x_h5md_parameters.append(ParamEntry(kind=particles_group_key, value=val, unit=units))
            # get the next atomsgroup
            if particles_subgroup:
                self.parse_atomsgroup(sec_atomsgroup, particles_subgroup)

    def is_valid_key_val(self, metainfo_class, key: str, val) -> bool:
        if hasattr(metainfo_class, key):
            quant_type = getattr(metainfo_class, key).get('type')
            is_MEnum = isinstance(quant_type, MEnum) if quant_type else False
            return False if is_MEnum and val not in quant_type._list else True
        else:
            return False

    @property
    def parameter_info(self):
        if self._parameter_info is None:
            self._parameter_info = {
                'force_calculations': {},
                'workflow': {}
            }
            if self._parameters_group is None:
                return self._parameter_info

            def get_parameters(parameter_group: Group) -> Dict:
                param_dict = {}
                for key, val in parameter_group.items():
                    if isinstance(val, h5py.Group):
                        param_dict[key] = get_parameters(val)
                    else:
                        param_dict[key] = self._data_parser.get_value(parameter_group, key)
                        if isinstance(param_dict[key], str):
                            if key == 'thermodynamic_ensemble':
                                param_dict[key] = param_dict[key].upper()  # TODO change enums to lower case and adjust Gromacs and Lammps code accordingly
                            else:
                                param_dict[key] = param_dict[key].lower()
                        elif isinstance(param_dict[key], (int, np.int32, np.int64)):
                            param_dict[key] = param_dict[key].item()

                return param_dict

            force_calculations_group = self._data_parser.get_value(self._parameters_group, 'force_calculations')
            if force_calculations_group is not None:
                self._parameter_info['force_calculations'] = get_parameters(force_calculations_group)
            workflow_group = self._data_parser.get_value(self._parameters_group, 'workflow')
            if workflow_group is not None:
                self._parameter_info['workflow'] = get_parameters(workflow_group)

        return self._parameter_info

    def parse_calculation(self):
        sec_run = self.archive.run[-1]
        calculation_info = self._observable_info.get('configurational')
        system_info = self._system_info.get('calculation')  # note: it is currently ensured in parse_system() that these have the same length as the system_map
        if not calculation_info:  # TODO should still create entries for system time link in this case
            return

        for step in self.steps:
            data = {
                'method_ref': sec_run.method[-1] if sec_run.method else None,
                'step': step,
                'energy': {},
            }
            data_h5md = {
                'x_h5md_custom_calculations': [],
                'x_h5md_energy_contributions': []
            }
            data['time'] = calculation_info.get('step', {}).get('time') # * self._time_unit
            if not data['time']:
                data['time'] = system_info.get('step', {}).get('time')

            if system_info.get(step):
                for key, val in system_info[step].items():
                    if key == 'forces':
                        data[key] = dict(total=dict(value=val))
                    else:
                        if hasattr(BaseCalculation, key):
                            data[key] = val
                        else:
                            unit = None
                            if hasattr(val, 'units'):
                                unit = val.units
                                val = val.magnitude
                            data_h5md['x_h5md_custom_calculations'].append(CalcEntry(kind=key, value=val, unit=unit))

            if calculation_info.get(step):
                for key, val in calculation_info[step].items():
                    key_split = key.split('-')
                    observable_name = key_split[0]
                    observable_label = key_split[1] if len(key_split) > 1 else key_split[0]
                    if 'energ' in observable_name:  # TODO check for energies or energy when matching name
                        if hasattr(Energy, observable_label):
                            data['energy'][observable_label] = dict(value=val)
                        else:
                            data_h5md['x_h5md_energy_contributions'].append(EnergyEntry(kind=key, value=val))
                    else:
                        if hasattr(BaseCalculation, observable_label):
                            data[observable_label] = val
                        else:
                            unit = None
                            if hasattr(val, 'units'):
                                unit = val.units
                                val = val.magnitude
                            data_h5md['x_h5md_custom_calculations'].append(CalcEntry(kind=key, value=val, unit=unit))

            self.parse_thermodynamics_step(data)
            sec_calc = sec_run.calculation[-1]

            if sec_calc.step != step:  # TODO check this comparison
                sec_calc = sec_run.m_create(Calculation)
                sec_calc.step = int(step)
                sec_calc.time = data['time']
            for calc_entry in data_h5md['x_h5md_custom_calculations']:
                sec_calc.x_h5md_custom_calculations.append(calc_entry)
            sec_energy = sec_calc.energy
            if not sec_energy:
                sec_energy = sec_calc.m_create(Energy)
            for energy_entry in data_h5md['x_h5md_energy_contributions']:
                sec_energy.x_h5md_energy_contributions.append(energy_entry)

    def parse_system(self):
        sec_run = self.archive.run[-1]

        system_info = self._system_info.get('system')
        if not system_info:
            self.logger.error('No particle information found in H5MD file.')
            return

        self._system_time_map = {}
        for i_step, step in enumerate(self.trajectory_steps):
            time = system_info[step].pop('time')
            atoms_dict = system_info[step]

            topology = None
            if i_step == 0 and self._connectivity_group: # TODO extend to time-dependent bond lists and topologies
                atoms_dict['bond_list'] = self._data_parser.get_value(self._connectivity_group, 'bonds')
                topology = self._data_parser.get_value(self._connectivity_group, 'particles_group', None)

            self.parse_trajectory_step({
                'atoms': atoms_dict
            })

            if topology:
                self.parse_atomsgroup(sec_run.system[i_step], topology)

    def parse_method(self):

        sec_method = Method()
        self.archive.run[-1].method.append(sec_method)
        sec_force_field = sec_method.m_create(ForceField)  # TODO @landinesa how do we remove m_create's here?
        sec_model = sec_force_field.m_create(Model)

        # get the atom parameters
        n_atoms = self._n_atoms[0] if self._n_atoms is not None else 0  # TODO Extend to non-static n_atoms
        for n in range(n_atoms):
            # sec_atom = sec_method.m_create(AtomParameters)
            sec_atom = AtomParameters()
            sec_method.atom_parameters.append(sec_atom)

            for key in self._atom_parameters.keys():
                sec_atom.m_set(sec_atom.m_get_quantity_definition(key), self._atom_parameters[key][n])

        # Get the interactions
        if self._connectivity_group:
            atom_labels = self._atom_parameters.get('label')
            interaction_keys = ['bonds', 'angles', 'dihedrals', 'impropers']
            interactions_by_type = []
            for interaction_key in interaction_keys:
                interaction_list = self._data_parser.get_value(self._connectivity_group, interaction_key)
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
        if force_calculation_parameters:
            sec_force_calculations = sec_force_field.m_create(ForceCalculations)
            sec_neighbor_searching = sec_force_calculations.m_create(NeighborSearching)

            for key, val in force_calculation_parameters.items():
                if not isinstance(val, dict):
                    if self.is_valid_key_val(ForceCalculations, key, val):
                        sec_force_calculations.m_set(sec_force_calculations.m_get_quantity_definition(key), val)
                    else:
                        units = val.units if hasattr(val, 'units') else None
                        val = val.magnitude if units is not None else val
                        sec_force_calculations.x_h5md_parameters.append(ParamEntry(kind=key, value=val, unit=units))
                else:
                    if key == 'neighbor_searching':
                        for neigh_key, neigh_val in val.items():
                            if self.is_valid_key_val(NeighborSearching, neigh_key, neigh_val):
                                sec_neighbor_searching.m_set(sec_neighbor_searching.m_get_quantity_definition(neigh_key), neigh_val)
                            else:
                                units = val.units if hasattr(val, 'units') else None
                                val = val.magnitude if units is not None else val
                                sec_neighbor_searching.x_h5md_parameters.append(ParamEntry(kind=key, value=val, unit=units))
                    else:
                        self.logger.warning('Unknown parameters in force calculations section. These will not be stored.')

    def get_workflow_properties_dict(
            self, observables: Dict, property_type_key=None, property_type_value_key=None,
            properties_known={}, property_keys_list=[], property_value_keys_list=[]):

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

        workflow_properties_dict = {}
        for observable_type, observable_dict in observables.items():
            flag_known_property = False
            if observable_type in properties_known:
                property_type_key = observable_type
                property_type_value_key = properties_known[observable_type]
                flag_known_property = True
            property_dict = {property_type_value_key: []}
            property_dict['label'] = observable_type
            for key, observable in observable_dict.items():
                property_values_dict = {'label': key}
                for quant_name, val in observable.items():
                    if quant_name == 'val':
                        continue
                    if quant_name == 'bins':
                        continue
                    if quant_name in property_keys_list:
                        property_dict[quant_name] = val
                    if quant_name in property_value_keys_list:
                        property_values_dict[quant_name] = val
                    # TODO Still need to add custom values here.

                val = observable.get('value')
                populate_property_dict(property_values_dict, 'value', val, flag_known_property=flag_known_property)
                bins = observable.get('bins')
                populate_property_dict(property_values_dict, 'bins', bins, flag_known_property=flag_known_property)
                property_dict[property_type_value_key].append(property_values_dict)

            if workflow_properties_dict.get(property_type_key):
                workflow_properties_dict[property_type_key].append(property_dict)
            else:
                workflow_properties_dict[property_type_key] = [property_dict]

        return workflow_properties_dict

    def parse_workflow(self):

        workflow_parameters = self.parameter_info.get('workflow').get('molecular_dynamics')
        # TODO should store parameters that do not match the enum vals as x_h5MD params,
        # not sure how with MDParser??
        if workflow_parameters is None:
            return

        workflow_results = {}
        ensemble_average_observables = self._observable_info.get('ensemble_average')
        ensemble_property_dict = {
            'property_type_key': 'ensemble_properties',
            'property_type_value_key': 'ensemble_property_values',
            'properties_known': {'radial_distribution_functions': 'radial_distribution_function_values'},
            'property_keys_list': EnsembleProperty.m_def.all_quantities.keys(),
            'property_value_keys_list': EnsemblePropertyValues.m_def.all_quantities.keys()
        }
        workflow_results.update(
            self.get_workflow_properties_dict(ensemble_average_observables, **ensemble_property_dict)
            )
        correlation_function_observables = self._observable_info.get('correlation_function')
        correlation_function_dict = {
            'property_type_key': 'correlation_functions',
            'property_type_value_key': 'correlation_function_values',
            'properties_known': {'mean_squared_displacements': 'mean_squared_displacement_values'},
            'property_keys_list': CorrelationFunction.m_def.all_quantities.keys(),
            'property_value_keys_list': CorrelationFunctionValues.m_def.all_quantities.keys()
        }
        workflow_results.update(
            self.get_workflow_properties_dict(correlation_function_observables, **correlation_function_dict)
            )
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

        # Get the overarching H5MD groups
        self._h5md_group = self._data_parser.get_value(self._data_parser.filehdf5, 'h5md')
        self._particles_group = self._data_parser.get_value(self._data_parser.filehdf5, 'particles')
        if self._particles_group:
            self._h5md_particle_group_all = self._data_parser.get_value(self._particles_group, 'all')
        if self._h5md_particle_group_all:
            self._h5md_positions_group_all = self._data_parser.get_value(self._h5md_particle_group_all, 'position')
        if self._h5md_positions_group_all:
            self._h5md_positions_value_all = self._data_parser.get_value(self._h5md_positions_group_all, 'value')
        if self._h5md_positions_value_all is not None:
            self._n_frames = len(self._h5md_positions_value_all) if self._h5md_positions_value_all is not None else None
            self._n_atoms = [len(pos) for pos in self._h5md_positions_value_all] if self._h5md_positions_value_all is not None else None
        self._observables_group = self._data_parser.get_value(self._data_parser.filehdf5, 'observables')
        self._connectivity_group = self._data_parser.get_value(self._data_parser.filehdf5, 'connectivity')
        self._parameters_group = self._data_parser.get_value(self._data_parser.filehdf5, 'parameters')

        # Parse the hdf5 groups
        sec_run = Run()
        self.archive.run.append(sec_run)

        group_h5md = self._h5md_group
        if group_h5md:
            program_name = self._data_parser.get_attribute(group_h5md, 'name', path='program')
            program_version = self._data_parser.get_attribute(group_h5md, 'version', path='program')
            sec_run.program = Program(name=program_name, version=program_version)
            h5md_version = self._data_parser.get_attribute(group_h5md, 'version', path=None)
            sec_run.x_h5md_version = h5md_version
            group_author = self._data_parser.get_value(group_h5md, 'author')
            h5md_author_name = self._data_parser.get_attribute(group_author, 'name', path=None)
            h5md_author_email = self._data_parser.get_attribute(group_author, 'email', path=None)
            sec_run.x_h5md_author = Author(name=h5md_author_name, email=h5md_author_email)
            group_creator = self._data_parser.get_value(group_h5md, 'creator')
            h5md_creator_name = self._data_parser.get_attribute(group_creator, 'name', path=None)
            h5md_creator_version = self._data_parser.get_attribute(group_creator, 'version', path=None)
            sec_run.x_h5md_creator = Program(name=h5md_creator_name, version=h5md_creator_version)
        else:
            self.logger.warning('"h5md" group missing in (H5MD)hdf5 file. Program and author metadata will be missing!')

        self.parse_atom_parameters()
        self.parse_system_info()
        self.parse_observable_info()

        # Populate the archive
        self.parse_method()

        self.parse_system()

        self.parse_calculation()

        self.parse_workflow()
