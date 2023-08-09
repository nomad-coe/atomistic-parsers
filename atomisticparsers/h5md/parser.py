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

# from nomad.units import ureg
from nomad.parsing.file_parser import FileParser
from nomad.datamodel.metainfo.simulation.run import Run, Program  #, TimeRun
from nomad.datamodel.metainfo.simulation.method import (
    Method, ForceField, Model, Interaction, AtomParameters, ForceCalculations, NeighborSearching
)
from nomad.datamodel.metainfo.simulation.system import (
    System, Atoms, AtomsGroup
)
from nomad.datamodel.metainfo.simulation.calculation import (
    Calculation, Energy, EnergyEntry, Forces, ForcesEntry, BaseCalculation
)
from nomad.datamodel.metainfo.workflow import (
    Workflow
)
# from nomad.datamodel.metainfo.simulation import workflow as workflow2
# from nomad.datamodel.metainfo.simulation.workflow import MolecularDynamics
from nomad.datamodel.metainfo.simulation.workflow import (
    GeometryOptimization, GeometryOptimizationMethod, GeometryOptimizationResults,
    BarostatParameters, ThermostatParameters, DiffusionConstantValues,
    MeanSquaredDisplacement, MeanSquaredDisplacementValues, MolecularDynamicsResults,
    RadialDistributionFunction, RadialDistributionFunctionValues,
    MolecularDynamics, MolecularDynamicsMethod
)
from .metainfo.h5md import ParamEntry, CalcEntry
# from nomad.atomutils import get_molecules_from_bond_list, is_same_molecule, get_composition
from nomad.units import ureg

MOL = 6.022140857e+23

class H5MDParser(FileParser):
    def __init__(self):
        self._n_frames = None
        self._n_atoms = None
        self._frame_rate = None
        self._atom_parameters = None
        self._system_info = None
        self._observable_info = None
        self._parameter_info = None
        self._hdf5_particle_group_all = None
        self._hdf5_positions_all = None
        # max cumulative number of atoms for all parsed trajectories to calculate sampling rate
        self._cum_max_atoms = 2500000

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
    def filehdf5(self):
        if self._file_handler is None:
            try:
                self._file_handler = h5py.File(self.mainfile, 'r')
            except Exception:
                self.logger.error('Error reading hdf5(h5MD) file.')

        return self._file_handler

    @property
    def hdf5_particle_group_all(self):
        if self._hdf5_particle_group_all is None:
            if not self.filehdf5:
                return
            self._hdf5_particle_group_all = self.hdf5_getter(self.filehdf5, 'particles.all')
        return self._hdf5_particle_group_all

    @property
    def hdf5_positions_all(self):
        if self._hdf5_positions_all is None:
            if self.hdf5_particle_group_all is None:
                return
            self._hdf5_positions_all = self.hdf5_getter(self.hdf5_particle_group_all, 'position.value')
        return self._hdf5_positions_all

    def apply_unit(self, quantity, unit, unit_factor):

        if quantity is None:
            return
        if unit:
            unit = ureg(unit)
            unit *= unit_factor
            quantity *= unit

        return quantity

    def decode_hdf5_bytes(self, dataset, default):
        if dataset is None:
            return
        elif type(dataset).__name__ == 'ndarray':
            if dataset.size == 0:
                return default
            dataset = [val.decode("utf-8") for val in dataset] if type(dataset[0]) == bytes else dataset
            dataset = [val.__bool__() for val in dataset] if type(dataset[0]).__name__ == 'bool_' else dataset
        elif type(dataset).__name__ == 'bool_':
            dataset = dataset.__bool__()
        else:
            dataset = dataset.decode("utf-8") if type(dataset) == bytes else dataset
        return dataset

    def hdf5_attr_getter(self, source, path, attribute, default=None):
        '''
        Extracts attribute from object based on path, and returns default if not defined.
        '''
        section_segments = path.split('.')
        for section in section_segments:
            try:
                value = source.get(section)
                source = value[-1] if isinstance(value, list) else value
            except Exception:
                return
        value = source.attrs.get(attribute)
        source = value[-1] if isinstance(value, list) else value
        source = self.decode_hdf5_bytes(source, default) if source is not None else default

        return source

    def hdf5_getter(self, source, path, default=None):
        '''
        Extracts attribute from object based on path, and returns default if not defined.
        '''
        section_segments = path.split('.')
        for section in section_segments:
            try:
                value = source.get(section)
                unit = self.hdf5_attr_getter(source, section, 'unit')
                unit_factor = self.hdf5_attr_getter(source, section, 'unit_factor', default=1.0)
                source = value[-1] if isinstance(value, list) else value
            except Exception:
                return

        if source is None:
            source = default
        elif type(source) == h5py.Dataset:
            source = source[()]
            source = self.apply_unit(source, unit, unit_factor)

        return self.decode_hdf5_bytes(source, default)

    @property
    def n_frames(self):
        if self._n_frames is None:
            if not self.filehdf5:
                return
            self._n_frames = len(self.hdf5_positions_all) if self.hdf5_positions_all is not None else None
        return self._n_frames

    @property
    def n_atoms(self):
        if self._n_atoms is None:
            if not self.filehdf5:
                return
            self._n_atoms = [len(pos) for pos in self.hdf5_positions_all] if self.hdf5_positions_all is not None else None
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
            if not self.filehdf5:
                return
            self._atom_parameters = {}
            n_atoms = self.n_atoms[0]  # TODO Extend to non-static n_atoms
            if n_atoms is None:
                return
            particles = self.hdf5_getter(self.filehdf5, 'particles.all')  # TODO Extend to arbitrary particle groups

            atom_parameter_keys = ['label', 'mass', 'charge']
            for key in atom_parameter_keys:
                value = self.hdf5_getter(particles, self._nomad_to_particles_group_map[key])
                if value is not None:
                    self._atom_parameters[key] = value
                else:
                    continue
                if type(self._atom_parameters[key]) == h5py.Group:
                    self.logger.warning('Time-dependent ' + key + ' currently not supported. Atom parameter values will not be stored.')
                    continue
                elif len(self._atom_parameters[key]) != n_atoms:
                    self.logger.warning('Inconsistent length of ' + key + ' . Atom parameter values will not be stored.')
                    continue
        return self._atom_parameters

    @property
    def system_info(self):
        if self._system_info is None:
            if not self.filehdf5:
                return
            self._system_info = {'system': {}, 'calculation': {}}
            particles = self.hdf5_particle_group_all
            positions = self.hdf5_positions_all
            n_frames = self.n_frames

            if positions is None:  # For now we require that positions are present in the H5MD file to store other particle attributes
                self.logger.warning('No positions available in H5MD file. Other particle attributes will not be stored')
                return
            self._system_info['system']['positions'] = positions
            self._system_info['system']['n_atoms'] = self.n_atoms
            # get the times and steps based on the positions
            self._system_info['system']['steps'] = self.hdf5_getter(particles, 'position.step')
            self._system_info['system']['times'] = self.hdf5_getter(particles, 'position.time')

            # get the remaining system particle quantities
            system_keys = {'labels': 'system', 'velocities': 'system', 'forces': 'calculation'}
            for key, sec_key in system_keys.items():
                value = self.hdf5_getter(particles, self._nomad_to_particles_group_map[key])
                if value is not None:
                    self._system_info[sec_key][key] = value
                else:
                    continue

                if type(self._system_info[sec_key][key]) == h5py.Group:
                    self._system_info[sec_key][key] = self.hdf5_getter(self._system_info[sec_key][key], 'value')
                    if self._system_info[sec_key][key] is None:
                        continue
                    elif len(self._system_info[sec_key][key]) != n_frames:  # TODO Should really check that the stored times for these quantities are exact, not just the same length
                        self.logger.warning('Distinct trajectory lengths of particle attributes not supported. ' + key + ' values will not be stored.')
                        continue
                else:
                    self._system_info[sec_key][key] = [self._system_info[sec_key][key]] * n_frames

            # TODO Should we extend this to pick up additional attributes in the particles group? Or require that we follow the H5MD schema strictly?

            # get the system box quantities
            box = self.hdf5_getter(self.filehdf5, 'particles.all.box')
            if box is None:
                return

            box_attributes = {'dimension': 'system', 'periodic': 'system'}
            for box_key, sec_key in box_attributes.items():
                value = self.hdf5_attr_getter(particles, 'box', self._nomad_to_box_group_map[box_key])
                if value is not None:
                    self._system_info[sec_key][box_key] = [value] * n_frames

            box_keys = {'lattice_vectors': 'system'}
            for box_key, sec_key in box_keys.items():
                value = self.hdf5_getter(box, self._nomad_to_box_group_map[box_key])
                if value is not None:
                    self._system_info[sec_key][box_key] = value
                else:
                    continue

                if type(self._system_info[sec_key][box_key]) == h5py.Group:
                    self._system_info[sec_key][box_key] = self.hdf5_getter(self._system_info[sec_key][box_key], 'value')
                    if self._system_info[sec_key][box_key] is None:
                        continue
                    elif len(self._system_info[sec_key][box_key]) != n_frames:  # TODO Should really check that the stored times for these quantities are exact, not just the same length
                        self.logger.warning('Distinct trajectory lengths of box vectors and positions is not supported. ' + box_key + ' values will not be stored.')
                        continue
                else:
                    self._system_info[sec_key][box_key] = [self._system_info[sec_key][box_key]] * n_frames
        return self._system_info

    @property
    def observable_info(self):
        if self._observable_info is None:
            self._observable_info = {
                'configurational': {},
                'ensemble_average': {},
                'correlation_function': {}
            }

            def get_observable_paths(observable_group, current_path, paths):
                for obs_key in observable_group.keys():
                    path = obs_key + '.'
                    observable = self.hdf5_getter(observable_group, obs_key)
                    observable_type = self.hdf5_getter(observable_group, obs_key).attrs.get('type')
                    if not observable_type:
                        paths = get_observable_paths(observable, current_path + path, paths)
                    else:
                        paths.append(current_path + path[:-1])

                return paths

            observable_group = self.hdf5_getter(self.filehdf5, 'observables')  # TODO Extend to arbitrary particle groups
            observable_paths = get_observable_paths(observable_group, current_path='', paths=[])

            for path in observable_paths:
                observable = self.hdf5_getter(observable_group, path)
                observable_type = self.hdf5_getter(observable_group, path).attrs.get('type')
                observable_name = '-'.join(path.split('.'))
                self._observable_info[observable_type][observable_name] = {}
                for key in observable.keys():
                    observable_attribute = self.hdf5_getter(observable, key)
                    if type(observable_attribute) == h5py.Group:
                        self.logger.warning('Group structures within individual observables not supported. ' + key + ' values will not be stored.')
                        continue
                    self._observable_info[observable_type][observable_name][key] = observable_attribute
        return self._observable_info

    def get_atomsgroup_fromh5md(self, nomad_sec, h5md_sec_particlesgroup):
        for i_key, key in enumerate(h5md_sec_particlesgroup.keys()):
            particles_group = {group_key: self.hdf5_getter(h5md_sec_particlesgroup[key], group_key) for group_key in h5md_sec_particlesgroup[key].keys()}
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
                unit = None
                val = particles_group.get(particles_group_key)
                has_units = hasattr(val, 'units') if val is not None else None
                if has_units:
                    unit = val.units
                    val = val.magnitude
                sec_atomsgroup.x_h5md_parameters.append(ParamEntry(kind=particles_group_key, value=val, unit=unit))
            # get the next atomsgroup
            if particles_subgroup:
                self.get_atomsgroup_fromh5md(sec_atomsgroup, particles_subgroup)

    def check_metainfo_for_key_and_Enum(self, metainfo_class, key, val):
        if key in metainfo_class.__dict__.keys():
            quant = metainfo_class.__dict__.get(key)
            if quant.get('type') is not None:
                if type(quant.type).__name__ == 'MEnum':
                    if val in quant.type._list:
                        return key
                    else:
                        return 'x_h5md_' + key
                else:
                    return key
            else:
                self.logger.warning(key + 'in ' + metainfo_class + ' is not a Quantity or does not have an associated type.')  # Not sure if this can ever happen
                return key
        else:
            return 'x_h5md_' + key

    @property
    def parameter_info(self):
        if self._parameter_info is None:
            self._parameter_info = {
                'force_calculations': {},
                'workflow': {}
            }

            def get_parameters_recursive(parameter_group):
                param_dict = {}
                for key, val in parameter_group.items():
                    if type(val) == h5py.Group:
                        param_dict[key] = get_parameters_recursive(val)
                    else:
                        param_dict[key] = self.hdf5_getter(parameter_group, key)
                        if type(param_dict[key]) == str:
                            param_dict[key] = param_dict[key].lower()
                        elif 'int' in type(param_dict[key]).__name__:
                            param_dict[key] = param_dict[key].item()

                return param_dict

            parameter_group = self.hdf5_getter(self.filehdf5, 'parameters')
            force_calculations_group = self.hdf5_getter(parameter_group, 'force_calculations')
            if force_calculations_group is not None:
                self._parameter_info['force_calculations'] = get_parameters_recursive(force_calculations_group)
            workflow_group = self.hdf5_getter(parameter_group, 'workflow')
            if workflow_group is not None:
                self._parameter_info['workflow'] = get_parameters_recursive(workflow_group)

        return self._parameter_info

    def parse_calculation(self):
        print('in parse calc')
        sec_run = self.archive.run[-1]
        sec_system = sec_run.system
        calculation_info = self.observable_info.get('configurational')
        system_info = self._system_info.get('calculation')  # note: it is currently ensured in parse_system() that these have the same length as the system_map
        if not calculation_info:  # TODO should still create entries for system time link in this case
            return
        time_step = None  # TODO GET TIME STEP FROM PARAMS SECTION

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

        for key, observable in calculation_info.items():
            if system_map_key == 'time':
                times = observable.get('time')
                if times is not None:
                    times = np.around(times.magnitude * ureg.convert(1.0, times.units, ureg.picosecond), 5)  # TODO What happens if no units are given?
                    for i_time, time in enumerate(times):
                        map_entry = system_map.get(time)
                        if map_entry:
                            map_entry[key] = i_time
                        else:
                            system_map[time] = {key: i_time}
                else:
                    self.logger.warning('No time information available for ' + observable + '. Cannot store values.')
            elif system_map_key == 'step':
                steps = observable.get('step')
                if steps:
                    steps = np.around(steps)
                    for i_step, step in enumerate(steps):
                        map_entry = system_map.get(step)
                        if map_entry:
                            map_entry[key] = i_step
                        else:
                            system_map[time] = {key: i_step}
            else:
                self.logger.error('system_map_key not assigned correctly.')

        for frame in sorted(system_map):
            print(frame)
            sec_scc = sec_run.m_create(Calculation)
            sec_scc.method_ref = sec_run.method[-1] if sec_run.method else None
            if system_map_key == 'time':
                sec_scc.time = frame
                if time_step:
                    sec_scc.step = int((frame / time_step).magnitude)
            elif system_map_key == 'step':
                sec_scc.step = frame
                if time_step:
                    sec_scc.time = sec_scc.step * time_step

            system_index = system_map[frame]['system']
            print(system_index)
            if system_index is not None:
                print(system_info.items())
                for key, val in system_info.items():
                    print(key)
                    if key == 'forces':
                        sec_scc.forces = Forces(total=ForcesEntry(value=val[system_index]))
                    else:
                        print(BaseCalculation.__dict__.keys())
                        print(key)
                        if key in BaseCalculation.__dict__.keys():
                            print(key)
                            # setattr(sec_scc, key, val)
                            sec_scc.m_set(sec_scc.m_get_quantity_definition(key), val[system_index])
                        else:
                            # setattr(sec_scc, 'x_h5md_' + key, val)
                            unit = None
                            if hasattr(val, 'units'):
                                unit = val.units
                                val = val.magnitude
                            sec_scc.x_h5md_custom_calculations.append(CalcEntry(kind=key, value=val, unit=unit))

                sec_scc.system_ref = sec_system[system_index]

            sec_energy = sec_scc.m_create(Energy)
            for key, observable in calculation_info.items():
                obs_index = system_map[frame].get(key)
                if obs_index:
                    val = observable.get('value', [None] * (obs_index + 1))[obs_index]
                    obs_name_short = key.split('-')[-1]
                    if 'energ' in key:  # TODO check for energies or energy when matching name
                        if obs_name_short in Energy.__dict__.keys():
                            # setattr(sec_energy, obs_name_short, EnergyEntry(value=val))
                            sec_energy.m_add_sub_section(getattr(Energy, obs_name_short), EnergyEntry(value=val))
                        else:
                            # setattr(sec_energy, 'x_h5md_' + key, EnergyEntry(value=val))
                            sec_energy.x_h5md_energy_contributions.append(
                                EnergyEntry(kind=key, value=val))
                    else:
                        if obs_name_short in BaseCalculation.__dict__.keys():
                            print(obs_name_short)
                            # setattr(sec_scc, obs_name_short, val)
                            sec_scc.m_set(sec_scc.m_get_quantity_definition(obs_name_short), val)
                        else:
                            # setattr(sec_scc, 'x_h5md_' + key, val)
                            unit = None
                            if hasattr(val, 'units'):
                                unit = val.units
                                val = val.magnitude
                            sec_scc.x_h5md_custom_calculations.append(CalcEntry(kind=key, value=val, unit=unit))

    def parse_system(self):
        sec_run = self.archive.run[-1]

        system_info = self.system_info.get('system')
        if not system_info:
            self.logger.error('No particle information found in H5MD file.')

        n_frames = len(system_info.get('times', []))
        self._system_time_map = {}
        self._system_step_map = {}
        for frame in range(n_frames):
            # if (n % self.frame_rate) > 0:
            #     continue
            sec_system = sec_run.m_create(System)
            sec_atoms = sec_system.m_create(Atoms)

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
                    print(key)
                    setattr(sec_atoms, key, system_info.get(key, [None] * (frame + 1))[frame])

            if frame == 0:  # TODO extend to time-dependent topologies
                connectivity = self.hdf5_getter(self.filehdf5, 'connectivity', None)
                bond_list = self.hdf5_getter(connectivity, 'bonds')
                if bond_list is not None:
                    setattr(sec_atoms, 'bond_list', bond_list)
                topology = self.hdf5_getter(connectivity, 'particles_group', None)
                if topology:
                    self.get_atomsgroup_fromh5md(sec_system, topology)

    def parse_method(self):

        sec_method = self.archive.run[-1].m_create(Method)
        sec_force_field = sec_method.m_create(ForceField)
        sec_model = sec_force_field.m_create(Model)

        # get the atom parameters
        n_atoms = self.n_atoms[0]  # TODO Extend to non-static n_atoms
        for n in range(n_atoms):
            sec_atom = sec_method.m_create(AtomParameters)
            for key in self.atom_parameters.keys():
                setattr(sec_atom, key, self.atom_parameters[key][n])

        # Get the interactions
        connectivity = self.hdf5_getter(self.filehdf5, 'connectivity', None)
        if not connectivity:
            return

        atom_labels = self.atom_parameters.get('label')
        interaction_keys = ['bonds', 'angles', 'dihedrals', 'impropers']
        for interaction_key in interaction_keys:
            interaction_list = self.hdf5_getter(connectivity, interaction_key)
            if interaction_list is None:
                continue
            elif type(interaction_list) == h5py.Group:
                self.logger.warning('Time-dependent ' + key + ' currently not supported. ' + key + ' list will not be stored')
                continue
            sec_interaction = sec_model.m_create(Interaction)
            sec_interaction.type = interaction_key
            sec_interaction.n_inter = len(interaction_list)
            sec_interaction.n_atoms = len(interaction_list[0])
            sec_interaction.atom_indices = interaction_list
            sec_interaction.atom_labels = np.array(atom_labels)[interaction_list] if atom_labels is not None else None

        # Get the force calculation parameters
        force_calculation_parameters = self.parameter_info.get('force_calculations')
        if force_calculation_parameters is None:
            return

        sec_force_calculations = sec_force_field.m_create(ForceCalculations)
        sec_neighbor_searching = sec_force_calculations.m_create(NeighborSearching)

        for key, val in force_calculation_parameters.items():
            if type(val) is not dict:
                key = self.check_metainfo_for_key_and_Enum(ForceCalculations, key, val)
                setattr(sec_force_calculations, key, val)
            else:
                if key == 'neighbor_searching':
                    for neigh_key, neigh_val in val.items():
                        neigh_key = self.check_metainfo_for_key_and_Enum(NeighborSearching, neigh_key, neigh_val)
                        setattr(sec_neighbor_searching, neigh_key, neigh_val)
                else:
                    self.logger.warning(key + 'is not a valid force calculations section. Corresponding parameters will not be stored.')

    def parse_workflow(self):

        workflow_parameters = self.parameter_info.get('workflow').get('molecular_dynamics')
        if workflow_parameters is None:
            return

        workflow = MolecularDynamics(
            method=MolecularDynamicsMethod(
                thermostat_parameters=ThermostatParameters(),
                barostat_parameters=BarostatParameters()
            ), results=MolecularDynamicsResults()
        )

        for key, val in workflow_parameters.items():
            if type(val) is not dict:
                if key == 'thermodynamic_ensemble':
                    val = val.upper()
                key = self.check_metainfo_for_key_and_Enum(MolecularDynamicsMethod, key, val)
                setattr(workflow.method, key, val)
            else:
                if key == 'thermostat_parameters':
                    for thermo_key, thermo_val in val.items():
                        thermo_key = self.check_metainfo_for_key_and_Enum(ThermostatParameters, thermo_key, thermo_val)
                        setattr(workflow.method.thermostat_parameters, thermo_key, thermo_val)
                elif key == 'barostat_parameters':
                    for baro_key, baro_val in val.items():
                        baro_key = self.check_metainfo_for_key_and_Enum(BarostatParameters, baro_key, baro_val)
                        setattr(workflow.method.barostat_parameters, baro_key, baro_val)
                else:
                    self.logger.warning(key + 'is not a valid molecular dynamics workflow section. Corresponding parameters will not be stored.')

        self.archive.workflow2 = workflow

    def init_parser(self):

        self.mainfile = self.filepath

    def parse(self, filepath, archive, logger):

        self.filepath = os.path.abspath(filepath)
        self.archive = archive
        self.logger = logging.getLogger(__name__) if logger is None else logger
        self._maindir = os.path.dirname(self.filepath)
        self._hoomdblue_files = os.listdir(self._maindir)
        self._basename = os.path.basename(filepath).rsplit('.', 1)[0]

        self.init_parser()

        if self.filehdf5 is None:
            return

        sec_run = self.archive.m_create(Run)

        program_name = self.hdf5_attr_getter(self.filehdf5, 'h5md.program', 'name', None)
        version = self.hdf5_attr_getter(self.filehdf5, 'h5md.program', 'version', None)
        program_version = '.'.join([str(i) for i in version]) if version is not None else None
        sec_run.program = Program(name=program_name, version=program_version)
        #  TODO get the remaining information from the h5md root level

        self.parse_method()

        self.parse_system()

        self.parse_calculation()

        self.parse_workflow()
