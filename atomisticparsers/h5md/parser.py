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
    Method, ForceField, Model, Interaction, AtomParameters
)
from nomad.datamodel.metainfo.simulation.system import (
    System, Atoms, AtomsGroup
)
from nomad.datamodel.metainfo.simulation.calculation import (
    Calculation, Energy, EnergyEntry, Forces, ForcesEntry
)
from nomad.datamodel.metainfo.workflow import (
    Workflow, MolecularDynamics
)
from nomad.datamodel.metainfo.simulation import workflow as workflow2
from nomad.atomutils import get_molecules_from_bond_list, is_same_molecule, get_composition
from nomad.units import ureg

MOL = 6.022140857e+23


class H5MDParser(FileParser):
    def __init__(self):
        super().__init__(None)

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


class H5MDParser:
    def __init__(self):
        self._frame_rate = None
        # max cumulative number of atoms for all parsed trajectories to calculate sampling rate
        self._cum_max_atoms = 2500000

        self._nomad_to_particles_group_map = {
            'positions': 'position',
            'velocities': 'velocity',
            'forces': 'force',
            'species': 'species',
            'labels': 'label',
            'label': 'label',
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
                self._file_handler = h5py.File('self.mainfile', 'r')
            except Exception:
                self.logger.error('Error reading hdf5(h5MD) file.')

        return self._file_handler

    def apply_unit(quantity, unit, unit_factor):

        if quantity is None:
            return
        if unit:
            unit = ureg(unit)
            unit *= unit_factor
            quantity *= unit

        return quantity

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
        source = source if source is not None else default

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

        return source

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

    @property
    def atom_parameters(self):
        self.atom_parameters = {}
        n_atoms = self.system_info.get('system').get('n_atoms')
        if n_atoms is None:
            return
        particles = self.hdf5_getter(self.filehdf5, 'particles.all')  # TODO Extend to arbitrary particle groups

        atom_parameter_keys = ['label', 'mass', 'charge']
        for key in atom_parameter_keys:
            self.atom_parameters[key] = self.hdf5_getter(particles, self._nomad_to_particles_group_map[key])
            if not self.atom_parameters[key]:
                continue

            if type(self.atom_parameters[key]) == h5py.Group:
                self.logger.warning('Time-dependent ' + key + ' currently not supported. Atom parameter values will not be stored.')
                continue
            elif len(self.atom_parameters[key]) != n_atoms:
                self.logger.warning('Inconsistent length of ' + key + ' . Atom parameter values will not be stored.')
                continue

    @property
    def system_info(self):
        self.system_info = {'system': {}, 'calculation': {}}
        particles = self.hdf5_getter(self.filehdf5, 'particles.all')  # TODO Extend to arbitrary particle groups

        self.system_info['system']['positions'] = self.hdf5_getter(particles, 'position.value')
        if not self.system_info['system']['positions']:  # For now we require that positions are present in the H5MD file to store other particle attributes
            self.logger.warning('No positions available in H5MD file. Other particle attributes will not be stored')
            return
        n_frames = len(self.system_info['system']['positions'])
        self.system_info['system']['n_atoms'] = len(self.system_info['system']['positions'][0])

        # get the times and steps based on the positions
        self.system_info['system']['steps'] = self.hdf5_getter(particles, 'position.step')
        self.system_info['system']['times'] = self.hdf5_getter(particles, 'position.time')

        # get the remaining system particle quantities
        system_keys = {'species': 'system', 'labels': 'system', 'velocities': 'system', 'forces': 'calculation'}
        for key, sec_key in system_keys.items():
            self.system_info[sec_key][key] = self.hdf5_getter(particles, self._nomad_to_particles_group_map[key])
            if not self.system_info[sec_key][key]:
                continue

            if type(self.system_info[sec_key][key]) == h5py.Group:
                self.system_info[sec_key][key] = self.hdf5_getter(self.system_info[sec_key][key], 'value')
                if self.system_info[sec_key][key] is None:
                    continue
                elif len(self.system_info[sec_key][key]) != n_frames:  # TODO Should really check that the stored times for these quantities are exact, not just the same length
                    self.logger.warning('Distinct trajectory lengths of particle attributes not supported. ' + key + ' values will not be stored.')
                    continue
            else:
                self.system_info[sec_key][key] = [self.system_info[sec_key][key]] * n_frames

        # TODO Should we extend this to pick up additional attributes in the particles group? Or require that we follow the H5MD schema strictly?

        # get the system box quantities
        box = self.hdf5_getter(self.filehdf5, 'particles.all.box')
        if not box:
            return

        box_attributes = {'dimension': 'system', 'periodic': 'system'}
        for box_key, sec_key in box_attributes.items():
            self.system_info[sec_key][box_key] = self.hdf5_attr_getter(self.filehdf5, 'box', self._nomad_to_box_group_map[box_key])
            self.system_info[sec_key][box_key] = [self.system_info[sec_key][box_key]] * n_frames

        box_keys = {'lattice_vectors': 'system'}
        for box_key, sec_key in box_keys.items():
            self.system_info[sec_key][box_key] = self.hdf5_getter(particles, self._nomad_to_box_group_map[box_key])
            if not self.system_info[sec_key][key]:
                continue

            if type(self.system_info[sec_key][key]) == h5py.Group:
                self.system_info[sec_key][key] = self.hdf5_getter(self.system_info[sec_key][key], 'value')
                if self.system_info[sec_key][key] is None:
                    continue
                elif len(self.system_info[sec_key][key]) != n_frames:  # TODO Should really check that the stored times for these quantities are exact, not just the same length
                    self.logger.warning('Distinct trajectory lengths of box vectors and positions is not supported. ' + key + ' values will not be stored.')
                    continue
            else:
                self.system_info[sec_key][key] = [self.system_info[sec_key][key]] * n_frames

    @property
    def obervable_info(self):
        self.obervable_info = {
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
            self.obervable_info[observable_type][observable_name] = {}
            for key in observable.keys():
                observable_attribute = self.hdf5_getter(observable, key)
                if type(observable_attribute) == h5py.Group:
                    self.logger.warning('Group structures within individual observables not supported. ' + key + ' values will not be stored.')
                    continue
                self.obervable_info[observable_type][observable_name][key] = observable_attribute

    @property
    def _nomad_to_observable_info_map(self):
        self._nomad_to_observable_info_map = {}


    def get_atomsgroup_fromh5md(self, nomad_sec, h5md_sec_particlesgroup):
        for i_key, key in enumerate(h5md_sec_particlesgroup.keys()):
            particles_group = dict(h5md_sec_particlesgroup[key])
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
                setattr(sec_atomsgroup, 'x_h5md_' + particles_group_key, particles_group.pop(particles_group_key, None))
            # get the next atomsgroup
            if particles_subgroup:
                self.get_atomsgroup_fromh5md(sec_atomsgroup, particles_subgroup)

    def parse_calculation(self):
        sec_run = self.archive.run[-1]
        sec_system = sec_run.system
        calculation_info = self.obervable_info.get('configurational')
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
                if times:
                    times = np.around(times.magnitude * ureg.convert(1.0, times.units, ureg.picosecond), 5)
                    for i_time, time in enumerate(times):
                        map_entry = system_map.get(time)
                        if map_entry:
                            map_entry[observable] = i_time
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
                            map_entry[observable] = i_step
                        else:
                            system_map[time] = {key: i_step}
            else:
                self.logger.error('system_map_key not assigned correctly.')

        for frame in sorted(system_map):
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

            system_index = system_map[frame]['system_index']
            if system_index is not None:
                sec_scc.forces = Forces(total=ForcesEntry(value=self.traj_parser.get_forces(system_index)))
                sec_scc.system_ref = sec_system[system_index]

            sec_energy = sec_scc.m_create(Energy)
            for key, observable in calculation_info.items():
                obs_index = system_map[frame].get(key)
                if obs_index:
                    value = observable.get('value', [None] * (obs_index + 1))[obs_index]
                    setattr(sec_scc, 'x_h5md_' + key, value)


            calculation_index = time_map[time]['calculation_index']
            if calculation_index is not None:
                # TODO add other energy contributions, properties
                energy_keys = ['LJ (SR)', 'Coulomb (SR)', 'Potential', 'Kinetic En.']

                sec_energy = sec_scc.m_create(Energy)
                for key in thermo_data.keys():
                    val = thermo_data.get(key)[calculation_index]
                    if val is None:
                        continue

                    if key == 'Total Energy':
                        sec_energy.total = EnergyEntry(value=val)
                    elif key == 'Potential':
                        sec_energy.potential = EnergyEntry(value=val)
                    elif key == 'Kinetic En.':
                        sec_energy.kinetic = EnergyEntry(value=val)
                    elif key == 'Coulomb (SR)':
                        sec_energy.coulomb = EnergyEntry(value=val)
                    elif key == 'Pressure':
                        sec_scc.pressure = val
                    elif key == 'Temperature':
                        sec_scc.temperature = val
                    if key in energy_keys:
                        sec_energy.contributions.append(
                            EnergyEntry(kind=self._metainfo_mapping[key], value=val))

    def parse_calculation_OLD(self):
        sec_run = self.archive.run[-1]
        sec_system = sec_run.system
        calculation_info = self.obervable_info.get('configurational')
        if not calculation_info:  # TODO should still create entries for system time link in this case
            return

        time_step = None  # TODO GET TIME STEP FROM PARAMS SECTION
        calculation_times_ps = []
        calculation_steps = []
        for key, observable in calculation_info.items():
            times = observable.get('time')
            if times is not None:
                calculation_times_ps.append(times.magnitude * ureg.convert(1.0, times.units, ureg.picosecond))
            steps = observable.get('step')
            if steps is not None:
                calculation_steps.append(steps)
        calculation_times_ps = np.around(np.unique(np.concatenate(calculation_times_ps)), 5) if calculation_times_ps else None
        calculation_steps = np.around(np.unique(np.concatenate(calculation_steps))).astype(int) if calculation_steps else None

        system_map = {}
        system_map_key = ''
        if self._system_time_map and calculation_times_ps:
            system_map_key = 'time'
            for i_calc, calculation_time in enumerate(calculation_times_ps):
                system_index = self._system_time_map.pop(
                    calculation_times_ps[i_calc], None) if calculation_times_ps[i_calc] is not None else None
                system_map[calculation_time] = {'system_index': system_index, 'calculation_index': i_calc}
            for time, i_sys in self._system_time_map.items():
                system_map[time] = {'system_index': i_sys, 'calculation_index': None}
        elif self._system_step_map and calculation_steps:
            system_map_key = 'step'
            for i_calc, calculation_step in enumerate(calculation_steps):
                system_index = self._system_step_map.pop(
                    calculation_steps[i_calc], None) if calculation_steps[i_calc] is not None else None
                system_map[calculation_step] = {'system_index': system_index, 'calculation_index': i_calc}
            for step, i_sys in self._system_step_map.items():
                system_map[step] = {'system_index': i_sys, 'calculation_index': None}
        else:
            self.logger.warning('No step or time available for system data. Cannot make calculation to system references.')
            if calculation_times_ps:
                system_map_key = 'time'
                for i_calc, calculation_time in enumerate(calculation_times_ps):
                    system_map[calculation_time] = {'system_index': None, 'calculation_index': i_calc}
            elif calculation_steps:
                system_map_key = 'step'
                for i_calc, calculation_step in enumerate(calculation_steps):
                    system_map[calculation_step] = {'system_index': None, 'calculation_index': i_calc}
            else:
                self.logger.warning('No step or time available for system or calculation data!')
                return

        # Add the observables to the system_map
        observables = self.obervable_info.get('configurational')
        for observable in observables.keys():
            if system_map_key == 'time':
                times = observable.get('time')
                if times:
                    for i_time, time in enumerate(times):
                        time_ps = round(time.magnitude * ureg.convert(1.0, time.units, ureg.picosecond), 5)
                        map_entry = system_map.get(time_ps)
                        if map_entry:
                            map_entry[observable] = i_time
                        else:
                            system_map[time_ps] = {'system_index': None, 'calculation_index': None, ''}
                else:
                    self.logger.warning('No time information available for ' + observable + '. Cannot store values.')
            elif system_map_key == 'step':
                steps = observable.get('step')
                if steps:
                    steps =
            else:
                self.logger.error('system_map_key not assigned correctly.')

        for frame in sorted(system_map):
            sec_scc = sec_run.m_create(Calculation)
            sec_scc.method_ref = sec_run.method[-1] if sec_run.method else None
            if system_map_key == 'time':
                sec_scc.time = frame
                if calculation_steps:
                    sec_scc.step = calculation_steps[system_map[frame]['calculation_index']]
                elif time_step:
                    sec_scc.step = int((frame / time_step).magnitude)
            elif system_map_key == 'step':
                sec_scc.step = frame
                if calculation_times_ps:
                    sec_scc.time = calculation_times_ps[system_map[frame]['calculation_index']]
                elif time_step:
                    sec_scc.time = sec_scc.step * time_step

            system_index = system_map[frame]['system_index']
            if system_index is not None:
                sec_scc.forces = Forces(total=ForcesEntry(value=self.traj_parser.get_forces(system_index)))
                sec_scc.system_ref = sec_system[system_index]

            calculation_index = time_map[time]['calculation_index']
            if calculation_index is not None:
                # TODO add other energy contributions, properties
                energy_keys = ['LJ (SR)', 'Coulomb (SR)', 'Potential', 'Kinetic En.']

                sec_energy = sec_scc.m_create(Energy)
                for key in thermo_data.keys():
                    val = thermo_data.get(key)[calculation_index]
                    if val is None:
                        continue

                    if key == 'Total Energy':
                        sec_energy.total = EnergyEntry(value=val)
                    elif key == 'Potential':
                        sec_energy.potential = EnergyEntry(value=val)
                    elif key == 'Kinetic En.':
                        sec_energy.kinetic = EnergyEntry(value=val)
                    elif key == 'Coulomb (SR)':
                        sec_energy.coulomb = EnergyEntry(value=val)
                    elif key == 'Pressure':
                        sec_scc.pressure = val
                    elif key == 'Temperature':
                        sec_scc.temperature = val
                    if key in energy_keys:
                        sec_energy.contributions.append(
                            EnergyEntry(kind=self._metainfo_mapping[key], value=val))

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
                    setattr(sec_atoms, key, system_info.get(key, [None] * (frame + 1))[frame])

            if frame == 0:  # TODO extend to time-dependent topologies
                topology = self.hdf5_getter(self.filehdf5, 'connectivity.topology', None)
                if topology:
                    self.get_atomsgroup_fromh5md(sec_system, topology)

    def parse_method(self):

        sec_method = self.archive.run[-1].m_create(Method)
        sec_force_field = sec_method.m_create(ForceField)
        sec_model = sec_force_field.m_create(Model)

        n_atoms = self.system_info.get('system').get('n_atoms')

        # get the atom parameters
        for n in range(n_atoms):
            sec_atom = sec_method.m_create(AtomParameters)
            for key in self.atom_parameters.keys():
                setattr(sec_atom, key, self.atom_parameters[key][n])

        # Get the interactions
        connectivity = self.hdf5_getter(self.filehdf5, 'connectivity', None)
        if not connectivity:
            return
        atom_types = np.array(getattr(particles, 'types', []))
        atom_typeid = np.array(getattr(particles, 'typeid', []))
        atom_labels = np.array(atom_types)[atom_typeid]

        interaction_keys = ['bonds', 'angles', 'dihedrals', 'impropers']
        for interaction_key in interaction_keys:
            interaction_list = self.hdf5_getter(connectivity, interaction_key)
            if not interaction_list:
                continue
            elif type(interaction_list) == h5py.Group:
                self.logger.warning('Time-dependent ' + key + ' currently not supported. ' + key + ' list will not be stored')
                continue
            sec_interaction = sec_model.m_create(Interaction)
            sec_interaction.type = interaction_key
            sec_interaction.n_inter = len(sec_interaction)
            sec_interaction.n_atoms = len(sec_interaction[0])
            sec_interaction.atom_indices = interaction_list
            sec_interaction.atom_labels = atom_labels[sec_interaction.atom_indices]

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

        if self.filehdf5 is None:
            return

        sec_run = self.archive.m_create(Run)

        program_name = self.hdf5_attr_getter(self.filehdf5, 'h5md.program', 'name', None)
        program_version = self.hdf5_attr_getter(self.filehdf5, 'h5md.program', 'version', None)
        sec_run.program = Program(name=program_name, version=program_version)
        #  TODO get the remaining information from the h5md root level

        self.parse_method()

        self.parse_system()

        self.parse_calculation()

        self.parse_workflow()
