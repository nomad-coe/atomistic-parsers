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
import re
import logging
import json
from datetime import datetime
from ase.cell import Cell
from ase.atoms import Atoms as aseatoms
from ase.spacegroup import Spacegroup
import numpy as np

from nomad.units import ureg
from nomad.datamodel import EntryArchive
from nomad.client import api

from nomad.datamodel.metainfo.simulation.run import Run, Program
from nomad.datamodel.metainfo.simulation.system import System, Atoms
from nomad.datamodel.metainfo.simulation.method import Method, ForceField, Model
from nomad.datamodel.metainfo.simulation.calculation import (
    BandEnergies, BandStructure, Calculation, Energy, EnergyEntry, Stress, StressEntry)
from nomad.datamodel.metainfo.workflow import Phonon, Workflow, Elastic, Interface
from .metainfo import openkim  # pylint: disable=unused-import


class Converter:
    def __init__(self, entries, logger=None):
        self.entries = entries
        self.logger = logger if logger is not None else logging.getLogger('__name__')
        self.material = dict()

    @property
    def entries(self):
        return self._entries

    @entries.setter
    def entries(self, value):
        self._entries = value
        self.material = dict()
        self.archive = EntryArchive()

    def calculate_nomad_error(self, property_name, transform_function=None):
        property_map = {
            # TODO add other properties: phonon
            # challenge for phonon is alligning phonon frequency array or extracting
            # frequencies at high-symm k-points
            'workflow[-1].elastic.elastic_constants_matrix_second_order': 'mechanical',
            'run[-1].calculation[-1].band_structure_phonon[-1].segment[*].energies': 'vibrational',
        }
        if property_name not in property_map:
            return

        try:
            elements = list(set(self.archive.run[-1].system[-1].atoms.labels))
        except Exception:
            return

        if transform_function is None:
            transform_function = lambda archive, property_name: np.array(archive.m_xpath(property_name)).flatten()

        page_after_value = None
        nomad_data = []
        query = {
            'results.material.elements': elements,
            'results.material.n_elements': len(elements),
            'results.properties.available_properties': property_map.get(property_name),
        }
        if self.material.get('space_group_number') is not None:
            query['results.material.symmetry.space_group_number'] = self.material.get('space_group_number')
        elif self.material.get('structure_name') is not None:
            query['results.material.symmetry.structure_name'] = self.material.get('structure_name')
        while True:
            response = api.post('entries/archive/query', data=json.dumps({
                'owner': 'public',
                'query': query,
                'pagination': {
                    'page_size': 5,
                    'page_after_value': page_after_value
                },
                'required': {
                    'run': {
                        'calculation[-1]': '*',
                        'system[-1]': {
                            'atoms': '*'
                        }
                    },
                    'workflow': '*'
                }
            }))
            assert response.status_code == 200
            result = response.json()
            nomad_data.extend([data['archive'] for data in result['data']])
            page_after_value = result['pagination'].get('next_page_after_value')

            if len(result['data']) < 5 or page_after_value is None:
                break

        if len(nomad_data) == 0:
            return

        try:
            nomad_data = np.array([transform_function(EntryArchive().m_from_dict(data), property_name) for data in nomad_data])
            openkim_data = np.array(transform_function(self.archive, property_name))
        except Exception:
            self.logger.error('Error transforming data.')

        if np.shape(nomad_data[0]) != np.shape(openkim_data):
            self.logger.error('Incompatible shape of openkim data and nomad.')
            return

        # calculate rms error of openkim data wrt nomad
        sec_workflow = self.archive.workflow[-1]
        sec_workflow.x_openkim_property = property_name
        sec_workflow.x_openkim_n_nomad_data = len(nomad_data)
        sec_workflow.x_openkim_nomad_std = np.average(np.std(nomad_data, axis=0))
        sec_workflow.x_openkim_nomad_rms_error = np.sqrt(np.average((nomad_data - openkim_data)**2))
        # TODO add references to the nomad calculations / workflows

    def convert(self, filename='openkim_archive.json'):
        def get_value(entry, key, array=False, default=None):
            val = entry.get(key, [] if array else default)
            return [val] if array and not isinstance(val, list) else val

        def symmetrize_matrix(matrix):
            matrix = np.array(matrix)
            return matrix + matrix.T - np.diag(matrix.diagonal())

        def get_atoms(entry):
            symbols = entry.get('species.source-value', [])
            basis = entry.get('basis-atom-coordinates.source-value', [[0., 0., 0.]])
            cellpar = []
            for x in ['a', 'b', 'c']:
                value = entry.get(f'{x}.si-value', cellpar[0] if cellpar else 1)
                cellpar.append([value] if not isinstance(value, list) else value)
            cellpar = (cellpar * ureg.m).to('angstrom').magnitude

            # TODO are angles denoted by alpha, beta, gamma in openkim? can they be lists?
            alpha = entry.get('alpha.source-value', 90)
            beta = entry.get('beta.source-value', 90)
            gamma = entry.get('gamma.source-value', 90)

            atoms = []
            for n in range(len(cellpar[0])):
                try:
                    cell = Cell.fromcellpar([cellpar[0][n], cellpar[1][n], cellpar[2][n], alpha, beta, gamma])
                    atom = aseatoms(scaled_positions=basis, cell=cell, pbc=True)
                    if len(symbols) == len(atom.numbers):
                        atom.symbols = symbols
                    else:
                        atom.symbols = ['X' for _ in atom.numbers] if len(symbols) == 0 else symbols
                    atoms.append(atom)
                except Exception:
                    self.logger.error('Error generating structure.')
            return atoms

        def phonon_transform_function(archive, property_name, **kwargs):
            # Transform phonon frequencies to select data only for special kpoints
            data = np.array(archive.m_xpath(property_name))
            kpoints_special = kwargs.get('kpoints')
            if kpoints_special is None:
                cell = Cell(archive.run[-1].system[-1].atoms.lattice_vectors.to('angstrom').magnitude)
                kpoints_special = cell.get_bravais_lattice().get_special_points_array()

            kpoints = np.array(archive.m_xpath('%s.kpoints' % property_name.rsplit('.', 1)[0]))

            data_special = []
            for kpoint in kpoints_special:
                index = np.where((kpoints[:, :, 0] == kpoint[0]) & (kpoints[:, :, 1] == kpoint[1]) & (kpoints[:, :, 2] == kpoint[2]))
                if np.size(index) == 0:
                    continue
                data_special.append(data[index[0][0], :, index[0][1], :])

            return np.array(data_special).flatten()

        # first entry is the parser-generated header used identify an  open-kim
        for entry in self.entries:
            sec_run = self.archive.m_create(Run)
            sec_run.program = Program(name='OpenKIM', version=entry.get('meta.runner.short-id'))

            compile_date = entry.get('meta.created_on')
            if compile_date is not None:
                dt = datetime.strptime(compile_date, '%Y-%m-%d %H:%M:%S.%f') - datetime(1970, 1, 1)
                sec_run.program.compilation_datetime = dt.total_seconds()

            # openkim metadata
            sec_run.x_openkim_meta = {key: entry.pop(key) for key in list(entry.keys()) if key.startswith('meta.')}

            atoms = get_atoms(entry)
            for atom in atoms:
                sec_system = sec_run.m_create(System)
                sec_atoms = sec_system.m_create(Atoms)
                sec_atoms.labels = atom.get_chemical_symbols()
                sec_atoms.positions = atom.get_positions() * ureg.angstrom
                sec_atoms.lattice_vectors = atom.get_cell().array * ureg.angstrom
                sec_atoms.periodic = [True, True, True]
            try:
                self.material['space_group_number'] = Spacegroup(entry['space-group.source-value']).no
                self.material['structure_name'] = entry['short-name.source-value'][-1]
            except Exception:
                pass

            # model parameters
            model = sec_run.x_openkim_meta.get('meta.model')
            if model is not None:
                sec_method = sec_run.m_create(Method)
                sec_method.force_field = ForceField(model=[Model(
                    name=model,
                    reference='https://openkim.org/id/%s' % model)])

            energies = get_value(entry, 'cohesive-potential-energy.si-value', True)
            for n, energy in enumerate(energies):
                sec_scc = sec_run.m_create(Calculation)
                sec_scc.energy = Energy(total=EnergyEntry(value=energy))

            temperatures = get_value(entry, 'temperature.si-value', True)
            for n, temperature in enumerate(temperatures):
                sec_scc = sec_run.calculation[n] if sec_run.calculation else sec_run.m_create(Calculation)
                sec_scc.temperature = temperature

            stress = get_value(entry, 'cauchy-stress.si-value')
            if stress is not None:
                sec_scc = sec_run.calculation[-1] if sec_run.calculation else sec_run.m_create(Calculation)
                stress_tensor = np.zeros((3, 3))
                stress_tensor[0][0] = stress[0]
                stress_tensor[1][1] = stress[1]
                stress_tensor[2][2] = stress[2]
                stress_tensor[1][2] = stress_tensor[2][1] = stress[3]
                stress_tensor[0][2] = stress_tensor[2][0] = stress[4]
                stress_tensor[0][1] = stress_tensor[1][0] = stress[5]
                sec_scc.stress = Stress(total=StressEntry(value=stress_tensor))
                # TODO add nomad error, which property corresponding to stress

            for key, val in entry.items():
                key = 'x_openkim_%s' % re.sub(r'\W', '_', key)
                try:
                    setattr(sec_run, key, val)
                except Exception:
                    pass

            # workflow
            property_id = entry.get('property-id', '')
            # elastic constants
            if 'elastic-constants' in property_id:
                sec_workflow = self.archive.m_create(Workflow)
                sec_workflow.type = 'elastic'
                sec_elastic = sec_workflow.m_create(Elastic)
                cij = [[get_value(entry, f'c{i}{j}.si-value', default=0) for j in range(1, 7)] for i in range(1, 7)]
                sg = self.material.get('space_group_number', 0)
                if sg <= 74:
                    # triclinic / monoclinic / orthorhombic
                    pass
                elif sg <= 142:  # tetragonal
                    cij[1][1] = cij[0][0]
                    cij[1][2] = cij[0][2]
                    cij[1][5] = -cij[0][5]
                    cij[4][4] = cij[3][3]
                elif sg <= 167:  # trigonal
                    cij[1][1] = cij[0][0]
                    cij[1][2] = cij[0][2]
                    cij[1][3] = -cij[0][3]
                    cij[1][4] = -cij[0][4]
                    cij[3][5] = -cij[0][4]
                    cij[4][5] = cij[0][3]
                    cij[5][5] = (cij[0][0] - cij[0][1]) / 2
                elif sg <= 194:  # hexagonal
                    cij[1][1] = cij[0][0]
                    cij[1][2] = cij[0][2]
                    cij[4][4] = cij[3][3]
                    cij[5][5] = (cij[0][0] - cij[0][1]) / 2
                elif sg <= 230:  # cubic
                    cij[0][2] = cij[0][1]
                    cij[1][1] = cij[0][0]
                    cij[1][2] = cij[0][1]
                    cij[2][2] = cij[0][0]
                    cij[4][4] = cij[3][3]
                    cij[5][5] = cij[3][3]

                sec_elastic.elastic_constants_matrix_second_order = symmetrize_matrix(cij)

                try:
                    self.calculate_nomad_error('workflow[-1].elastic.elastic_constants_matrix_second_order')
                except Exception:
                    self.logger.error('Failed to calculate nomad error.')

                if 'strain-gradient' in property_id:
                    # TODO implement symmetry
                    dij = [[get_value(entry, f'd-{i}-{j}.si-value', default=0) for i in range(1, 19)] for j in range(1, 19)]
                    sec_elastic.elastic_constants_gradient_matrix_second_order = symmetrize_matrix(dij)

                if 'excess.si-value' in entry:
                    sec_elastic.x_openkim_excess = entry['excess.si-value']

            if 'gamma-surface' in property_id or 'stacking-fault' in property_id or 'twinning-fault' in property_id:
                sec_workflow = self.archive.m_create(Workflow)
                sec_workflow.type = 'interface'
                sec_interface = sec_workflow.m_create(Interface)
                if 'gamma-surface.si-value' in entry:
                    directions, displacements = [], []
                    for key in entry.keys():
                        direction = re.match(r'fault-plane-shift-fraction-(\d+).source-value', key)
                        if direction:
                            directions.append(direction.group(1))
                            displacements.append(entry[key])
                    sec_interface.dimensionality = len(directions)
                    sec_interface.shift_direction = directions
                    sec_interface.displacement_fraction = displacements
                    sec_interface.gamma_surface = entry['gamma-surface.si-value']

                if 'fault-plane-energy.si-value' in entry:
                    sec_interface.dimensionality = 1
                    sec_interface.displacement_fraction = [entry['fault-plane-shift-fraction.source-value']]
                    sec_interface.energy_fault_plane = entry['fault-plane-energy.si-value']

                sec_interface.energy_extrinsic_stacking_fault = entry.get('extrinsic-stacking-fault-energy.si-value')
                sec_interface.energy_intrinsic_stacking_fault = entry.get('intrinsic-stacking-fault-energy.si-value')
                sec_interface.energy_unstable_stacking_fault = entry.get('unstable-stacking-energy.si-value')
                sec_interface.energy_unstable_twinning_fault = entry.get('unstable-twinning-energy.si-value')
                sec_interface.slip_fraction = entry.get('unstable-slip-fraction.source-value')

            if 'phonon-dispersion' in property_id:
                sec_workflow = self.archive.m_create(Workflow)
                sec_workflow.type = 'phonon'
                sec_phonon = sec_workflow.m_create(Phonon)
                if 'response-frequency.si-value' in entry:
                    sec_scc = sec_run.calculation[-1] if sec_run.calculation else sec_run.m_create(Calculation)
                    sec_bandstructure = sec_scc.m_create(BandStructure, Calculation.band_structure_phonon)
                    # TODO find a way to segment the frequencies
                    sec_segment = sec_bandstructure.m_create(BandEnergies)
                    energies = entry['response-frequency.si-value']
                    if len(np.shape(energies)) == 1:
                        energies = energies[0]
                    sec_segment.energies = [energies]
                    try:
                        wavevector = entry['wave-vector-direction.si-value']
                        cell = sec_run.system[-1].atoms.lattice_vectors.magnitude
                        # TODO how about spin-polarized case, not sure about calculation of kpoints value
                        sec_segment.kpoints = np.dot(wavevector, cell)
                    except Exception:
                        pass

                try:
                    self.calculate_nomad_error('run[-1].calculation[-1].band_structure_phonon[-1].segment[*].energies', phonon_transform_function)
                except Exception:
                    self.logger.error('Failed to calculate nomad error.')

                if 'wave-number.si-value' in entry:
                    sec_phonon.x_openkim_wave_number = [entry['wave-number.si-value']]

        # write archive to file
        if filename is not None:
            with open(filename, 'w') as f:
                json.dump(self.archive.m_to_dict(), f, indent=4)


class OpenKIMParser:
    def __init__(self):
        pass

    def parse(self, filepath, archive, logger):
        logger = logger if logger is not None else logging.getLogger('__name__')

        try:
            with open(os.path.abspath(filepath), 'rt') as f:
                archive_data = json.load(f)
        except Exception:
            logger.error('Error reading openkim archive')
            return

        if isinstance(archive_data, dict) and archive_data.get('run') is not None:
            archive.m_update_from_dict(archive_data)
            return

        # support for old version
        if isinstance(archive_data, dict) and archive_data.get('QUERY') is not None:
            archive_data = archive_data['QUERY']

        converter = Converter(archive_data, logger)
        converter.archive = archive
        converter.convert()


def openkim_entries_to_nomad_archive(entries, filename=None):
    if isinstance(entries, str):
        if filename is None:
            filename = 'openkim_archive_%s.json' % os.path.basename(entries).rstrip('.json')
        with open(entries) as f:
            entries = json.load(f)

    Converter(entries).convert(filename)
