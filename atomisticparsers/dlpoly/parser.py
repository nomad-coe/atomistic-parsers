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
from typing import Dict, Any

from nomad.units import ureg
from nomad.parsing.file_parser import TextParser, Quantity
from runschema.run import Run, Program
from runschema.method import (
    Method,
    ForceField,
    Model,
    Interaction,
    AtomParameters,
    MoleculeParameters,
)
from atomisticparsers.utils import MDParser
from .metainfo.dl_poly import m_package  # pylint: disable=unused-import


re_f = r'[-+]?\d*\.\d*(?:[Ee][-+]\d+)?'
re_n = r'[\n\r]'
MOL = 6.02214076e23


class TrajParser(TextParser):
    def init_quantities(self):
        def to_info(val_in):
            val = val_in.strip().split()
            if len(val) == 5:
                return dict(
                    step=val[0],
                    n_atoms=val[1],
                    log_level=val[2],
                    pbc_type=val[3],
                    dt=val[4],
                )
            elif len(val) == 2:
                return dict(log_level=val[0], pbc_type=val[1])
            elif len(val) > 2:
                return dict(log_level=val[0], pbc_type=val[1], n_atoms=val[2])
            return dict()

        self._quantities = [
            Quantity(
                'frame',
                rf'(\d+ +\d+.*\s+{re_f}[\s\S]+?)(?:timestep|\Z)',
                repeats=True,
                sub_parser=TextParser(
                    quantities=[
                        Quantity(
                            'info',
                            # rf'(\d+ +\d+[ \d]*(?:{re_f})*) *(?:{re_f})*) *{re_n}',
                            rf'(\d+ +\d+[ \d\.Ee\-\+]*){re_n}',
                            str_operation=to_info,
                        ),
                        Quantity(
                            'lattice_vectors',
                            rf'({re_f} +{re_f} +{re_f})\s+'
                            rf'({re_f} +{re_f} +{re_f})\s+'
                            rf'({re_f} +{re_f} +{re_f})',
                            dtype=np.dtype(np.float64),
                            shape=(3, 3),
                        ),
                        Quantity(
                            'atoms',
                            rf'({re_n} *[A-Za-z]+\S* *\d*.*(?:\s+{re_f})+)',
                            repeats=True,
                            sub_parser=TextParser(
                                quantities=[
                                    Quantity(
                                        'label', rf'{re_n} *([A-Z][a-z]*)\S* *\d*.*'
                                    ),
                                    Quantity(
                                        'array',
                                        rf'({re_f} +{re_f} +{re_f})',
                                        repeats=True,
                                        dtype=np.dtype(np.float64),
                                    ),
                                ]
                            ),
                        ),
                    ]
                ),
            )
        ]


class FieldParser(TextParser):
    def __init__(self):
        super().__init__()

    def init_quantities(self):
        def to_atoms(val_in):
            keys = [
                'label',
                'mass',
                'charge',
                'x_dl_poly_nrept',
                'x_dl_poly_ifrz',
                'x_dl_poly_igrp',
            ]
            return [
                {keys[n]: val_n for n, val_n in enumerate(val[: len(keys)])}
                for val in [v.split() for v in val_in.strip().splitlines()]
            ]

        def to_shell(val_in):
            keys = [
                'x_dl_poly_atom_indices_core',
                'x_dl_poly_atom_indices_shell',
                'x_dl_poly_force_constant',
                'x_dl_poly_force_constant_anharmonic',
            ]
            return [
                {keys[n]: val_n for n, val_n in enumerate(val[: len(keys)])}
                for val in [v.split() for v in val_in.strip().splitlines()]
            ]

        def to_bonds(val_in):
            val = [v.split() for v in val_in.strip().splitlines()]
            potentials = {
                'harm': 'Harmonic',
                'mors': 'Morse',
                '12-6': '12-6',
                'rhrm': 'Restraint',
                'quar': 'Quartic',
                'buck': 'Buckingham',
                'fene': 'FENE',
                'coul': 'Coulombic',
            }
            interactions = []
            for val_n in val:
                interactions.append(
                    dict(
                        functional_form=potentials.get(val_n[0], val_n[0]),
                        # atom index starts from 1
                        atom_indices=[[int(n) - 1 for n in val_n[1:3]]],
                        parameters=[float(v) for v in val_n[3:]],
                    )
                )
            return interactions

        def to_angles(val_in):
            val = [v.split() for v in val_in.strip().splitlines()]
            potentials = {
                'harm': 'Harmonic',
                'quar': 'Quartic',
                'thrm': 'Truncated harmonic',
                'sharm': 'Screened harmonic',
                'bvs1': 'Screened Vessa',
                'bvs2': 'Truncated Vessa',
                'bcos': 'Harmonic cosine',
                'cos': 'Cosine',
                'mmsb': 'MM strech-bend',
                'stst': 'Compass stretch-stretch',
                'stbe': 'Compass stretch-bend',
                'cmps': 'Compass all terms',
            }
            interactions = []
            for val_n in val:
                interactions.append(
                    dict(
                        functional_form=potentials.get(val_n[0], val_n[0]),
                        atom_indices=[[int(n) - 1 for n in val_n[1:4]]],
                        parameters=[float(v) for v in val_n[4:]],
                    )
                )
            return interactions

        def to_dihedrals(val_in):
            val = [v.split() for v in val_in.strip().splitlines()]
            potentials = {
                'cos': 'Cosine',
                'harm': 'Harmonic',
                'bcos': 'Harmonic cosine',
                'cos3': 'Triple cosine',
                'ryck': 'Ryckaert-Bellemans',
                'rbf': 'Fluorinated Ryckaert-Bellemans',
                'opls': 'OPLS',
            }
            interactions = []
            for val_n in val:
                interactions.append(
                    dict(
                        functional_form=potentials.get(val_n[0], val_n[0]),
                        atom_indices=[[int(n) - 1 for n in val_n[1:5]]],
                        parameters=[float(v) for v in val_n[5:]],
                    )
                )
            return interactions

        def to_inversions(val_in):
            val = [v.split() for v in val_in.strip().splitlines()]
            potentials = {
                'harm': 'Harmonic',
                'bcos': 'Harmonic cosine',
                'plan': 'Planar',
                'calc': 'Calcite',
            }
            interactions = []
            for val_n in val:
                interactions.append(
                    dict(
                        functional_form=potentials.get(val_n[0], val_n[0]),
                        atom_indices=[[int(n) - 1 for n in val_n[1:5]]],
                        parameters=[float(v) for v in val_n[5:]],
                    )
                )
            return interactions

        def to_teth(val_in):
            val = [v.split() for v in val_in.strip().splitlines()]
            potentials = {'harm': 'Harmonic', 'rhrm': 'Restraint', 'quar': 'Quartic'}
            interactions = []
            for val_n in val:
                interactions.append(
                    dict(
                        functional_form=potentials.get(val_n[0], val_n[0]),
                        atom_indices=[[int(n) - 1 for n in val_n[1:2]]],
                        parameters=[float(v) for v in val_n[2:]],
                    )
                )
            return interactions

        def to_rigid(val_in):
            val = [v.split() for v in val_in.strip().splitlines()]
            return [[int(vi) - 1 for vi in v[1 : int(v[0]) + 1]] for v in val]

        def to_vdw(val_in):
            val = [v.split() for v in val_in.strip().splitlines()]
            potentials = {
                '12-6': '12-6',
                'lj': 'Lennard-Jones',
                'nm': 'n-m',
                'buck': 'Buckingham',
                'bhm': 'Born-Huggins-Meyer',
                'hbnd': '12-10 H-bond',
                'snm': 'Shifted force n-m',
                'mors': 'Morse',
                'wca': 'WCA',
                'tab': 'Table',
            }
            interactions = []
            for val_n in val:
                interactions.append(
                    dict(
                        functional_form=potentials.get(val_n[2], val_n[2]),
                        atom_labels=[val_n[:2]],
                        parameters=[float(v) for v in val_n[3:]],
                    )
                )
            return interactions

        def to_tbp(val_in):
            val = [v.split() for v in val_in.strip().splitlines()]
            potentials = {
                'thrm': 'Truncated harmonic',
                'shrm': 'Screened Harmonic',
                'bvs1': 'Screened Vessa',
                'bvs2': 'Truncated Vessa',
                'hbnd': 'H-bond',
            }
            interactions = []
            for val_n in val:
                interactions.append(
                    dict(
                        functional_form=potentials.get(val_n[3], val_n[3]),
                        atom_labels=[val_n[:3]],
                        parameters=[float(v) for v in val_n[4:]],
                    )
                )
            return interactions

        def to_fbp(val_in):
            val = [v.split() for v in val_in.strip().splitlines()]
            potentials = {
                'harm': 'Harmonic',
                'hcos': 'Harmonic cosine',
                'plan': 'Planar',
            }
            interactions = []
            for val_n in val:
                interactions.append(
                    dict(
                        functional_form=potentials.get(val_n[4], val_n[4]),
                        atom_labels=[val_n[:4]],
                        parameters=[float(v) for v in val_n[5:]],
                    )
                )
            return interactions

        def to_metal(val_in):
            val = [v.split() for v in val_in.strip().splitlines()]
            potentials = {
                'eam': 'EAM',
                'fnsc': 'Finnis-Sinclair',
                'stch': 'Sutton-Chen',
                'gupt': 'Gupta',
            }
            interactions = []
            for val_n in val:
                interactions.append(
                    dict(
                        functional_form=potentials.get(val_n[2], val_n[2]),
                        atom_labels=[val_n[:2]],
                        parameters=[float(v) for v in val_n[3:]],
                    )
                )
            return interactions

        self._quantities = [
            Quantity('units', r'[Uu][Nn][Ii][Tt][Ss] +(.+)', dtype=str),
            Quantity(
                'neutral_groups', r'(neutral group)', str_operation=lambda x: True
            ),
            Quantity(
                'molecule_types',
                r'[Mm][Oo][Ll][Ee][Cc][Uu][Ll].+ +(\d+)',
                dtype=np.int32,
            ),
            Quantity(
                'molecule',
                r'(.+\s*[Nn][Uu][Mm][Mm][Oo][Ll][Ss] +\d+[\s\S]+?[Ff][Ii][Nn][Ii][Ss][Hh])',
                repeats=True,
                sub_parser=TextParser(
                    quantities=[
                        Quantity(
                            'label_nummols',
                            rf' *(.+?){re_n} *[Nn][Uu][Mm][Mm][Oo][Ll][Ss] *(\d+)',
                            str_operation=lambda x: x.rsplit(' ', 1),
                        ),
                        Quantity(
                            'atoms',
                            rf'[Aa][Tt][Oo][Mm][Ss] +\d+\s+((?:[A-Z][\w\-\+]* +{re_f}.+\s*)+)',
                            str_operation=to_atoms,
                        ),
                        Quantity(
                            'shell',
                            rf'[Ss][Hh][Ee][Ll][Ll] +\d+ +\d+\s+((?:\d+ +\d+ +{re_f}.+\s*)+)',
                            str_operation=to_shell,
                            convert=False,
                        ),
                        Quantity(
                            'bonds',
                            rf'[Bb][Oo][Nn][Dd][Ss] +\d+\s+((?:\w+ +\d+ +\d+ +{re_f}.+\s*)+)',
                            str_operation=to_bonds,
                            convert=False,
                        ),
                        Quantity(
                            'angles',
                            rf'[Aa][Nn][Gg][Ll][Ee][Ss] +\d+\s+((?:\w+ +\d+ +\d+ +\d+ +{re_f} +{re_f}.+\s*)+)',
                            str_operation=to_angles,
                            convert=False,
                        ),
                        Quantity(
                            'constraints',
                            rf'[Cc][Oo][Nn][Ss][Tt][Rr][Aa][Ii][Nn][Tt][Ss] +\d+\s+((?:\d+ +\d+ +{re_f}.*\s*)+)',
                            convert=False,
                            str_operation=lambda x: [
                                dict(
                                    atom_indices=[[int(v) - 1 for v in val[:2]]],
                                    parameters=[float(v) for v in val[2:3]],
                                )
                                for val in [v.split() for v in x.strip().splitlines()]
                            ],
                        ),
                        # TODO add pmf constraints
                        Quantity(
                            'dihedrals',
                            rf'[Dd][Ii][Hh][Ee][Dd][Rr][Aa][Ll][Ss] +\d+\s+((?:\w+ +\d+ +\d+ +\d+ +\d+ +{re_f}.+\s*)+)',
                            str_operation=to_dihedrals,
                            convert=False,
                        ),
                        Quantity(
                            'inversions',
                            rf'[Ii][Nn][Vv][Ee][Rr][Ss][Ii][Oo][Nn][Ss] +\d+\s+((?:\w+ +\d+ +\d+ +\d+ +\d+ +{re_f}.+\s*)+)',
                            str_operation=to_inversions,
                            convert=False,
                        ),
                        Quantity(
                            'rigid',
                            r'[Rr][Ii][Gg][Ii][Dd].+?\d+\s+((?:\d+.+\s+)+)',
                            str_operation=to_rigid,
                            convert=False,
                        ),
                        Quantity(
                            'teth',
                            rf'[Tt][Ee][Tt][Hh] +\d+\s+((?:\w+ +\d+ +{re_f}.+\s+)+)',
                            str_operation=to_teth,
                            convert=False,
                        ),
                    ]
                ),
            ),
            # non-bonded interactions
            Quantity(
                'vdw',
                rf'[Vv][Dd][Ww].+\d+\s+((?:[A-Z][\w\-\+]* +[A-Z][\w\-\+]* +\w+.*\s*)+)',
                str_operation=to_vdw,
                convert=False,
            ),
            Quantity(
                'tbp',
                r'[Tt][Bb][Pp].+\d+\s+((?:[A-Z][\w\-\+]* +[A-Z][\w\-\+]*.*\s*)+)',
                str_operation=to_tbp,
                convert=False,
            ),
            Quantity(
                'fbp',
                r'[Ff][Bb][Pp].+\d+\s+((?:[A-Z][\w\-\+]* +[A-Z][\w\-\+]*.*\s*)+)',
                str_operation=to_fbp,
                convert=False,
            ),
            Quantity(
                'metal',
                r'[Mm][Ee][Tt][Aa][Ll].+\d+\s+((?:[A-Z][\w\-\+]* +[A-Z][\w\-\+]*.*\s*)+)',
                str_operation=to_metal,
                convert=False,
            ),
            # TODO implement Tersoff and external fields
        ]


class MainfileParser(TextParser):
    def __init__(self):
        super().__init__()

    def init_quantities(self):
        self._quantities = [
            Quantity(
                'program_version_date',
                r'\*\* +version\: +(\S+) +/ +(\w+) +(\d+)',
                dtype=str,
            ),
            Quantity(
                'program_name',
                r'when publishing research data obtained using (\S+)',
                dtype=str,
            ),
            Quantity(
                'control_parameters',
                r'SIMULATION CONTROL PARAMETERS([\s\S]+?)SYS',
                sub_parser=TextParser(
                    quantities=[
                        Quantity(
                            'parameter',
                            r'([ \w]+) \(*\w*\)*((?:  |\:) [\w \+\-\.]+)',
                            str_operation=lambda x: [
                                v.strip() for v in x.replace(':', '  ').rsplit('  ', 1)
                            ],
                            repeats=True,
                        )
                    ]
                ),
            ),
            Quantity(
                'system_specification',
                r'TEM SPECIFICATION([\s\S]+?system volume.+)',
                sub_parser=TextParser(
                    quantities=[
                        Quantity(
                            'energy_unit',
                            r'energy units += +(.+)',
                            dtype=str,
                            flatten=False,
                        )
                    ]
                ),
            ),
            Quantity(
                'properties',
                r'(step +eng_tot +temp_tot[\s\S]+?)(?:run terminating|\Z)',
                sub_parser=TextParser(
                    quantities=[
                        Quantity(
                            'names',
                            r'(step +eng_tot +temp_tot.+\s+.+\s+.+)',
                            str_operation=lambda x: [
                                v.strip()
                                for v in x.replace('\n', '   ').split('   ')
                                if v
                            ],
                        ),
                        Quantity(
                            'instantaneous',
                            rf'{re_n} +(\d+ +{re_f}.+\s+.+\s+.+)',
                            repeats=True,
                        ),
                        Quantity(
                            'average',
                            r'rolling(.+)\s+averages(.+)\s+(.+)',
                            repeats=True,
                        ),
                    ]
                ),
            ),
        ]

    def get_control_parameters(self):
        return {
            val[0]: val[1]
            for val in self.get('control_parameters', {}).get('parameter', [])
        }


class DLPolyParser(MDParser):
    def __init__(self):
        self.mainfile_parser = MainfileParser()
        self.traj_parser = TrajParser()
        self.field_parser = FieldParser()
        self._units = {
            'kj/mol': ureg.kJ / MOL,
            'kj': ureg.kJ / MOL,
            'ev': ureg.eV,
            'kcal/mol': ureg.J * 4184.0 / MOL,
            'kcal': ureg.J * 4184.0 / MOL,
            'dl_poly internal units (10 j/mol)': 10 * ureg.J / MOL,
        }
        self._metainfo_map = {
            'step': 'step',
            'eng_tot': 'energy_total',
            'temp_tot': 'temperature',
            'eng_cfg': 'energy_contribution_configurational',
            'eng_vdw': 'energy_van_der_waals',
            'eng_src': 'energy_contribution_short_range',
            'eng_cou': 'energy_coulomb',
            'eng_bnd': 'energy_contribution_bond',
            'eng_ang': 'energy_contribution_angle',
            'eng_dih': 'energy_contribution_dihedral',
            'eng_tet': 'energy_contribution_tethering',
            'time(ps)': 'time',
            'time': 'time',
            'eng_pv': 'enthalpy',
            'temp_rot': 'x_dl_poly_temperature_rotational',
            # TODO include virial to nomad metainfo
            'vir_cfg': 'x_dl_poly_virial_configurational',
            'vir_src': 'x_dl_poly_virial_short_range',
            'vir_cou': 'x_dl_poly_virial_coulomb',
            'vir_bnd': 'x_dl_poly_virial_bond',
            'vir_ang': 'x_dl_poly_virial_angle',
            'vir_con': 'x_dl_poly_virial_constraint',
            'vir_tet': 'x_dl_poly_virial_tethering',
            'cpu  (s)': 'time_physical',
            'cpu time': 'time_physical',
            'volume': 'x_dl_poly_volume',
            'temp_shl': 'x_dl_poly_core_shell',
            'eng_shl': 'energy_contribution_core_shell',
            'vir_shl': 'x_dl_poly_core_shell',
            'vir_pmf': 'x_dl_poly_virial_potential_mean_force',
            'press': 'pressure',
        }
        super().__init__()

    def write_to_archive(self) -> None:
        self.mainfile_parser.logger = self.logger
        self.mainfile_parser.mainfile = self.mainfile
        self.traj_parser.logger = self.logger
        self.field_parser.logger = self.logger
        self.maindir = os.path.dirname(self.mainfile)

        sec_run = Run()
        self.archive.run.append(sec_run)
        version_date = self.mainfile_parser.get('program_version_date', [None, None])
        sec_run.program = Program(
            name=self.mainfile_parser.program_name, version=version_date[0]
        )
        sec_run.x_dl_poly_program_version_date = str(version_date[1])

        def get_system_data(frame_index):
            frame = self.traj_parser.get('frame')[frame_index]
            labels = [
                atom.get('label') if atom.get('label') else 'X'
                for atom in frame.get('atoms', [])
            ]
            lattice_vectors = frame.get('lattice_vectors') * ureg.angstrom
            array = np.transpose(
                [atom.get('array') for atom in frame.get('atoms', [])], axes=(1, 0, 2)
            )
            positions = array[0] * ureg.angstrom
            velocities = None
            if len(array) > 1:
                velocities = array[1] * ureg.angstrom / ureg.ps
            return dict(
                atoms=dict(
                    labels=labels,
                    lattice_vectors=lattice_vectors,
                    positions=positions,
                    velocities=velocities,
                )
            )

        # parse initial system from CONFIG
        self.traj_parser.mainfile = os.path.join(self.maindir, 'CONFIG')
        self.parse_trajectory_step(get_system_data(0))

        # method
        sec_method = Method()
        sec_run.method.append(sec_method)
        control_parameters = self.mainfile_parser.get_control_parameters()
        sec_method.x_dl_poly_control_parameters = control_parameters
        sec_force_field = ForceField()
        sec_method.force_field = sec_force_field
        sec_model = Model()
        sec_force_field.model.append(sec_model)
        # get interactions from FIELD file
        self.field_parser.mainfile = os.path.join(self.maindir, 'FIELD')
        units = {'mass': ureg.amu, 'charge': ureg.elementary_charge}
        for molecule in self.field_parser.get('molecule', []):
            # atom parameters
            sec_molecule_parameters = MoleculeParameters()
            sec_method.molecule_parameters.append(sec_molecule_parameters)
            sec_molecule_parameters.label = molecule.get('label_nummols', [None, None])[
                0
            ]
            for atom in molecule.get('atoms', []):
                sec_atom_parameters = AtomParameters()
                sec_molecule_parameters.atom_parameters.append(sec_atom_parameters)
                for key, val in atom.items():
                    val = val * units.get(key, 1)
                    setattr(sec_atom_parameters, key, val)
            # interactions
            for interaction_type in [
                'bonds',
                'angles',
                'dihedrals',
                'inversions',
                'teth',
            ]:
                for interaction in molecule.get(interaction_type, []):
                    sec_interaction = Interaction()
                    sec_model.contributions.append(sec_interaction)
                    for key, val in interaction.items():
                        sec_interaction.m_set(
                            sec_interaction.m_get_quantity_definition(key), val
                        )
            # add constraints to initial system
            constraint_data = []
            for constraint in molecule.get('constraints', []):
                constraint_data.append(
                    dict(
                        kind='fixed bond length',
                        atom_indices=constraint.get('atom_indices'),
                        parameters=constraint.get('parameters'),
                    )
                )
            # rigid atoms
            for rigid in molecule.get('rigid', []):
                constraint_data.append(dict(kind='static atoms', atom_indices=[rigid]))
            self.parse_section(dict(constraint=constraint_data), sec_run.system[0])
        # TODO add atom groups in system

        # non-bonded
        for interaction_type in ['vdw', 'tbp', 'fbp', 'metal']:
            for interaction in self.field_parser.get(interaction_type, []):
                sec_interaction = Interaction()
                sec_model.contributions.append(sec_interaction)
                for key, val in interaction.items():
                    quantity_def = sec_interaction.m_def.all_quantities.get(key)
                    if quantity_def:
                        sec_interaction.m_set(quantity_def, val)

        system_spec = self.mainfile_parser.get('system_specification', {})
        n_atoms = len(sec_run.system[-1].atoms.positions)
        energy_unit = (
            self._units.get(system_spec.get('energy_unit', '').strip().lower(), 1)
            * n_atoms
        )
        properties = self.mainfile_parser.get('properties', {})
        names = [self._metainfo_map.get(name) for name in properties.get('names', [])]
        # parse trajectory from HISTORY
        self.traj_parser.mainfile = os.path.join(self.maindir, 'HISTORY')

        # set up md parser
        self.n_atoms = n_atoms
        traj_steps = [
            frame.get('info', {}).get('step')
            for frame in self.traj_parser.get('frame', [])
        ]
        self.trajectory_steps = traj_steps
        step_n = names.index('step')
        self.thermodynamics_steps = [
            int(val[step_n]) for val in properties.get('instantaneous', [])
        ]

        for step in self.trajectory_steps:
            self.parse_trajectory_step(get_system_data(traj_steps.index(step)))

        unit_conversion = {'m': 60, 'f': 0.001, 's': 1, 'p': 1}
        for step in self.thermodynamics_steps:
            data = {}
            energy: Dict[str, Any] = {'contributions': []}
            instantaneous = properties.get('instantaneous')[
                self.thermodynamics_steps.index(step)
            ]
            # instataneous should be an array of floats, however some outputs may not be all floats and in that
            # case TextParser fails to convert them properly because it reuses the data type from
            # previous parsing run. Convert them manually here
            for n, val in enumerate(instantaneous):
                try:
                    if isinstance(val, str):
                        instantaneous[n] = float(val[:-1]) * unit_conversion.get(
                            val[-1], 1.0
                        )
                    else:
                        instantaneous[n] = float(val)
                except Exception:
                    instantaneous[n] = None

            for n, name in enumerate(names):
                if name is None or instantaneous[n] is None:
                    continue
                if name.startswith('energy_contribution_'):
                    energy['contributions'].append(
                        dict(
                            kind=name.replace('energy_contribution_', ''),
                            value=instantaneous[n] * energy_unit,
                        )
                    )
                elif name == 'energy_enthalpy':
                    energy['enthalpy'] = instantaneous[n] * energy_unit
                elif name.startswith('energy_'):
                    energy[name.replace('energy_', '')] = dict(
                        value=instantaneous[n] * energy_unit
                    )
                elif name == 'enthalpy':
                    data['enthalpy'] = instantaneous[n] * energy_unit
                elif 'temperature' in name:
                    data[name] = instantaneous[n] * ureg.kelvin
                elif 'pressure' in name:
                    # TODO verify if atmosphere is the unit
                    data[name] = instantaneous[n] * ureg.atm
                elif 'volume' in name:
                    data[name] = instantaneous[n] * ureg.angstrom**3
                elif name == 'step':
                    data[name] = int(instantaneous[n])
                elif name == 'time':
                    data[name] = instantaneous[n] * ureg.ps
                elif name == 'time_physical':
                    data[name] = instantaneous[n] * ureg.s
                    index = self.thermodynamics_steps.index(step)
                    time_start = (
                        properties.get('instantaneous')[index - 1][n]
                        if index > 0
                        else 0
                    )
                    data['time_calculation'] = data[name] - time_start * ureg.s
                else:
                    data[name] = instantaneous[n]
            data['energy'] = energy
            if step in traj_steps:
                frame = self.traj_parser.get('frames')[traj_steps.index(step)]
                array = np.transpose(
                    [atom.get('array') for atom in frame.get('atoms', [])]
                )
                if len(array) > 2:
                    data['forces'] = dict(
                        total=dict(
                            value=np.transpose(array[2])
                            * ureg.amu
                            * ureg.angstrom
                            / ureg.ps**2
                        )
                    )
            self.parse_thermodynamics_step(data)

        ensemble_type = control_parameters.get('Ensemble')
        self.parse_md_workflow(
            dict(
                method=dict(
                    thermodynamic_ensemble=ensemble_type.split()[0]
                    if ensemble_type is not None
                    else None,
                    integration_timestep=control_parameters.get(
                        'fixed simulation timestep', 0
                    )
                    * ureg.ps,
                )
            )
        )
