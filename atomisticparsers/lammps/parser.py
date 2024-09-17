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
import os
from ase import data as asedata
import re

from nomad.units import ureg

from nomad.parsing.file_parser import Quantity, TextParser
from runschema.run import Run, Program
from runschema.method import (
    NeighborSearching,
    ForceCalculations,
    ForceField,
    Method,
    Model,
    AtomParameters,
)
from runschema.system import AtomsGroup
from simulationworkflowschema import (
    GeometryOptimization,
    GeometryOptimizationMethod,
    GeometryOptimizationResults,
)
from .metainfo.lammps import (
    x_lammps_section_input_output_files,
    x_lammps_section_control_parameters,
)
from atomisticparsers.utils import MDAnalysisParser, MDParser
from simulationworkflowschema.molecular_dynamics import (
    get_bond_list_from_model_contributions,
)


re_float = r'[-+]?\d+\.*\d*(?:[Ee][-+]\d+)?'
re_n = r'[\n\r]'


def get_unit(units_type, property_type=None, dimension=3):
    mole = 6.022140857e23

    units_type = units_type.lower()
    if units_type == 'real':
        units = dict(
            mass=ureg.g / mole,
            distance=ureg.angstrom,
            time=ureg.fs,
            energy=ureg.J * 4184.0 / mole,
            velocity=ureg.angstrom / ureg.fs,
            force=ureg.J * 4184.0 / ureg.angstrom / mole,
            torque=ureg.J * 4184.0 / mole,
            temperature=ureg.K,
            pressure=ureg.atm,
            dynamic_viscosity=ureg.poise,
            charge=ureg.elementary_charge,
            dipole=ureg.elementary_charge * ureg.angstrom,
            electric_field=ureg.V / ureg.angstrom,
            density=ureg.g / ureg.cm**dimension,
        )

    elif units_type == 'metal':
        units = dict(
            mass=ureg.g / mole,
            distance=ureg.angstrom,
            time=ureg.ps,
            energy=ureg.eV,
            velocity=ureg.angstrom / ureg.ps,
            force=ureg.eV / ureg.angstrom,
            torque=ureg.eV,
            temperature=ureg.K,
            pressure=ureg.bar,
            dynamic_viscosity=ureg.poise,
            charge=ureg.elementary_charge,
            dipole=ureg.elementary_charge * ureg.angstrom,
            electric_field=ureg.V / ureg.angstrom,
            density=ureg.g / ureg.cm**dimension,
        )

    elif units_type == 'si':
        units = dict(
            mass=ureg.kg,
            distance=ureg.m,
            time=ureg.s,
            energy=ureg.J,
            velocity=ureg.m / ureg.s,
            force=ureg.N,
            torque=ureg.N * ureg.m,
            temperature=ureg.K,
            pressure=ureg.Pa,
            dynamic_viscosity=ureg.Pa * ureg.s,
            charge=ureg.C,
            dipole=ureg.C * ureg.m,
            electric_field=ureg.V / ureg.m,
            density=ureg.kg / ureg.m**dimension,
        )

    elif units_type == 'cgs':
        units = dict(
            mass=ureg.g,
            distance=ureg.cm,
            time=ureg.s,
            energy=ureg.erg,
            velocity=ureg.cm / ureg.s,
            force=ureg.dyne,
            torque=ureg.dyne * ureg.cm,
            temperature=ureg.K,
            pressure=ureg.dyne / ureg.cm**2,
            dynamic_viscosity=ureg.poise,
            charge=ureg.esu,
            dipole=ureg.esu * ureg.cm,
            electric_field=ureg.dyne / ureg.esu,
            density=ureg.g / ureg.cm**dimension,
        )

    elif units_type == 'electron':
        units = dict(
            mass=ureg.amu,
            distance=ureg.bohr,
            time=ureg.fs,
            energy=ureg.hartree,
            velocity=ureg.bohr / ureg.atomic_unit_of_time,
            force=ureg.hartree / ureg.bohr,
            temperature=ureg.K,
            pressure=ureg.Pa,
            charge=ureg.elementary_charge,
            dipole=ureg.debye,
            electric_field=ureg.V / ureg.cm,
        )

    elif units_type == 'micro':
        units = dict(
            mass=ureg.pg,
            distance=ureg.microm,
            time=ureg.micros,
            energy=ureg.pg * ureg.microm**2 / ureg.micros**2,
            velocity=ureg.microm / ureg.micros,
            force=ureg.pg * ureg.microm / ureg.micros**2,
            torque=ureg.pg * ureg.microm**2 / ureg.micros**2,
            temperature=ureg.K,
            pressure=ureg.pg / (ureg.microm * ureg.micros**2),
            dynamic_viscosity=ureg.pg / (ureg.microm * ureg.micros),
            charge=ureg.pC,
            dipole=ureg.pC * ureg.microm,
            electric_field=ureg.V / ureg.microm,
            density=ureg.pg / ureg.microm**dimension,
        )

    elif units_type == 'nano':
        units = dict(
            mass=ureg.ag,
            distance=ureg.nm,
            time=ureg.ns,
            energy=ureg.ag * ureg.nm**2 / ureg.ns**2,
            velocity=ureg.nm / ureg.ns,
            force=ureg.ag * ureg.nm / ureg.ns**2,
            torque=ureg.ag * ureg.nm**2 / ureg.ns**2,
            temperature=ureg.K,
            pressure=ureg.ag / (ureg.nm * ureg.ns**2),
            dynamic_viscosity=ureg.ag / (ureg.nm * ureg.ns),
            charge=ureg.elementary_charge,
            dipole=ureg.elementary_charge * ureg.nm,
            electric_field=ureg.V / ureg.nm,
            density=ureg.ag / ureg.nm**dimension,
        )

    else:
        # units = dict(
        #     mass=1, distance=1, time=1, energy=1, velocity=1, force=1,
        #     torque=1, temperature=1, pressure=1, dynamic_viscosity=1, charge=1,
        #     dipole=1, electric_field=1, density=1)
        units = dict()

    if property_type:
        return units.get(property_type, None)
    else:
        return units


class DataParser(TextParser):
    def __init__(self):
        self._headers = [
            'atoms',
            'bonds',
            'angles',
            'dihedrals',
            'impropers',
            'atom types',
            'bond types',
            'angle types',
            'dihedral types',
            'improper types',
            'extra bond per atom',
            'extra/bond/per/atom',
            'extra angle per atom',
            'extra/angle/per/atom',
            'extra dihedral per atom',
            'extra/dihedral/per/atom',
            'extra improper per atom',
            'extra/improper/per/atom',
            'extra special per atom',
            'extra/special/per/atom',
            'ellipsoids',
            'lines',
            'triangles',
            'bodies',
        ]
        self._sections = [
            'Atoms',
            'Velocities',
            'Masses',
            'Ellipsoids',
            'Lines',
            'Triangles',
            'Bodies',
            'Bonds',
            'Angles',
            'Dihedrals',
            'Impropers',
            'Pair Coeffs',
            'PairIJ Coeffs',
            'Bond Coeffs',
            'Angle Coeffs',
            'Dihedral Coeffs',
            'Improper Coeffs',
            'BondBond Coeffs',
            'BondAngle Coeffs',
            'MiddleBondTorsion Coeffs',
            'EndBondTorsion Coeffs',
            'AngleTorsion Coeffs',
            'AngleAngleTorsion Coeffs',
            'BondBond13 Coeffs',
            'AngleAngle Coeffs',
        ]
        self._interactions = [
            section for section in self._sections if section.endswith('Coeffs')
        ]
        super().__init__(None)

    def init_quantities(self):
        self._quantities = [
            Quantity(header, rf'{re_n} *(\d+) +{header}', repeats=True, dtype=np.int32)
            for header in self._headers
        ]

        def get_section_value(val):
            val = val.strip().splitlines()
            name = None

            if val[0][0] == '#':
                name = val[0][1:].strip()
                val = val[1:]

            value = []
            for i in range(len(val)):
                v = val[i].split('#')[0].split()
                if not v:
                    continue

                try:
                    value.append(np.array(v, dtype=float))
                except Exception:
                    break

            return name, np.array(value)

        self._quantities.extend(
            [
                Quantity(
                    section,
                    rf'{section} *(#*.*{re_n}\s+(?:[\d ]+{re_float}.+\s+)+)',
                    str_operation=get_section_value,
                    repeats=True,
                )
                for section in self._sections
            ]
        )

    def get_interactions(self):
        styles_coeffs = []
        for interaction in self._interactions:
            coeffs = self.get(interaction, None)
            if coeffs is None:
                continue
            if isinstance(coeffs, tuple):
                coeffs = list(coeffs)

            styles_coeffs += coeffs

        return styles_coeffs


class TrajParser(TextParser):
    def __init__(self):
        self._masses = None
        self._reference_masses = dict(
            masses=np.array(asedata.atomic_masses), symbols=asedata.chemical_symbols
        )
        self._chemical_symbols = None
        super().__init__(None)

    def init_quantities(self):
        def get_pbc_cell(val):
            val = val.split()

            pbc = [v == 'pp' for v in val[:3]]

            cell = np.zeros((3, 3))
            for i in range(3):
                cell[i][i] = float(val[i * 2 + 4]) - float(val[i * 2 + 3])

            return pbc, cell

        def get_atoms_info(val):
            val = val.split('\n')
            keys = val[0].split()
            values = np.array([v.split() for v in val[1:] if v], dtype=float)
            values = values[values[:, 0].argsort()].T
            return {keys[i]: values[i] for i in range(len(keys))}

        self._quantities = [
            Quantity(
                'time_step',
                r'\s*ITEM:\s*TIMESTEP\s*\n\s*(\d+)\s*\n',
                comment='#',
                repeats=True,
            ),
            Quantity(
                'n_atoms',
                r'\s*ITEM:\s*NUMBER OF ATOMS\s*\n\s*(\d+)\s*\n',
                comment='#',
                repeats=True,
            ),
            Quantity(
                'pbc_cell',
                r'\s*ITEM: BOX BOUNDS\s*([\s\w]+)\n([\+\-\d\.eE\s]+)\n',
                str_operation=get_pbc_cell,
                comment='#',
                repeats=True,
            ),
            Quantity(
                'atoms_info',
                r'\s*ITEM:\s*ATOMS\s*([ \w]+\n)*?([\+\-eE\d\.\n ]+)',
                str_operation=get_atoms_info,
                comment='#',
                repeats=True,
            ),
        ]

    @property
    def with_trajectory(self):
        return self.get('atoms_info') is not None

    @property
    def n_frames(self):
        return len(self.get('atoms_info', []))

    @property
    def masses(self):
        return self._masses

    @masses.setter
    def masses(self, val):
        self._masses = val
        if self._masses is None:
            return

        self._masses = val
        if self._chemical_symbols is None:
            masses = self._masses[0][1]
            self._chemical_symbols = {}
            for i in range(len(masses)):
                symbol_idx = np.argmin(
                    abs(self._reference_masses['masses'] - masses[i][1])
                )
                self._chemical_symbols[masses[i][0]] = self._reference_masses[
                    'symbols'
                ][symbol_idx]

    def get_atom_labels(self, idx):
        atoms_info = self.get('atoms_info')
        if atoms_info is None:
            return

        atoms_id = atoms_info[idx].get('id')
        default = ['X' for _ in atoms_id] if atoms_id is not None else None
        atoms_type = atoms_info[idx].get('type')
        if atoms_type is None:
            return default
        if self._chemical_symbols is None:
            return default

        try:
            atom_labels = [self._chemical_symbols[atype] for atype in atoms_type]
        except Exception:
            self.logger.error('Error resolving atom labels.')
            return

        return atom_labels

    def get_positions(self, idx):
        atoms_info = self.get('atoms_info')
        if atoms_info is None:
            return

        atoms_info = atoms_info[idx]

        cell = self.get('pbc_cell')
        cell = None if cell is None else cell[idx][1]
        if 'xs' in atoms_info and 'ys' in atoms_info and 'zs' in atoms_info:
            if cell is None:
                return
            positions = np.array(
                [atoms_info['xs'], atoms_info['ys'], atoms_info['zs']]
            ).T
            positions = positions * np.linalg.norm(cell, axis=1) + np.amin(cell, axis=1)

        elif 'xu' in atoms_info and 'yu' in atoms_info and 'zu' in atoms_info:
            positions = np.array(
                [atoms_info['xu'], atoms_info['yu'], atoms_info['zu']]
            ).T

        elif 'xsu' in atoms_info and 'ysu' in atoms_info and 'zsu' in atoms_info:
            if cell is None:
                return
            positions = np.array(
                [atoms_info['xsu'], atoms_info['ysu'], atoms_info['zsu']]
            ).T
            positions = positions * np.linalg.norm(cell, axis=1) + np.amin(cell, axis=1)

        elif 'x' in atoms_info and 'y' in atoms_info and 'z' in atoms_info:
            positions = np.array([atoms_info['x'], atoms_info['y'], atoms_info['z']]).T
            if 'ix' in atoms_info and 'iy' in atoms_info and 'iz' in atoms_info:
                if cell is None:
                    return
                positions_img = np.array(
                    [atoms_info['ix'], atoms_info['iy'], atoms_info['iz']]
                ).T

                positions += positions_img * np.linalg.norm(cell, axis=1)
        else:
            positions = None

        return positions

    def get_velocities(self, idx):
        atoms_info = self.get('atoms_info')
        if atoms_info is None:
            return
        atoms_info = atoms_info[idx]
        if 'vx' not in atoms_info or 'vy' not in atoms_info or 'vz' not in atoms_info:
            return

        return np.array([atoms_info['vx'], atoms_info['vy'], atoms_info['vz']]).T

    def get_forces(self, idx):
        atoms_info = self.get('atoms_info')
        if atoms_info is None:
            return
        atoms_info = atoms_info[idx]
        if 'fx' not in atoms_info or 'fy' not in atoms_info or 'fz' not in atoms_info:
            return
        return np.array([atoms_info['fx'], atoms_info['fy'], atoms_info['fz']]).T

    def get_lattice_vectors(self, idx):
        pbc_cell = self.get('pbc_cell')
        if pbc_cell is None:
            return
        return pbc_cell[idx][1]

    def get_pbc(self, idx):
        pbc_cell = self.get('pbc_cell')
        if pbc_cell is None:
            return
        return pbc_cell[idx][0]

    def get_n_atoms(self, idx):
        n_atoms = self.get('n_atoms')
        if n_atoms is None:
            return len(self.get_positions(idx))
        return n_atoms[idx]

    def get_step(self, idx):
        step = self.get('time_step')
        if step is None:
            return
        return step[idx]


class XYZTrajParser(TrajParser):
    def __init__(self):
        super().__init__()

    def init_quantities(self):
        def get_atoms_info(val_in):
            val = [v.split('#')[0].split() for v in val_in.strip().split('\n')]
            symbols = []
            for v in val:
                if v[0].isalpha():
                    if v[0] not in symbols:
                        symbols.append(v[0])
                    v[0] = symbols.index(v[0]) + 1
            val = np.transpose(np.array([v for v in val if len(v) == 4], dtype=float))
            # val[0] is the atomic number
            val[0] = [list(set(val[0])).index(v) + 1 for v in val[0]]
            return dict(type=val[0], x=val[1], y=val[2], z=val[3])

        self.quantities = [
            Quantity(
                'atoms_info',
                r'((?:\d+|[A-Z][a-z]?) [\s\S]+?)(?:\s\d+\n|\Z)',
                str_operation=get_atoms_info,
                comment='#',
                repeats=True,
            )
        ]


class LogParser(TextParser):
    def __init__(self):
        self._commands = [
            'angle_coeff',
            'angle_style',
            'atom_modify',
            'atom_style',
            'balance',
            'bond_coeff',
            'bond_style',
            'bond_write',
            'boundary',
            'change_box',
            'clear',
            'comm_modify',
            'comm_style',
            'compute',
            'compute_modify',
            'create_atoms',
            'create_bonds',
            'create_box',
            'delete_bonds',
            'dielectric',
            'dihedral_coeff',
            'dihedral_style',
            'dimension',
            'displace_atoms',
            'dump',
            'dump_modify',
            'dynamical_matrix',
            'echo',
            'fix',
            'fix_modify',
            'group',
            'group2ndx',
            'ndx2group',
            'hyper',
            'if',
            'improper_coeff',
            'improper_style',
            'include',
            'info',
            'jump',
            'kim_init',
            'kim_interactions',
            'kim_query',
            'kim_param',
            'kim_property',
            'kspace_modify',
            'kspace_style',
            'label',
            'lattice',
            'log',
            'mass',
            'message',
            'min_modify',
            'min_style',
            'minimize',
            'minimize/kk',
            'molecule',
            'neb',
            'neb/spin',
            'neigh_modify',
            'neighbor',
            'newton',
            'next',
            'package',
            'pair_coeff',
            'pair_modify',
            'pair_style',
            'pair_write',
            'partition',
            'prd',
            'print',
            'processors',
            'quit',
            'read_data',
            'read_dump',
            'read_restart',
            'region',
            'replicate',
            'rerun',
            'reset_atom_ids',
            'reset_mol_ids',
            'reset_timestep',
            'restart',
            'run',
            'run_style',
            'server',
            'set',
            'shell',
            'special_bonds',
            'suffix',
            'tad',
            'temper/grem',
            'temper/npt',
            'thermo',
            'thermo_modify',
            'thermo_style',
            'third_order',
            'timer',
            'timestep',
            'uncompute',
            'undump',
            'unfix',
            'units',
            'variable',
            'velocity',
            'write_coeff',
            'write_data',
            'write_dump',
            'write_restart',
        ]
        self._interactions = [
            'atom',
            'pair',
            'bond',
            'angle',
            'dihedral',
            'improper',
            'kspace',
        ]
        self._units = None
        super().__init__(None)

    def init_quantities(self):
        def str_op(val):
            val = val.split('#')[0]
            val = val.replace('&\n', ' ').split()
            val = val if len(val) > 1 else val[0]
            return val

        self._quantities = [
            Quantity(
                name,
                r'\n\s*%s\s+(?!.*\$\{)([${}\w\. \/\#\-]+)(\&\n[\w\. \/\#\-]*)*' % name,
                str_operation=str_op,
                comment='#',
                repeats=True,
            )
            for name in self._commands
        ]

        self._quantities.append(
            Quantity(
                'program_version',
                r'\s*LAMMPS\s*\(([\w ]+)\)\n',
                dtype=str,
                repeats=False,
                flatten=False,
            )
        )

        self._quantities.append(
            Quantity('finished', r'\s*Dangerous builds\s*=\s*(\d+)', repeats=False)
        )

        self._quantities.append(
            Quantity(
                'minimization_stats',
                r'\s*Minimization stats:\s*([\s\S]+?)\n\n',
                flatten=False,
            )
        )

        def str_to_thermo(val):
            res = {}
            if val.count('Step') > 1:
                val = (
                    val.replace('--', '').replace('=', '').replace('(sec)', '').split()
                )
                val = [v.strip() for v in val]

                for i in range(len(val)):
                    if val[i][0].isalpha():
                        res.setdefault(val[i], [])
                        res[val[i]].append(float(val[i + 1]))

            else:
                val = val.split('\n')
                keys = [v.strip() for v in val[0].split()]
                val = np.array([v.split() for v in val[1:] if v], dtype=float).T

                res = {key: [] for key in keys}
                for i in range(len(keys)):
                    res[keys[i]] = val[i]

            return res

        self._quantities.append(
            Quantity(
                'thermo_data',
                r'\s*\-*(\s*Step\s*[\-\s\w\.\=\(\)]*[ \-\.\d\n]+)Loop',
                str_operation=str_to_thermo,
                repeats=False,
                convert=False,
            )
        )

    @property
    def units(self):
        if self._units is None:
            units_type = self.get('units', ['lj'])[0]
            self._units = get_unit(units_type)
        return self._units

    def get_thermodynamic_data(self):
        thermo_data = self.get('thermo_data')

        if thermo_data is None:
            return

        data = {}
        for key, val in thermo_data.items():
            low_key = key.lower()
            if low_key.startswith('e_') or low_key.endswith('eng'):
                data[key] = val * self.units.get('energy', 1)
            elif low_key == 'press':
                data[key] = val * self.units.get('pressure', 1)
            elif low_key == 'temp':
                data[key] = val * self.units.get('temperature', 1)
            else:
                data[key] = val
        return data

    def get_traj_files(self):
        dump = self.get('dump')
        if dump is None:
            self.logger.warning('Trajectory not specified in directory, will scan.')
            # TODO improve matching of traj file
            traj_files = os.listdir(self.maindir)
            traj_files = [
                f for f in traj_files if f.endswith('trj') or f.endswith('xyz')
            ]
            # further eliminate
            if len(traj_files) > 1:
                prefix = os.path.basename(self.mainfile).rsplit('.', 1)[0]
                traj_files = [f for f in traj_files if prefix in f]
        else:
            traj_files = []
            if type(dump[0]) in [str, int]:
                dump = [dump]
            traj_files = [d[4] for d in dump]
        traj_files = [
            i for n, i in enumerate(traj_files) if i not in traj_files[:n]
        ]  # remove duplicates

        return [os.path.join(self.maindir, f) for f in traj_files]

    def get_data_files(self):
        def check_file_header(file_path, regex_pattern):
            header_size = 1024
            file_path = f'{self.maindir}/{file_path}'
            try:
                with open(file_path, 'rb') as file:
                    file_header = file.read(header_size)
                    file_header_str = file_header.decode(errors='ignore')
            except Exception:
                file_header_str = ''

            return re.search(regex_pattern, file_header_str)

        read_data = self.get('read_data')
        if read_data is None or 'CPU' in read_data:
            self.logger.warning('Data file not specified in directory, will scan.')
            data_files = os.listdir(self.maindir)
            data_files = [
                f for f in data_files if f.endswith('data') or f.startswith('data')
            ]
            if not data_files:
                data_files = os.listdir(self.maindir)
                data_files = [
                    f for f in data_files if check_file_header(f, 'LAMMPS data file')
                ]  # TODO: Should this be the default?
            if len(data_files) > 1:
                prefix = os.path.basename(self.mainfile).rsplit('.', 1)
                prefix = (
                    prefix[1] if len(prefix) > 1 and prefix[1] != 'log' else prefix[0]
                )
                data_files = [f for f in data_files if prefix in f]
        else:
            data_files = read_data

        return [os.path.join(self.maindir, f) for f in data_files]

    def get_pbc(self):
        pbc = self.get('boundary', ['p', 'p', 'p'])
        return [v == 'p' for v in pbc]

    def get_sampling_method(self):
        fix_style = self.get('fix', [[''] * 3])[0][2]

        sampling_method = (
            'langevin_dynamics' if 'langevin' in fix_style else 'molecular_dynamics'
        )
        return sampling_method, fix_style

    def get_thermostat_settings(self):
        fix = self.get('fix', [None])[0]
        if fix is None:
            return {}

        try:
            fix_style = fix[2]
        except IndexError:
            return {}

        temp_unit = self.units.get('temperature', 1)
        press_unit = self.units.get('pressure', 1)
        time_unit = self.units.get('time', 1)

        res = dict()
        if fix_style.lower() == 'nvt':
            try:
                res['target_T'] = float(fix[5]) * temp_unit
                res['thermostat_tau'] = float(fix[6]) * time_unit
            except Exception:
                pass

        elif fix_style.lower() == 'npt':
            try:
                res['target_T'] = float(fix[5]) * temp_unit
                res['thermostat_tau'] = float(fix[6]) * time_unit
                res['target_P'] = float(fix[9]) * press_unit
                res['barostat_tau'] = float(fix[10]) * time_unit
            except Exception:
                pass

        elif fix_style.lower() == 'nph':
            try:
                res['target_P'] = float(fix[5]) * press_unit
                res['barostat_tau'] = float(fix[6]) * time_unit
            except Exception:
                pass

        elif fix_style.lower() == 'langevin':
            try:
                res['target_T'] = float(fix[4]) * temp_unit
                res['langevin_gamma'] = float(fix[5]) * time_unit
            except Exception:
                pass

        else:
            self.logger.warning('Fix style not supported', data=dict(style=fix_style))

        return res

    def get_interactions(self):
        styles_coeffs = []
        for interaction in self._interactions:
            styles = self.get('%s_style' % interaction, None)
            if styles is None:
                continue

            if isinstance(styles[0], str):
                styles = [styles]

            for i in range(len(styles)):
                if interaction == 'kspace':
                    coeff = [[float(c) for c in styles[i][1:]]]
                    style = styles[i][0]

                else:
                    coeff = self.get('%s_coeff' % interaction)
                    style = ' '.join([str(si) for si in styles[i]])

                styles_coeffs.append((style.strip(), coeff))

        return styles_coeffs


class TrajParsers:
    def __init__(self, parsers):
        self._parsers = parsers
        for parser in parsers:
            parser.parse()

    def __getitem__(self, index):
        if self._parsers:
            return self._parsers[index]

    def eval(self, key, *args, **kwargs):
        for parser in self._parsers:
            parser_method = getattr(parser, key)
            if parser_method is not None:
                val = (
                    parser_method(*args, **kwargs) if args or kwargs else parser_method
                )
                if val is not None:
                    return val


class LammpsParser(MDParser):
    def __init__(self):
        super().__init__()
        self.log_parser = LogParser()
        self._traj_parser = TrajParser()
        self._xyztraj_parser = XYZTrajParser()
        self._mdanalysistraj_parser = MDAnalysisParser(
            topology_format='DATA', format='LAMMPSDUMP'
        )
        self.data_parser = DataParser()
        self.aux_log_parser = LogParser()
        self._energy_mapping = {
            'e_pair': 'pair',
            'e_bond': 'bond',
            'e_angle': 'angle',
            'e_dihed': 'dihedral',
            'e_impro': 'improper',
            'e_coul': 'coulomb',
            'e_vdwl': 'van der Waals',
            'e_mol': 'molecular',
            'e_long': 'kspace long range',
            'e_tail': 'van der Waals long range',
            'kineng': 'kinetic',
            'poteng': 'potential',
        }

    def get_time_step(self):
        time_unit = self.log_parser.units.get('time', None)
        time_step = self.log_parser.get('timestep', [0], unit=time_unit)[0]
        return time_step

    def parse_thermodynamic_data(self):
        sec_run = self.archive.run[-1]
        # sec_system = sec_run.system

        time_step = self.get_time_step()
        thermo_data = self.log_parser.get_thermodynamic_data()
        if thermo_data is None:
            thermo_data = self.aux_log_parser.get_thermodynamic_data()
        if not thermo_data:
            thermo_data = {}
        self.thermodynamics_steps = [int(n) for n in thermo_data.get('Step', [])]

        if not thermo_data:
            return

        for step in self.thermodynamics_steps:
            step_data = {
                'step': step,
                'time': step * time_step,
                'method_ref': sec_run.method[-1] if sec_run.method else None,
                'energy': {'contributions': []},
            }
            if step in self._trajectory_steps:
                step_data['forces'] = (
                    dict(
                        total=dict(
                            value=self.traj_parsers.eval(
                                'get_forces', self._trajectory_steps.index(step)
                            )
                        )
                    ),
                )

            data_n = self._thermodynamics_steps.index(step)
            for key, val in thermo_data.items():
                key = key.lower()
                if (kind := self._energy_mapping.get(key)) is not None:
                    step_data['energy']['contributions'].append(
                        dict(kind=kind, value=val[data_n])
                    )
                elif key == 'toteng':
                    step_data['energy']['current'] = dict(value=val[data_n])
                    step_data['energy']['total'] = dict(value=val[data_n])
                elif key == 'press':
                    step_data['pressure'] = val[data_n]
                elif key == 'temp':
                    step_data['temperature'] = val[data_n]
                elif key == 'cpu':
                    # approx time calc is dt / dstep
                    max_step = len(self._thermodynamics_steps) - 1
                    # calc time cannot be calculated for last iter, will be zero
                    delta_time = float(val[min(data_n + 1, max_step)]) - float(
                        val[data_n]
                    )
                    delta_step = (
                        1
                        if data_n == max_step
                        else self._thermodynamics_steps[data_n + 1] - step
                    )
                    step_data['time_calculation'] = delta_time * ureg.s / delta_step
                    step_data['time_physical'] = (
                        float(val[data_n]) * ureg.s + step_data['time_calculation']
                    )

            self.parse_thermodynamics_step(step_data)

    def parse_workflow(self):
        sec_run = self.archive.run[-1]
        sec_calc = sec_run.get('calculation')
        sec_lammps = sec_run.x_lammps_section_control_parameters[-1]

        units = self.log_parser.units
        if not units:
            self.logger.warning(
                'Unit information not available. Assuming "real" units in workflow metainfo!'
            )
            units = get_unit('real')
        energy_conversion = ureg.convert(1.0, units.get('energy'), ureg.joule)
        force_conversion = ureg.convert(1.0, units.get('force'), ureg.newton)
        temperature_conversion = ureg.convert(
            1.0, units.get('temperature'), ureg.kelvin
        )
        pressure_conversion = ureg.convert(1.0, units.get('pressure'), ureg.pascal)

        minimization_stats = self.log_parser.get('minimization_stats', None)
        workflow = None
        if minimization_stats is not None:
            workflow = GeometryOptimization(
                method=GeometryOptimizationMethod(),
                results=GeometryOptimizationResults(),
            )
            workflow.method.type = 'atomic'

            min_style = self.log_parser.get('min_style')
            min_style = min_style[0].lower() if min_style else 'none'
            min_style_map = {
                'cg': 'polak_ribiere_conjugant_gradient',
                'hftn': 'hessian_free_truncated_newton',
                'sd': 'steepest_descent',
                'quickmin': 'damped_dynamics',
                'fire': 'damped_dynamics',
                'spin': 'damped_dynamics',
            }
            value = min_style_map.get(
                min_style,
                [val for key, val in min_style_map.items() if key in min_style],
            )
            value = (
                value
                if not isinstance(value, list)
                else value[0]
                if len(value) != 0
                else None
            )
            workflow.method.method = value

            minimization_stats = minimization_stats.split('\n')
            energy_index = [
                i
                for i, s in enumerate(minimization_stats)
                if 'Energy initial, next-to-last, final' in s
            ]
            if len(energy_index) != 0:
                energy_stats = minimization_stats[energy_index[0] + 1].split()
                workflow.results.final_energy_difference = (
                    float(energy_stats[-1]) - float(energy_stats[-2])
                ) * energy_conversion

            force_index = [
                i
                for i, s in enumerate(minimization_stats)
                if 'Force two-norm initial, final = 3167.24 0.509175' in s
            ]
            if len(force_index) != 0:
                force_stats = minimization_stats[force_index[0]].split('=')[1]
                force_stats = force_stats.split()
                workflow.results.final_force_maximum = (
                    float(force_stats[-1]) * force_conversion
                )

            minimize_parameters = self.log_parser.get('minimize')
            if not minimize_parameters:
                minimize_parameters = self.log_parser.get('minimize/kk')
            if minimize_parameters:
                workflow.method.optimization_steps_maximum = int(
                    minimize_parameters[0][2]
                )
                workflow.method.convergence_tolerance_force_maximum = (
                    minimize_parameters[0][1] * force_conversion
                )
                workflow.method.convergence_tolerance_energy_difference = (
                    minimize_parameters[0][0] * energy_conversion
                )

            energies = []
            steps = []
            for calc in sec_calc:
                val = calc.get('energy')
                energy = val.get('total') if val else None
                if energy:
                    energies.append(energy.value.magnitude)
                    step = calc.get('step')
                    steps.append(step)
            workflow.results.energies = energies
            workflow.results.steps = steps
            workflow.results.optimization_steps = len(energies) + 1
            self.archive.workflow2 = workflow

        else:
            method, results = {}, {}
            results['finished_normally'] = self.log_parser.get('finished') is not None
            dump_params = sec_lammps.x_lammps_inout_control_dump.split()
            if ',' in dump_params[3]:
                coordinate_save_frequency = dump_params[3].replace(',', '')
            else:
                coordinate_save_frequency = dump_params[3]
            method['coordinate_save_frequency'] = int(coordinate_save_frequency)
            method['n_steps'] = (len(sec_run.system) - 1) * method[
                'coordinate_save_frequency'
            ]
            if (
                'vx' in dump_params[7:]
                or 'vy' in dump_params[7:]
                or 'vz' in dump_params[7:]
            ):
                method['velocity_save_frequency'] = int(dump_params[3])
            if (
                'fx' in dump_params[7:]
                or 'fy' in dump_params[7:]
                or 'fz' in dump_params[7:]
            ):
                method['force_save_frequency'] = int(dump_params[3])
            if sec_lammps.x_lammps_inout_control_thermo is not None:
                method['thermodynamics_save_frequency'] = int(
                    sec_lammps.x_lammps_inout_control_thermo.split()[0]
                )
            # runstyle has 2 options: Velocity-Verlet (default) or rRESPA Multi-Timescale
            runstyle = sec_lammps.x_lammps_inout_control_runstyle
            if runstyle is not None:
                if 'respa' in runstyle.lower:
                    method['integrator_type'] = 'rRESPA_multitimescale'
                else:
                    method['integrator_type'] = 'velocity_verlet'
            else:
                method['integrator_type'] = 'velocity_verlet'
            integration_timestep = self.get_time_step()
            method['integration_timestep'] = integration_timestep

            thermostat_parameters, barostat_parameters = {}, {}
            val = self.log_parser.get('fix', None)
            if val is not None:
                val_remove_duplicates = val if len(val) == 1 else []
                val_tmp = val[0]
                for i in range(1, len(val)):
                    if val[i][:3] == val[i - 1][:3]:
                        val_tmp = val[i]
                    else:
                        val_remove_duplicates.append(val_tmp)
                        val_tmp = val[i]
                val_remove_duplicates.append(val_tmp)
                val = val_remove_duplicates
                for fix in val:
                    fix = [str(i).lower() for i in fix]
                    fix = np.array(fix)
                    fix_group = fix[1]
                    fix_style = fix[2]

                    if fix_group != 'all':  # ignore any complex settings
                        continue

                    reference_temperature = None
                    coupling_constant = None
                    if 'nvt' in fix_style or 'npt' in fix_style:
                        thermostat_parameters['thermostat_type'] = 'nose_hoover'
                        if 'temp' in fix:
                            i_temp = np.where(fix == 'temp')[0]
                            reference_temperature = float(fix[i_temp + 2])  # stop temp
                            coupling_constant = (
                                float(fix[i_temp + 3]) * integration_timestep
                            )
                    elif fix_style == 'temp/berendsen':
                        thermostat_parameters['thermostat_type'] = 'berendsen'
                        i_temp = 3
                        reference_temperature = float(fix[i_temp + 2])  # stop temp
                        coupling_constant = (
                            float(fix[i_temp + 3]) * integration_timestep
                        )
                    elif fix_style == 'temp/csvr':
                        thermostat_parameters['thermostat_type'] = 'velocity_rescaling'
                        i_temp = 3
                        reference_temperature = float(fix[i_temp + 2])  # stop temp
                        coupling_constant = (
                            float(fix[i_temp + 3]) * integration_timestep
                        )
                    elif fix_style == 'temp/csld':
                        thermostat_parameters['thermostat_type'] = (
                            'velocity_rescaling_langevin'
                        )
                        i_temp = 3
                        reference_temperature = float(fix[i_temp + 2])  # stop temp
                        coupling_constant = (
                            float(fix[i_temp + 3]) * integration_timestep
                        )
                    elif fix_style == 'langevin':
                        thermostat_parameters['thermostat_type'] = 'langevin_schneider'
                        i_temp = 3
                        reference_temperature = float(fix[i_temp + 2])  # stop temp
                        coupling_constant = (
                            float(fix[i_temp + 3]) * integration_timestep
                        )
                    elif 'brownian' in fix_style:
                        thermostat_parameters['thermostat_type'] = 'brownian'
                        i_temp = 3
                        reference_temperature = float(fix[i_temp + 2])  # stop temp
                        # coupling_constant =  # ignore multiple coupling parameters
                    thermostat_parameters['reference_temperature'] = (
                        reference_temperature * temperature_conversion
                    )
                    thermostat_parameters['coupling_constant'] = coupling_constant

                    barostat_type = None
                    if 'npt' in fix_style or 'nph' in fix_style:
                        coupling_constant = np.zeros(shape=(3, 3))
                        reference_pressure = np.zeros(shape=(3, 3))
                        compressibility = None
                        barostat_type = 'nose_hoover'
                        if 'iso' in fix:
                            i_baro = np.where(fix == 'iso')[0]
                            barostat_parameters['coupling_type'] = 'isotropic'
                            np.fill_diagonal(coupling_constant, float(fix[i_baro + 3]))
                            np.fill_diagonal(reference_pressure, float(fix[i_baro + 2]))
                        else:
                            barostat_parameters['coupling_type'] = 'anisotropic'
                        if 'x' in fix:
                            i_baro = np.where(fix == 'x')[0]
                            coupling_constant[0, 0] = float(fix[i_baro + 3])
                            reference_pressure[0, 0] = float(fix[i_baro + 2])
                        if 'y' in fix:
                            i_baro = np.where(fix == 'y')[0]
                            coupling_constant[1, 1] = float(fix[i_baro + 3])
                            reference_pressure[1, 1] = float(fix[i_baro + 2])
                        if 'z' in fix:
                            i_baro = np.where(fix == 'z')[0]
                            coupling_constant[2, 2] = float(fix[i_baro + 3])
                            reference_pressure[2, 2] = float(fix[i_baro + 2])
                        if 'xy' in fix:
                            i_baro = np.where(fix == 'xy')[0]
                            coupling_constant[0, 1] = float(fix[i_baro + 3])
                            coupling_constant[1, 0] = float(fix[i_baro + 3])
                            reference_pressure[0, 1] = float(fix[i_baro + 2])
                            reference_pressure[1, 0] = float(fix[i_baro + 2])
                        if 'yz' in fix:
                            i_baro = np.where(fix == 'yz')[0]
                            coupling_constant[1, 2] = float(fix[i_baro + 3])
                            coupling_constant[2, 1] = float(fix[i_baro + 3])
                            reference_pressure[1, 2] = float(fix[i_baro + 2])
                            reference_pressure[2, 1] = float(fix[i_baro + 2])
                        if 'xz' in fix:
                            i_baro = np.where(fix == 'xz')[0]
                            coupling_constant[0, 3] = float(fix[i_baro + 3])
                            coupling_constant[3, 0] = float(fix[i_baro + 3])
                            reference_pressure[0, 3] = float(fix[i_baro + 2])
                            reference_pressure[3, 0] = float(fix[i_baro + 2])
                        barostat_parameters['reference_pressure'] = (
                            reference_pressure * pressure_conversion
                        )  # stop pressure
                        barostat_parameters['coupling_constant'] = (
                            coupling_constant * integration_timestep
                        )
                        barostat_parameters['compressibility'] = compressibility

                    if fix_style == 'press/berendsen':
                        barostat_type = 'berendsen'
                        if 'iso' in fix:
                            i_baro = np.where(fix == 'iso')[0]
                            barostat_parameters['coupling_type'] = 'isotropic'
                            np.fill_diagonal(coupling_constant, float(fix[i_baro + 3]))
                        elif 'aniso' in fix:
                            i_baro = np.where(fix == 'aniso')[0]
                            barostat_parameters['coupling_type'] = 'anisotropic'
                            coupling_constant[:3] += 1.0
                            coupling_constant[:3] *= float(fix[i_baro + 3])
                        else:
                            barostat_parameters['coupling_type'] = 'anisotropic'
                        if 'x' in fix:
                            i_baro = np.where(fix == 'x')[0]
                            coupling_constant[0] = float(fix[i_baro + 3])
                        if 'y' in fix:
                            i_baro = np.where(fix == 'y')[0]
                            coupling_constant[1] = float(fix[i_baro + 3])
                        if 'z' in fix:
                            i_baro = np.where(fix == 'z')[0]
                            coupling_constant[2] = float(fix[i_baro + 3])
                        if 'couple' in fix:
                            i_baro = np.where(fix == 'couple')[0]
                            couple = fix[i_baro]
                            if couple == 'xyz':
                                barostat_parameters['coupling_type'] = 'isotropic'
                            elif couple == 'xy' or couple == 'yz' or couple == 'xz':
                                barostat_parameters['coupling_type'] = 'anisotropic'
                        barostat_parameters['reference_pressure'] = (
                            float(fix[i_baro + 2]) * pressure_conversion
                        )  # stop pressure
                        barostat_parameters['coupling_constant'] = (
                            np.ones(shape=(3, 3))
                            * float(fix[i_baro + 3])
                            * integration_timestep
                        )
                    barostat_parameters['barostat_type'] = barostat_type

            if thermostat_parameters:
                method['thermodynamic_ensemble'] = (
                    'NPT' if barostat_type == 'nose_hoover' else 'NVT'
                )
            elif barostat_type == 'nose_hoover':
                method['thermodynamic_ensemble'] = 'NPH'
            else:
                method['thermodynamic_ensemble'] = 'NVE'

            method['thermostat_parameters'] = thermostat_parameters
            method['barostat_parameters'] = barostat_parameters

            self.parse_md_workflow(dict(method=method, results=results))

    def parse_system(self):
        sec_run = self.archive.run[-1]

        n_traj = self.traj_parsers.eval('n_frames')
        if n_traj is None:
            return

        self.n_atoms = [self.traj_parsers.eval('get_n_atoms', n) for n in range(n_traj)]
        self.trajectory_steps = [
            step
            for n in range(n_traj)
            if (step := self.traj_parsers.eval('get_step', n)) is not None
        ]

        units = self.log_parser.units

        def apply_unit(value, unit):
            if not hasattr(value, 'units'):
                value = value * units.get(unit, 1)
            return value

        def get_composition(children_names):
            children_count_tup = np.unique(children_names, return_counts=True)
            formula = ''.join(
                [f'{name}({count})' for name, count in zip(*children_count_tup)]
            )
            return formula

        for step in self.trajectory_steps:
            traj_n = self._trajectory_steps.index(step)
            lattice_vectors = self.traj_parsers.eval('get_lattice_vectors', traj_n)
            if lattice_vectors is not None:
                lattice_vectors = apply_unit(lattice_vectors, 'distance')
            velocities = self.traj_parsers.eval('get_velocities', traj_n)
            if velocities is not None:
                velocities = apply_unit(velocities, 'velocity')
            bond_list = []
            if traj_n == 0:  # TODO add references to the bond list for other steps
                bond_list = get_bond_list_from_model_contributions(
                    sec_run, method_index=-1, model_index=-1
                )
            self.parse_trajectory_step(
                {
                    'atoms': {
                        'n_atoms': self.traj_parsers.eval('get_n_atoms', traj_n),
                        'lattice_vectors': lattice_vectors,
                        'periodic': self.traj_parsers.eval('get_pbc', traj_n),
                        'positions': apply_unit(
                            self.traj_parsers.eval('get_positions', traj_n), 'distance'
                        ),
                        'labels': self.traj_parsers.eval('get_atom_labels', traj_n),
                        'velocities': velocities,
                        'bond_list': bond_list if bond_list else None,
                    }
                }
            )

        if not sec_run.system:
            return

        sec_system = sec_run.system[-1]
        # parse atomsgroup (moltypes --> molecules --> residues)
        atoms_info = self._mdanalysistraj_parser.get('atoms_info', None)
        if atoms_info is None:
            atoms_info = self.traj_parsers.eval('atoms_info')
            if isinstance(atoms_info, list):
                atoms_info = (
                    atoms_info[0] if atoms_info else None
                )  # using info from the initial frame
        if atoms_info is not None:
            atoms_moltypes = np.array(atoms_info.get('moltypes', []))
            atoms_molnums = np.array(atoms_info.get('molnums', []))
            atoms_resids = np.array(atoms_info.get('resids', []))
            atoms_elements = np.array(atoms_info.get('elements', ['X'] * self.n_atoms))
            atoms_types = np.array(atoms_info.get('types', []))
            atom_labels = sec_system.atoms.get('labels')
            if 'X' in atoms_elements:
                atoms_elements = (
                    np.array(atom_labels)
                    if atom_labels and 'X' not in atom_labels
                    else atoms_types
                )
            atoms_resnames = np.array(atoms_info.get('resnames', []))
            moltypes = np.unique(atoms_moltypes)
            for i_moltype, moltype in enumerate(moltypes):
                # Only add atomsgroup for initial system for now
                sec_molecule_group = AtomsGroup()
                sec_run.system[0].atoms_group.append(sec_molecule_group)
                sec_molecule_group.label = f'group_{moltype}'
                sec_molecule_group.type = 'molecule_group'
                sec_molecule_group.index = i_moltype
                sec_molecule_group.atom_indices = np.where(atoms_moltypes == moltype)[0]
                sec_molecule_group.n_atoms = len(sec_molecule_group.atom_indices)
                sec_molecule_group.is_molecule = False
                # mol_nums is the molecule identifier for each atom
                mol_nums = atoms_molnums[sec_molecule_group.atom_indices]
                moltype_count = np.unique(mol_nums).shape[0]
                sec_molecule_group.composition_formula = f'{moltype}({moltype_count})'

                molecules = atoms_molnums
                for i_molecule, molecule in enumerate(
                    np.unique(molecules[sec_molecule_group.atom_indices])
                ):
                    sec_molecule = AtomsGroup()
                    sec_molecule_group.atoms_group.append(sec_molecule)
                    sec_molecule.index = i_molecule
                    sec_molecule.atom_indices = np.where(molecules == molecule)[0]
                    sec_molecule.n_atoms = len(sec_molecule.atom_indices)
                    # use first particle to get the moltype
                    # not sure why but this value is being cast to int, cast back to str
                    sec_molecule.label = str(
                        atoms_moltypes[sec_molecule.atom_indices[0]]
                    )
                    sec_molecule.type = 'molecule'
                    sec_molecule.is_molecule = True

                    mol_resids = np.unique(atoms_resids[sec_molecule.atom_indices])
                    n_res = mol_resids.shape[0]
                    if n_res == 1:
                        elements = atoms_elements[sec_molecule.atom_indices]
                        sec_molecule.composition_formula = get_composition(elements)
                    else:
                        mol_resnames = atoms_resnames[sec_molecule.atom_indices]
                        restypes = np.unique(mol_resnames)
                        for i_restype, restype in enumerate(restypes):
                            sec_monomer_group = AtomsGroup()
                            sec_molecule.atoms_group.append(sec_monomer_group)
                            restype_indices = np.where(atoms_resnames == restype)[0]
                            sec_monomer_group.label = f'group_{restype}'
                            sec_monomer_group.type = 'monomer_group'
                            sec_monomer_group.index = i_restype
                            sec_monomer_group.atom_indices = np.intersect1d(
                                restype_indices, sec_molecule.atom_indices
                            )
                            sec_monomer_group.n_atoms = len(
                                sec_monomer_group.atom_indices
                            )
                            sec_monomer_group.is_molecule = False

                            restype_resids = np.unique(
                                atoms_resids[sec_monomer_group.atom_indices]
                            )
                            restype_count = restype_resids.shape[0]
                            sec_monomer_group.composition_formula = (
                                f'{restype}({restype_count})'
                            )
                            for i_res, res_id in enumerate(restype_resids):
                                sec_residue = AtomsGroup()
                                sec_monomer_group.atoms_group.append(sec_residue)
                                sec_residue.index = i_res
                                atom_indices = np.where(atoms_resids == res_id)[0]
                                sec_residue.atom_indices = np.intersect1d(
                                    atom_indices, sec_monomer_group.atom_indices
                                )
                                sec_residue.n_atoms = len(sec_residue.atom_indices)
                                sec_residue.label = str(restype)
                                sec_residue.type = 'monomer'
                                sec_residue.is_molecule = False
                                elements = atoms_elements[sec_residue.atom_indices]
                                sec_residue.composition_formula = get_composition(
                                    elements
                                )

                        names = atoms_resnames[sec_molecule.atom_indices]
                        ids = atoms_resids[sec_molecule.atom_indices]
                        # filter for the first instance of each residue, as to not overcount
                        __, ids_count = np.unique(ids, return_counts=True)
                        # get the index of the first atom of each residue
                        ids_firstatom = np.cumsum(ids_count)[:-1]
                        # add the 0th index manually
                        ids_firstatom = np.insert(ids_firstatom, 0, 0)
                        names_firstatom = names[ids_firstatom]
                        sec_molecule.composition_formula = get_composition(
                            names_firstatom
                        )

    def parse_method(self):
        sec_run = self.archive.run[-1]

        if self.traj_parsers[0].mainfile is None or self.data_parser.mainfile is None:
            return

        if self.traj_parsers.eval('n_frames') is None:
            return

        sec_method = Method()
        sec_run.method.append(sec_method)
        sec_force_field = ForceField()
        sec_method.force_field = sec_force_field
        sec_model = Model()
        sec_force_field.model.append(sec_model)

        # Old parsing of method with text parser
        masses = self.data_parser.get('Masses', None)
        self.traj_parsers[0].masses = masses
        # @Landinesa: we should be able to set the atom masses with the TrajParser, but I don't quite understand how to use this.
        # Can you add the implementation here, and then we can make the MDA implementation below as a backup?
        # Can you also get the charges somehow?

        # parse method with MDAnalysis (should be a backup for the charges and masses...but the interactions are most easily read from the MDA universe right now)
        n_atoms = self.traj_parsers.eval('get_n_atoms', 0)
        if n_atoms is not None:
            atoms_info = self._mdanalysistraj_parser.get('atoms_info', None)
            for n in range(n_atoms):
                sec_atom = AtomParameters()
                sec_method.atom_parameters.append(sec_atom)
                sec_atom.charge = atoms_info.get('charges', [None] * (n + 1))[n]
                sec_atom.mass = atoms_info.get('masses', [None] * (n + 1))[n]

        # TODO address case types are numbered instead of giving atom labels (fix tests accordingly)
        interactions = self._mdanalysistraj_parser.get_interactions()
        # for interaction in interactions:
        #     for key, val in interaction.items():
        #         quantity_def = Interaction.m_def.all_quantities.get(key)
        #         if quantity_def and quantity_def.shape:
        #             # TODO reshape properly
        #             interaction[key] = [val]
        self.parse_interactions(interactions, sec_model)

        # Force Calculation Parameters
        sec_force_calculations = ForceCalculations()
        sec_force_field.force_calculations = sec_force_calculations
        for pairstyle in self.log_parser.get('pair_style', []):
            pairstyle_args = pairstyle[1:]
            pairstyle = pairstyle[0].lower()
            if (
                'lj' in pairstyle and 'coul' not in pairstyle
            ):  # only cover the simplest case
                sec_force_calculations.vdw_cutoff = (
                    float(pairstyle_args[-1]) * ureg.nanometer
                )
            if 'coul' in pairstyle:
                if 'streitz' in pairstyle:
                    cutoff = float(pairstyle_args[0])
                else:
                    cutoff = float(pairstyle_args[-1])
                sec_force_calculations.coulomb_cutoff = cutoff * ureg.nanometer
            val = self.log_parser.get('kspace_style', None)
            if val is not None:
                kspacestyle = val[0][0].lower()
                if 'ewald' in kspacestyle:
                    sec_force_calculations.coulomb_type = 'ewald'
                elif 'pppm' in kspacestyle:
                    sec_force_calculations.coulomb_type = (
                        'particle_particle_particle_mesh'
                    )
                elif 'msm' in kspacestyle:
                    sec_force_calculations.coulomb_type = 'multilevel_summation'

        sec_neighbor_searching = NeighborSearching()
        sec_force_calculations.neighbor_searching = sec_neighbor_searching
        val = self.log_parser.get('neighbor', None)
        if val is not None:
            neighbor = val[0][0]  # just use the first instance for now
            vdw_cutoff = sec_force_calculations.vdw_cutoff
            if vdw_cutoff is not None:
                sec_neighbor_searching.neighbor_update_cutoff = (
                    float(neighbor) * ureg.nanometer
                )
                sec_neighbor_searching.neighbor_update_cutoff += vdw_cutoff
        val = self.log_parser.get('neigh_modify', None)
        if val is not None:
            neighmodify = val[0]  # just use the first instace for now
            neighmodify = np.array([str(i).lower() for i in neighmodify])
            if 'every' in neighmodify:
                index = np.where(neighmodify == 'every')[0]
                sec_neighbor_searching.neighbor_update_frequency = int(
                    neighmodify[index + 1]
                )

    def parse_input(self):
        sec_run = self.archive.run[-1]
        sec_input_output_files = x_lammps_section_input_output_files()
        sec_run.x_lammps_section_input_output_files.append(sec_input_output_files)

        if self.data_parser.mainfile is not None:
            sec_input_output_files.x_lammps_inout_file_data = os.path.basename(
                self.data_parser.mainfile
            )

        if self.traj_parsers[0].mainfile is not None:
            sec_input_output_files.x_lammps_inout_file_trajectory = os.path.basename(
                self.traj_parsers[0].mainfile
            )

        sec_control_parameters = x_lammps_section_control_parameters()
        sec_run.x_lammps_section_control_parameters.append(sec_control_parameters)
        keys = self.log_parser._commands
        for key in keys:
            val = self.log_parser.get(key, None)
            if val is None:
                continue
            val = val[0] if len(val) == 1 else val
            key = (
                'x_lammps_inout_control_%s'
                % key.replace('_', '').replace('/', '').lower()
            )
            if hasattr(sec_control_parameters, key):
                if isinstance(val, list):
                    val = ' '.join([str(v) for v in val])
                setattr(sec_control_parameters, key, str(val))

    def write_to_archive(self):
        self.log_parser.mainfile = self.mainfile
        self.log_parser.logger = self.logger
        self._traj_parser.logger = self.logger
        self._mdanalysistraj_parser.logger = self.logger
        self._xyztraj_parser.logger = self.logger
        self.data_parser.logger = self.logger
        # auxilliary log parser for thermo data
        self.aux_log_parser.logger = self.logger
        self.log_parser._units = None
        self._traj_parser._chemical_symbols = None

        sec_run = Run()
        self.archive.run.append(sec_run)

        # parse basic
        sec_run.program = Program(
            name='LAMMPS', version=self.log_parser.get('program_version', '')
        )

        # parse data file associated with calculation
        data_files = self.log_parser.get_data_files()
        if len(data_files) > 1:
            self.logger.warning('Multiple data files are specified')
        if data_files:
            self.data_parser.mainfile = data_files[0]

        # parse trajectorty file associated with calculation
        traj_files = self.log_parser.get_traj_files()
        if len(traj_files) > 1:
            self.logger.warning('Multiple traj files are specified')

        parsers = []
        for n, traj_file in enumerate(traj_files):
            # parser initialization for each traj file cannot be avoided as there are
            # cases where traj files can share the same parser
            file_type = self.log_parser.get(
                'dump', [[1, 'all', traj_file[-3:]]] * (n + 1)
            )[n][2]
            if file_type == 'dcd' and data_files:
                traj_parser = MDAnalysisParser(topology_format='DATA', format='DCD')
                traj_parser.mainfile = data_files[0]
                traj_parser.auxilliary_files = [traj_file]
                self._mdanalysistraj_parser = traj_parser
            elif file_type == 'xyz' and data_files:
                traj_parser = MDAnalysisParser(topology_format='DATA', format='XYZ')
                traj_parser.mainfile = data_files[0]
                traj_parser.auxilliary_files = [traj_file]
                self._mdanalysistraj_parser = traj_parser
            elif file_type == 'custom' and data_files:
                custom_options = self.log_parser.get('dump')[n][5:]
                custom_options = [
                    option.replace('xu', 'x') for option in custom_options
                ]
                custom_options = [
                    option.replace('yu', 'y') for option in custom_options
                ]
                custom_options = [
                    option.replace('zu', 'z') for option in custom_options
                ]
                custom_options = ' '.join(custom_options)
                traj_parser = MDAnalysisParser(
                    topology_format='DATA',
                    format='LAMMPSDUMP',
                    atom_style=custom_options,
                )
                if data_files:
                    traj_parser.mainfile = data_files[0]
                traj_parser.auxilliary_files = [traj_file]
                # try to check if MDAnalysis can construct the universe or at least parse
                # the atoms, otherwise will fall back to TrajParser
                if traj_parser.universe is None or 'X' in traj_parser.get(
                    'atoms_info', {}
                ).get('names', []):
                    # mda necessary to calculate rdf and atomsgroup
                    if n == 0:
                        self._mdanalysistraj_parser = traj_parser
                    traj_parser = TrajParser()
                    traj_parser.mainfile = traj_file
            else:
                traj_parser = TrajParser()
                traj_parser.mainfile = traj_file
                # TODO provide support for other file types
            parsers.append(traj_parser)

        self.traj_parsers = TrajParsers(parsers)
        if self.traj_parsers[0] is None:
            return

        # parse data from auxiliary log file
        if self.log_parser.get('log') is not None:
            self.aux_log_parser.mainfile = os.path.join(
                self.log_parser.maindir, self.log_parser.get('log')[0]
            )
            # we assign units here which is read from log parser
            self.aux_log_parser._units = self.log_parser.units

        self.parse_method()

        self.parse_system()

        # include input controls from log file
        self.parse_input()

        # parse thermodynamic data from log file
        self.parse_thermodynamic_data()

        self.parse_workflow()

        self._mdanalysistraj_parser.close()
        for parser in parsers:
            parser.close()
