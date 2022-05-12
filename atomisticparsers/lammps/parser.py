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
import logging
from ase import data as asedata

from nomad.units import ureg

from nomad.parsing.file_parser import Quantity, TextParser
from nomad.datamodel.metainfo.simulation.run import Run, Program
from nomad.datamodel.metainfo.simulation.method import (
    ForceField, Method, Interaction, Model
)
from nomad.datamodel.metainfo.simulation.system import (
    System, Atoms, AtomsGroup
)
from nomad.datamodel.metainfo.simulation.calculation import (
    Calculation, Energy, EnergyEntry, Forces, ForcesEntry
)
from nomad.datamodel.metainfo.workflow import (
    DiffusionConstantValues, MeanSquaredDisplacement, MeanSquaredDisplacementValues,
    RadialDistributionFunction, RadialDistributionFunctionValues,
    Workflow, MolecularDynamics
)
from .metainfo.lammps import x_lammps_section_input_output_files, x_lammps_section_control_parameters
from atomisticparsers.utils import MDAnalysisParser


def get_unit(units_type, property_type=None, dimension=3):
    mole = 6.022140857e+23

    units_type = units_type.lower()
    if units_type == 'real':
        units = dict(
            mass=ureg.g / mole, distance=ureg.angstrom, time=ureg.fs,
            energy=ureg.J * 4184.0 / mole, velocity=ureg.angstrom / ureg.fs,
            force=ureg.J * 4184.0 / ureg.angstrom / mole, torque=ureg.J * 4184.0 / mole,
            temperature=ureg.K, pressure=ureg.atm, dynamic_viscosity=ureg.poise, charge=ureg.elementary_charge,
            dipole=ureg.elementary_charge * ureg.angstrom, electric_field=ureg.V / ureg.angstrom,
            density=ureg.g / ureg.cm ** dimension)

    elif units_type == 'metal':
        units = dict(
            mass=ureg.g / mole, distance=ureg.angstrom, time=ureg.ps,
            energy=ureg.eV, velocity=ureg.angstrom / ureg.ps, force=ureg.eV / ureg.angstrom, torque=ureg.eV,
            temperature=ureg.K, pressure=ureg.bar, dynamic_viscosity=ureg.poise, charge=ureg.elementary_charge,
            dipole=ureg.elementary_charge * ureg.angstrom, electric_field=ureg.V / ureg.angstrom,
            density=ureg.g / ureg.cm ** dimension)

    elif units_type == 'si':
        units = dict(
            mass=ureg.kg, distance=ureg.m, time=ureg.s, energy=ureg.J, velocity=ureg.m / ureg.s, force=ureg.N,
            torque=ureg.N * ureg.m, temperature=ureg.K, pressure=ureg.Pa, dynamic_viscosity=ureg.Pa * ureg.s,
            charge=ureg.C, dipole=ureg.C * ureg.m, electric_field=ureg.V / ureg.m, density=ureg.kg / ureg.m ** dimension)

    elif units_type == 'cgs':
        units = dict(
            mass=ureg.g, distance=ureg.cm, time=ureg.s, energy=ureg.erg, velocity=ureg.cm / ureg.s, force=ureg.dyne,
            torque=ureg.dyne * ureg.cm, temperature=ureg.K, pressure=ureg.dyne / ureg. cm ** 2, dynamic_viscosity=ureg.poise,
            charge=ureg.esu, dipole=ureg.esu * ureg.cm, electric_field=ureg.dyne / ureg.esu,
            density=ureg.g / ureg.cm ** dimension)

    elif units_type == 'electron':
        units = dict(
            mass=ureg.amu, distance=ureg.bohr, time=ureg.fs, energy=ureg.hartree,
            velocity=ureg.bohr / ureg.atomic_unit_of_time, force=ureg.hartree / ureg.bohr, temperature=ureg.K,
            pressure=ureg.Pa, charge=ureg.elementary_charge, dipole=ureg.debye, electric_field=ureg.V / ureg.cm)

    elif units_type == 'micro':
        units = dict(
            mass=ureg.pg, distance=ureg.microm, time=ureg.micros, energy=ureg.pg * ureg.microm ** 2 / ureg.micros ** 2,
            velocity=ureg.microm / ureg.micros, force=ureg.pg * ureg.microm / ureg.micros ** 2, torque=ureg.pg * ureg.microm ** 2 / ureg.micros ** 2,
            temperature=ureg.K, pressure=ureg.pg / (ureg.microm * ureg.micros ** 2), dynamic_viscosity=ureg.pg / (ureg.microm * ureg.micros),
            charge=ureg.pC, dipole=ureg.pC * ureg.microm, electric_field=ureg.V / ureg.microm,
            density=ureg.pg / ureg.microm ** dimension)

    elif units_type == 'nano':
        units = dict(
            mass=ureg.ag, distance=ureg.nm, time=ureg.ns, energy=ureg.ag * ureg.nm ** 2 / ureg.ns ** 2, velocity=ureg.nm / ureg.ns,
            force=ureg.ag * ureg.nm / ureg.ns ** 2, torque=ureg.ag * ureg.nm ** 2 / ureg.ns ** 2, temperature=ureg.K, pressure=ureg.ag / (ureg.nm * ureg.ns ** 2),
            dynamic_viscosity=ureg.ag / (ureg.nm * ureg.ns), charge=ureg.elementary_charge,
            dipole=ureg.elementary_charge * ureg.nm, electric_field=ureg.V / ureg.nm, density=ureg.ag / ureg.nm ** dimension)

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
            'atoms', 'bonds', 'angles', 'dihedrals', 'impropers', 'atom types', 'bond types',
            'angle types', 'dihedral types', 'improper types', 'extra bond per atom',
            'extra/bond/per/atom', 'extra angle per atom', 'extra/angle/per/atom',
            'extra dihedral per atom', 'extra/dihedral/per/atom', 'extra improper per atom',
            'extra/improper/per/atom', 'extra special per atom', 'extra/special/per/atom',
            'ellipsoids', 'lines', 'triangles', 'bodies', 'xlo xhi', 'ylo yhi', 'zlo zhi',
            'xy xz yz']
        self._sections = [
            'Atoms', 'Velocities', 'Masses', 'Ellipsoids', 'Lines', 'Triangles', 'Bodies',
            'Bonds', 'Angles', 'Dihedrals', 'Impropers', 'Pair Coeffs', 'PairIJ Coeffs',
            'Bond Coeffs', 'Angle Coeffs', 'Dihedral Coeffs', 'Improper Coeffs',
            'BondBond Coeffs', 'BondAngle Coeffs', 'MiddleBondTorsion Coeffs',
            'EndBondTorsion Coeffs', 'AngleTorsion Coeffs', 'AngleAngleTorsion Coeffs',
            'BondBond13 Coeffs', 'AngleAngle Coeffs']
        self._interactions = [
            section for section in self._sections if section.endswith('Coeffs')]
        super().__init__(None)

    def init_quantities(self):
        self._quantities = []
        for header in self._headers:
            self._quantities.append(Quantity(
                header, r'\s*([\+\-eE\d\. ]+)\s*%s\s*\n' % header, comment='#', repeats=True))

        def get_section_value(val):
            val = val.split('\n')
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

        for section in self._sections:
            self._quantities.append(
                Quantity(
                    section, r'\s*%s\s*(#*\s*[\s\S]*?\n)\n*([\deE\-\+\.\s]+)\n' % section,
                    str_operation=get_section_value, repeats=True))

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
            masses=np.array(asedata.atomic_masses), symbols=asedata.chemical_symbols)
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
                'time_step', r'\s*ITEM:\s*TIMESTEP\s*\n\s*(\d+)\s*\n', comment='#',
                repeats=True),
            Quantity(
                'n_atoms', r'\s*ITEM:\s*NUMBER OF ATOMS\s*\n\s*(\d+)\s*\n', comment='#',
                repeats=True),
            Quantity(
                'pbc_cell', r'\s*ITEM: BOX BOUNDS\s*([\s\w]+)([\+\-\d\.eE\s]+)\n',
                str_operation=get_pbc_cell, comment='#', repeats=True),
            Quantity(
                'atoms_info', r's*ITEM:\s*ATOMS\s*([ \w]+\n)*?([\+\-eE\d\.\n ]+)',
                str_operation=get_atoms_info, comment='#', repeats=True)
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
                symbol_idx = np.argmin(abs(self._reference_masses['masses'] - masses[i][1]))
                self._chemical_symbols[masses[i][0]] = self._reference_masses['symbols'][symbol_idx]

    def get_atom_labels(self, idx):
        atoms_info = self.get('atoms_info')
        if atoms_info is None:
            return

        atoms_type = atoms_info[idx].get('type')
        if atoms_type is None:
            return

        if self._chemical_symbols is None:
            return

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
            positions = np.array([atoms_info['xs'], atoms_info['ys'], atoms_info['zs']]).T
            positions = positions * np.linalg.norm(cell, axis=1) + np.amin(cell, axis=1)

        elif 'xu' in atoms_info and 'yu' in atoms_info and 'zu' in atoms_info:
            positions = np.array([atoms_info['xu'], atoms_info['yu'], atoms_info['zu']]).T

        elif 'xsu' in atoms_info and 'ysu' in atoms_info and 'zsu' in atoms_info:
            if cell is None:
                return
            positions = np.array([atoms_info['xsu'], atoms_info['ysu'], atoms_info['zsu']]).T
            positions = positions * np.linalg.norm(cell, axis=1) + np.amin(cell, axis=1)

        elif 'x' in atoms_info and 'y' in atoms_info and 'z' in atoms_info:
            positions = np.array([atoms_info['x'], atoms_info['y'], atoms_info['z']]).T
            if 'ix' in atoms_info and 'iy' in atoms_info and 'iz' in atoms_info:
                if cell is None:
                    return
                positions_img = np.array([
                    atoms_info['ix'], atoms_info['iy'], atoms_info['iz']]).T

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
                'atoms_info', r'((?:\d+|[A-Z][a-z]?) [\s\S]+?)(?:\s\d+\n|\Z)',
                str_operation=get_atoms_info, comment='#', repeats=True)
        ]


class LogParser(TextParser):
    def __init__(self):
        self._commands = [
            'angle_coeff', 'angle_style', 'atom_modify', 'atom_style', 'balance',
            'bond_coeff', 'bond_style', 'bond_write', 'boundary', 'change_box', 'clear',
            'comm_modify', 'comm_style', 'compute', 'compute_modify', 'create_atoms',
            'create_bonds', 'create_box', 'delete_bonds', 'dielectric', 'dihedral_coeff',
            'dihedral_style', 'dimension', 'displace_atoms', 'dump', 'dump_modify',
            'dynamical_matrix', 'echo', 'fix', 'fix_modify', 'group', 'group2ndx',
            'ndx2group', 'hyper', 'if', 'improper_coeff', 'improper_style', 'include',
            'info', 'jump', 'kim_init', 'kim_interactions', 'kim_query', 'kim_param',
            'kim_property', 'kspace_modify', 'kspace_style', 'label', 'lattice', 'log',
            'mass', 'message', 'min_modify', 'min_style', 'minimize', 'minimize/kk',
            'molecule', 'neb', 'neb/spin', 'neigh_modify', 'neighbor', 'newton', 'next',
            'package', 'pair_coeff', 'pair_modify', 'pair_style', 'pair_write',
            'partition', 'prd', 'print', 'processors', 'quit', 'read_data', 'read_dump',
            'read_restart', 'region', 'replicate', 'rerun', 'reset_atom_ids',
            'reset_mol_ids', 'reset_timestep', 'restart', 'run', 'run_style', 'server',
            'set', 'shell', 'special_bonds', 'suffix', 'tad', 'temper/grem', 'temper/npt',
            'thermo', 'thermo_modify', 'thermo_style', 'third_order', 'timer', 'timestep',
            'uncompute', 'undump', 'unfix', 'units', 'variable', 'velocity', 'write_coeff',
            'write_data', 'write_dump', 'write_restart']
        self._interactions = [
            'atom', 'pair', 'bond', 'angle', 'dihedral', 'improper', 'kspace']
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
                name, r'\n\s*%s\s+([\w\. \/\#\-]+)(\&\n[\w\. \/\#\-]*)*' % name,
                str_operation=str_op, comment='#', repeats=True) for name in self._commands]

        self._quantities.append(Quantity(
            'program_version', r'\s*LAMMPS\s*\(([\w ]+)\)\n', dtype=str, repeats=False,
            flatten=False)
        )

        self._quantities.append(Quantity(
            'finished', r'\s*Dangerous builds\s*=\s*(\d+)', repeats=False)
        )

        def str_to_thermo(val):
            res = {}
            if val.count('Step') > 1:
                val = val.replace('--', '').replace('=', '').replace('(sec)', '').split()
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

        self._quantities.append(Quantity(
            'thermo_data', r'\s*\-*(\s*Step\s*[\-\s\w\.\=\(\)]*[ \-\.\d\n]+)Loop',
            str_operation=str_to_thermo, repeats=False, convert=False)
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
        for key, val in thermo_data.items():
            low_key = key.lower()
            if low_key.startswith('e_') or low_key.endswith('eng'):
                thermo_data[key] = val * self.units.get('energy', 1)
            elif low_key == 'press':
                thermo_data[key] = val * self.units.get('pressure', 1)
            elif low_key == 'temp':
                thermo_data[key] = val * self.units.get('temperature', 1)

        return thermo_data

    def get_traj_files(self):
        dump = self.get('dump')
        if dump is None:
            self.logger.warn('Trajectory not specified in directory, will scan.')
            # TODO improve matching of traj file
            traj_files = os.listdir(self.maindir)
            traj_files = [f for f in traj_files if f.endswith('trj') or f.endswith('xyz')]
            # further eliminate
            if len(traj_files) > 1:
                prefix = os.path.basename(self.mainfile).rsplit('.', 1)[0]
                traj_files = [f for f in traj_files if prefix in f]
        else:
            traj_files = []
            if type(dump[0]) in [str, int]:
                dump = [dump]
            traj_files = [d[4] for d in dump]

        return [os.path.join(self.maindir, f) for f in traj_files]

    def get_data_files(self):
        read_data = self.get('read_data')
        if read_data is None or 'CPU' in read_data:
            self.logger.warn('Data file not specified in directory, will scan.')
            # TODO improve matching of data file
            data_files = os.listdir(self.maindir)
            data_files = [f for f in data_files if f.endswith('data') or f.startswith('data')]
            if len(data_files) > 1:
                prefix = os.path.basename(self.mainfile).rsplit('.', 1)[1]  # JFR- @Alvin Please check this - changed from [0] to [1] for case that filename is leading with log
                data_files = [f for f in data_files if prefix in f]
        else:
            data_files = read_data

        return [os.path.join(self.maindir, f) for f in data_files]

    def get_pbc(self):
        pbc = self.get('boundary', ['p', 'p', 'p'])
        return [v == 'p' for v in pbc]

    def get_sampling_method(self):
        fix_style = self.get('fix', [[''] * 3])[0][2]

        sampling_method = 'langevin_dynamics' if 'langevin' in fix_style else 'molecular_dynamics'
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
            self.logger.warn('Fix style not supported', data=dict(style=fix_style))

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
                    coeff = self.get("%s_coeff" % interaction)
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

    def eval(self, key, *args):
        for parser in self._parsers:
            val = getattr(parser, key)(*args) if args else getattr(parser, key)
            if val is not None:
                return val


class LammpsParser:
    def __init__(self):
        self.log_parser = LogParser()
        self._traj_parser = TrajParser()
        self._xyztraj_parser = XYZTrajParser()
        self._mdanalysistraj_parser = MDAnalysisParser(topology_format='DATA', format='LAMMPSDUMP')
        self.data_parser = DataParser()
        self.aux_log_parser = LogParser()
        self._energy_mapping = {
            'e_pair': 'pair', 'e_bond': 'bond', 'e_angle': 'angle', 'e_dihed': 'dihedral',
            'e_impro': 'improper', 'e_coul': 'coulomb', 'e_vdwl': 'van der Waals',
            'e_mol': 'molecular', 'e_long': 'kspace long range',
            'e_tail': 'van der Waals long range', 'kineng': 'kinetic', 'poteng': 'potential'}
        self._frame_rate = None
        # max cumulative number of atoms for all parsed trajectories to calculate sampling rate
        self._cum_max_atoms = 1000000

    @property
    def frame_rate(self):
        if self._frame_rate is None:
            n_frames = self.traj_parsers.eval('n_frames')
            n_atoms = [self.traj_parsers.eval('get_n_atoms', n) for n in range(n_frames)]
            n_atoms = max(n_atoms) if n_atoms else 0

            if n_atoms == 0 or n_frames == 0:
                self._frame_rate = 1
            else:
                cum_atoms = n_atoms * n_frames
                self._frame_rate = 1 if cum_atoms <= self._cum_max_atoms else cum_atoms // self._cum_max_atoms
        return self._frame_rate

    def get_time_step(self):
        time_unit = self.log_parser.units.get('time', None)
        time_step = self.log_parser.get('timestep', [0], unit=time_unit)[0]
        return time_step

    def parse_thermodynamic_data(self):
        time_step = self.get_time_step()
        thermo_data = self.log_parser.get_thermodynamic_data()
        if thermo_data is None:
            thermo_data = self.aux_log_parser.get_thermodynamic_data()
        if not thermo_data:
            return

        sec_run = self.archive.run[-1]
        sec_sccs = sec_run.calculation

        n_thermo = len(thermo_data.get('Step', []))
        n_steps = len([n for n in range(n_thermo) if n % self.frame_rate == 0])
        create_scc = True
        if sec_sccs:
            if len(sec_sccs) != n_steps:
                self.logger.warn(
                    '''Mismatch in number of calculations and number of property
                    evaluations!, will create new sections''',
                    data=dict(n_calculations=len(sec_sccs), n_evaluations=n_steps))

            else:
                create_scc = False
        for n in range(n_thermo):
            if (n % self.frame_rate) > 0:
                continue

            if create_scc:
                sec_scc = sec_run.m_create(Calculation)
            else:
                sec_scc = sec_sccs[n // self.frame_rate]

            sec_energy = sec_scc.m_create(Energy)
            for key, val in thermo_data.items():
                key = key.lower()
                if key in self._energy_mapping:
                    sec_energy.contributions.append(
                        EnergyEntry(kind=self._energy_mapping[key], value=val[n]))
                elif key == 'toteng':
                    sec_energy.current = EnergyEntry(value=val[n])
                    sec_energy.total = EnergyEntry(value=val[n])
                elif key == 'press':
                    sec_scc.pressure = val[n]
                elif key == 'temp':
                    sec_scc.temperature = val[n]
                elif key == 'step':
                    sec_scc.step = int(val[n])
                    sec_scc.time = sec_scc.step * time_step
                elif key == 'cpu':
                    sec_scc.time_calculation = float(val[n])

    def parse_workflow(self):
        sec_workflow = self.archive.m_create(Workflow)
        sec_md = sec_workflow.m_create(MolecularDynamics)

        run_style = self.log_parser.get('run_style', ['verlet'])[0]
        run = self.log_parser.get('run', [0])[0]

        time_step = self.get_time_step()
        sampling_method, ensemble_type = self.log_parser.get_sampling_method()

        sec_workflow.type = sampling_method
        sec_md.x_lammps_integrator_type = run_style
        sec_md.x_lammps_number_of_steps_requested = run
        sec_md.x_lammps_integrator_dt = time_step
        sec_md.time_step = time_step
        sec_md.ensemble_type = ensemble_type.upper()

        thermo_settings = self.log_parser.get_thermostat_settings()
        target_T = thermo_settings.get('target_T', None)
        if target_T is not None:
            sec_md.x_lammps_thermostat_target_temperature = target_T
        thermostat_tau = thermo_settings.get('thermostat_tau', None)
        if thermostat_tau is not None:
            sec_md.x_lammps_thermostat_tau = thermostat_tau
        target_P = thermo_settings.get('target_P', None)
        if target_P is not None:
            sec_md.x_lammps_barostat_target_pressure = target_P
        barostat_tau = thermo_settings.get('barostat_P', None)
        if barostat_tau is not None:
            sec_md.x_lammps_barostat_tau = barostat_tau
        langevin_gamma = thermo_settings.get('langevin_gamma', None)
        if langevin_gamma is not None:
            sec_md.x_lammps_langevin_gamma = langevin_gamma

        sec_md.finished_normally = self.log_parser.get('finished') is not None
        sec_md.with_trajectory = self.traj_parsers.eval('with_trajectory')

    def parse_system(self):
        sec_run = self.archive.run[-1]

        n_frames = self.traj_parsers.eval('n_frames')
        units = self.log_parser.units

        def apply_unit(value, unit):
            if not hasattr(value, 'units'):
                value = value * units.get(unit, 1)
            return value

        def get_composition(children_names):
            children_count_tup = np.unique(children_names, return_counts=True)
            formula = ''.join([f'{name}({count})' for name, count in zip(*children_count_tup)])
            return formula

        for i in range(n_frames):
            if (i % self.frame_rate) > 0:
                continue

            sec_system = sec_run.m_create(System)
            sec_atoms = sec_system.m_create(Atoms)
            sec_atoms.n_atoms = self.traj_parsers.eval('get_n_atoms', i)
            lattice_vectors = self.traj_parsers.eval('get_lattice_vectors', i)
            if lattice_vectors is not None:
                sec_atoms.lattice_vectors = apply_unit(lattice_vectors, 'distance')
            sec_atoms.periodic = self.traj_parsers.eval('get_pbc', i)

            sec_system.atoms.positions = apply_unit(self.traj_parsers.eval('get_positions', i), 'distance')
            atom_labels = self.traj_parsers.eval('get_atom_labels', i)
            sec_system.atoms.labels = atom_labels

            velocities = self.traj_parsers.eval('get_velocities', i)
            if velocities is not None:
                sec_system.atoms.velocities = apply_unit(velocities, 'velocity')

            forces = self.traj_parsers.eval('get_forces', i)
            if forces is not None:
                sec_scc = sec_run.m_create(Calculation)
                sec_scc.forces = Forces(total=ForcesEntry(value=apply_unit(forces, 'force')))

        # parse atomsgroup (moltypes --> molecules --> residues)
        atom_info = self.traj_parsers.eval('atom_info')
        if atom_info is None:
            atom_info = self._mdanalysistraj_parser.get('atom_info', None)
        if atom_info is not None:
            atoms_moltypes = np.array(atom_info.get('moltypes', []))
            atoms_molnums = np.array(atom_info.get('molnums', []))
            atoms_resids = np.array(atom_info.get('resids', []))
            atoms_elements = np.array(atom_info.get('elements', []))
            atoms_types = np.array(atom_info.get('types', []))
            if 'X' in atoms_elements:
                atoms_elements = np.array(sec_system.atoms.labels)
            if 'X' in atoms_elements:
                atoms_elements = atoms_types
            atoms_resnames = np.array(atom_info.get('resnames', []))
            moltypes = np.unique(atoms_moltypes)
            for i_moltype, moltype in enumerate(moltypes):
                # Only add atomsgroup for initial system for now
                sec_molecule_group = sec_run.system[0].m_create(AtomsGroup)
                sec_molecule_group.label = f'seg_{i_moltype}_{moltype}'
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
                for i_molecule, molecule in enumerate(np.unique(molecules[sec_molecule_group.atom_indices])):
                    sec_molecule = sec_molecule_group.m_create(AtomsGroup)
                    sec_molecule.index = i_molecule
                    sec_molecule.atom_indices = np.where(molecules == molecule)[0]
                    sec_molecule.n_atoms = len(sec_molecule.atom_indices)
                    # use first particle to get the moltype
                    # not sure why but this value is being cast to int, cast back to str
                    sec_molecule.label = str(atoms_moltypes[sec_molecule.atom_indices[0]])
                    sec_molecule.type = 'molecule'
                    sec_molecule.is_molecule = True

                    mol_resids = np.unique(atoms_resids[sec_molecule.atom_indices])
                    n_res = mol_resids.shape[0]
                    if n_res == 1:
                        elements = atoms_elements[sec_molecule.atom_indices]
                        sec_molecule.composition_formula = get_composition(elements)
                    else:
                        for i_res, res_id in enumerate(mol_resids):
                            sec_residue = sec_molecule.m_create(AtomsGroup)
                            sec_residue.index = i_res
                            atom_indices = np.where(atoms_resids == res_id)[0]
                            sec_residue.atom_indices = np.intersect1d(atom_indices, sec_molecule.atom_indices)
                            sec_residue.n_atoms = len(sec_residue.atom_indices)
                            res_names = atoms_resnames[sec_residue.atom_indices]
                            sec_residue.label = str(res_names[0])
                            sec_residue.type = 'monomer'
                            sec_residue.is_molecule = False
                            elements = atoms_elements[sec_residue.atom_indices]
                            sec_residue.composition_formula = get_composition(elements)

                        names = atoms_resnames[sec_molecule.atom_indices]
                        ids = atoms_resids[sec_molecule.atom_indices]
                        # filter for the first instance of each residue, as to not overcount
                        __, ids_count = np.unique(ids, return_counts=True)
                        # get the index of the first atom of each residue
                        ids_firstatom = np.cumsum(ids_count)[:-1]
                        # add the 0th index manually
                        ids_firstatom = np.insert(ids_firstatom, 0, 0)
                        names_firstatom = names[ids_firstatom]
                        sec_molecule.composition_formula = get_composition(names_firstatom)

            # calculate molecular radial distribution functions
            sec_molecular_dynamics = self.archive.workflow[-1].molecular_dynamics
            rdf_results = self.traj_parsers.eval('calc_molecular_rdf')
            rdf_results = rdf_results() if rdf_results is not None else None
            if rdf_results is None:
                rdf_results = self._mdanalysistraj_parser.calc_molecular_rdf()
            if rdf_results is not None:
                sec_rdfs = sec_molecular_dynamics.m_create(RadialDistributionFunction)
                sec_rdfs.type = 'Molecular'
                sec_rdfs.n_smooth = rdf_results.get('n_smooth')
                sec_rdfs.variables_name = np.array(['distance'])
                for i_pair, pair_type in enumerate(rdf_results.get('types')):
                    sec_rdf_values = sec_rdfs.m_create(RadialDistributionFunctionValues)
                    sec_rdf_values.label = str(pair_type)
                    sec_rdf_values.n_bins = len(rdf_results['bins'][i_pair]) if rdf_results.get(
                        'bins') is not None else []
                    sec_rdf_values.bins = rdf_results['bins'][i_pair] if rdf_results.get(
                        'bins') is not None else []
                    sec_rdf_values.value = rdf_results['value'][i_pair] if rdf_results.get(
                        'value') is not None else []

            # calculate the molecular mean squared displacements
            msd_results = self.traj_parsers.eval('calc_molecular_mean_squard_displacements')
            msd_results = msd_results() if msd_results is not None else None
            if msd_results is None:
                msd_results = self._mdanalysistraj_parser.calc_molecular_mean_squard_displacements()
            if msd_results is not None:
                sec_msds = sec_molecular_dynamics.m_create(MeanSquaredDisplacement)
                sec_msds.type = 'Molecular'
                for i_type, moltype in enumerate(msd_results.get('types')):
                    sec_msd_values = sec_msds.m_create(MeanSquaredDisplacementValues)
                    sec_msd_values.label = str(moltype)
                    sec_msd_values.n_times = len(msd_results['times'][i_type]) if msd_results.get(
                        'times') is not None else []
                    sec_msd_values.times = msd_results['times'][i_type] if msd_results.get(
                        'times') is not None else []
                    sec_msd_values.value = msd_results['value'][i_type] if msd_results.get(
                        'value') is not None else []
                    sec_diffusion = sec_msd_values.m_create(DiffusionConstantValues)
                    sec_diffusion.value = msd_results['diffusion_constant'][i_type] if msd_results.get(
                        'diffusion_constant') is not None else []
                    sec_diffusion.error_type = 'Pearson correlation coefficient'
                    sec_diffusion.errors = msd_results['error_diffusion_constant'][i_type] if msd_results.get(
                        'error_diffusion_constant') is not None else []

    def parse_method(self):
        sec_run = self.archive.run[-1]

        if self.traj_parsers[0].mainfile is None or self.data_parser.mainfile is None:
            return

        masses = self.data_parser.get('Masses', None)

        self.traj_parsers[0].masses = masses

        sec_method = sec_run.m_create(Method)
        sec_force_field = sec_method.m_create(ForceField)
        sec_model = sec_force_field.m_create(Model)

        interactions = self.log_parser.get_interactions()
        if not interactions:
            interactions = self.data_parser.get_interactions()

        for interaction in interactions:
            if not interaction[0] or interaction[1] is None or np.size(interaction[1]) == 0:
                continue
            sec_interaction = sec_model.m_create(Interaction)
            sec_interaction.type = str(interaction[0])
            sec_interaction.parameters = [[float(ai) for ai in a] for a in interaction[1]]

    def parse_input(self):
        sec_run = self.archive.run[-1]
        sec_input_output_files = sec_run.m_create(x_lammps_section_input_output_files)

        if self.data_parser.mainfile is not None:
            sec_input_output_files.x_lammps_inout_file_data = os.path.basename(
                self.data_parser.mainfile)

        if self.traj_parsers[0].mainfile is not None:
            sec_input_output_files.x_lammps_inout_file_trajectory = os.path.basename(
                self.traj_parsers[0].mainfile)

        sec_control_parameters = sec_run.m_create(x_lammps_section_control_parameters)
        keys = self.log_parser._commands
        for key in keys:
            val = self.log_parser.get(key, None)
            if val is None:
                continue
            val = val[0] if len(val) == 1 else val
            key = 'x_lammps_inout_control_%s' % key.replace('_', '').replace('/', '').lower()
            if hasattr(sec_control_parameters, key):
                if isinstance(val, list):
                    val = ' '.join([str(v) for v in val])
                setattr(sec_control_parameters, key, str(val))

    def init_parser(self):
        self.log_parser.mainfile = self.filepath
        self.log_parser.logger = self.logger
        self._traj_parser.logger = self.logger
        self._mdanalysistraj_parser.logger = self.logger
        self._xyztraj_parser.logger = self.logger
        self.data_parser.logger = self.logger
        # auxilliary log parser for thermo data
        self.aux_log_parser.logger = self.logger
        self.log_parser._units = None
        self._traj_parser._chemical_symbols = None
        self._frame_rate = None

    def parse(self, filepath, archive, logger):
        self.filepath = filepath
        self.archive = archive
        self.logger = logger if logger is not None else logging

        self.init_parser()

        sec_run = self.archive.m_create(Run)

        # parse basic
        sec_run.program = Program(
            name='LAMMPS', version=self.log_parser.get('program_version', ''))

        # parse data file associated with calculation
        data_files = self.log_parser.get_data_files()
        if len(data_files) > 1:
            self.logger.warn('Multiple data files are specified')
        if data_files:
            self.data_parser.mainfile = data_files[0]

        # parse trajectorty file associated with calculation
        traj_files = self.log_parser.get_traj_files()
        if len(traj_files) > 1:
            self.logger.warn('Multiple traj files are specified')

        parsers = []
        for n, traj_file in enumerate(traj_files):
            # parser initialization for each traj file cannot be avoided as there are
            # cases where traj files can share the same parser
            file_type = self.log_parser.get('dump', [[1, 'all', traj_file[-3:]]] * (n + 1))[n][2]
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
                custom_options = [option.replace('xu', 'x') for option in custom_options]
                custom_options = [option.replace('yu', 'y') for option in custom_options]
                custom_options = [option.replace('zu', 'z') for option in custom_options]
                custom_options = ' '.join(custom_options)
                traj_parser = MDAnalysisParser(topology_format='DATA', format='LAMMPSDUMP', atom_style=custom_options)
                if data_files:
                    traj_parser.mainfile = data_files[0]
                traj_parser.auxilliary_files = [traj_file]
                # try to check if MDAnalysis can construct the universe or at least parse
                # the atoms, otherwise will fall back to TrajParser
                if traj_parser.universe is None or 'X' in traj_parser.get('atom_info', {}).get('names', []):
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

        # parse data from auxiliary log file
        if self.log_parser.get('log') is not None:
            self.aux_log_parser.mainfile = os.path.join(
                self.log_parser.maindir, self.log_parser.get('log')[0])
            # we assign units here which is read from log parser
            self.aux_log_parser._units = self.log_parser.units

        self.parse_method()

        self.parse_workflow()

        self.parse_system()

        # parse thermodynamic data from log file
        self.parse_thermodynamic_data()

        # include input controls from log file
        self.parse_input()
