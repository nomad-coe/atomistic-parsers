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
from ase.io import read as aseread
from ase import Atoms as aseAtoms
from datetime import datetime

from nomad.units import ureg
from nomad.parsing.file_parser import Quantity, TextParser
from runschema.run import Run, Program, TimeRun
from runschema.method import Method, TB, xTB, Interaction
from runschema.system import System, Atoms
from runschema.calculation import (
    Calculation,
    ScfIteration,
    Energy,
    EnergyEntry,
    BandEnergies,
    Multipoles,
    MultipolesEntry,
)
from simulationworkflowschema import (
    SinglePoint,
    GeometryOptimization,
    GeometryOptimizationMethod,
)
from atomisticparsers.utils import MDParser
from atomisticparsers.xtb.metainfo import xtb  # pylint: disable=unused-import


re_f = r'[-+]?\d+\.\d*(?:[Ee][-+]\d+)?'
re_n = r'[\n\r]'


class OutParser(TextParser):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def init_quantities(self):
        re_f = r'[\d\.E\+\-]+'

        def str_to_eigenvalues(val_in):
            occupations, energies = [], []
            for val in val_in.strip().split('\n'):
                val = val.split('(')[0].split()
                if not val[0].isdecimal():
                    continue
                occupations.append(float(val.pop(1)) if len(val) > 3 else 0.0)
                energies.append(float(val[1]))
            return occupations, energies * ureg.hartree

        def str_to_parameters(val_in):
            val = [v.strip() for v in val_in.split('  ', 1)]
            val[1] = val[1].split()
            return val

        def str_to_wall_time(val_in):
            name, d, h, m, s = val_in.rsplit(' ', 4)
            return name.strip(), 24 * 60 * 60 * float(d) + 60 * 60 * float(
                h
            ) + 60 * float(m) + float(s)

        common_quantities = [
            Quantity(
                'setup',
                r'SETUP\s*:\s*([\s\S]+?\.+\n *\n)',
                sub_parser=TextParser(
                    quantities=[
                        Quantity(
                            'parameter',
                            r'\n +\: +(.+?\s{2,}[\w\.\-\+]+)',
                            str_operation=lambda x: [
                                v.strip() for v in x.split('  ', 1)
                            ],
                            repeats=True,
                        )
                    ]
                ),
            ),
            Quantity(
                'summary',
                r'(SUMMARY[\s\S]+?\:\n *\n)',
                sub_parser=TextParser(
                    quantities=[
                        Quantity(
                            'energy_total',
                            rf':: total energy\s*({re_f})',
                            unit=ureg.hartree,
                            dtype=np.float64,
                        ),
                        Quantity(
                            'x_xtb_gradient_norm',
                            rf':: gradient norm\s*({re_f})',
                            unit=ureg.hartree / ureg.angstrom,
                            dtype=np.float64,
                        ),
                        Quantity(
                            'x_xtb_hl_gap',
                            rf':: HOMO-LUMO gap\s*({re_f})',
                            unit=ureg.eV,
                            dtype=np.float64,
                        ),
                        Quantity(
                            'energy_x_xtb_scc',
                            rf':: SCC energy\s*({re_f})',
                            unit=ureg.hartree,
                            dtype=np.float64,
                        ),
                        Quantity(
                            'energy_x_xtb_isotropic_es',
                            rf':: \-\> isotropic ES\s*({re_f})',
                            unit=ureg.hartree,
                            dtype=np.float64,
                        ),
                        Quantity(
                            'energy_x_xtb_anisotropic_es',
                            rf':: \-\> anisotropic ES\s*({re_f})',
                            unit=ureg.hartree,
                            dtype=np.float64,
                        ),
                        Quantity(
                            'energy_x_xtb_anisotropic_xc',
                            rf':: \-\> anisotropic XC\s*({re_f})',
                            unit=ureg.hartree,
                            dtype=np.float64,
                        ),
                        Quantity(
                            'energy_x_xtb_dispersion',
                            rf':: \-\> dispersion\s*({re_f})',
                            unit=ureg.hartree,
                            dtype=np.float64,
                        ),
                        Quantity(
                            'energy_electrostatic',
                            rf':: \-\> electrostatic\s*({re_f})',
                            unit=ureg.hartree,
                            dtype=np.float64,
                        ),
                        Quantity(
                            'energy_x_xtb_repulsion',
                            rf':: repulsion energy\s*({re_f})',
                            unit=ureg.hartree,
                            dtype=np.float64,
                        ),
                        Quantity(
                            'energy_x_xtb_halogen_bond_corr',
                            rf':: halogen bond corr\.\s*({re_f})',
                            unit=ureg.hartree,
                            dtype=np.float64,
                        ),
                        Quantity(
                            'energy_x_xtb_add_restraining',
                            rf':: repulsion energy\s*({re_f})',
                            unit=ureg.hartree,
                            dtype=np.float64,
                        ),
                        Quantity(
                            'charge_total',
                            rf':: total charge\s*({re_f})',
                            unit=ureg.elementary_charge,
                            dtype=np.float64,
                        ),
                    ]
                ),
            ),
        ]

        orbital_quantities = [
            Quantity(
                'eigenvalues',
                r'# +Occupation +Energy.+\s*\-+([\s\S]+?)\-+\n',
                str_operation=str_to_eigenvalues,
            ),
            Quantity(
                'hl_gap', rf'HL\-Gap\s*({re_f})', dtype=np.float64, unit=ureg.hartree
            ),
            Quantity(
                'energy_fermi',
                rf'Fermi\-level\s*({re_f})',
                dtype=np.float64,
                unit=ureg.hartree,
            ),
        ]

        property_quantities = orbital_quantities + [
            Quantity(
                'dipole',
                r'(dipole\:[\s\S]+?)molecular',
                sub_parser=TextParser(
                    quantities=[
                        Quantity(
                            'q',
                            rf'q only: +({re_f} +{re_f} +{re_f})',
                            dtype=np.dtype(np.float64),
                            unit=ureg.elementary_charge * ureg.bohr,
                        ),
                        Quantity(
                            'full',
                            rf'full: +({re_f} +{re_f} +{re_f})',
                            dtype=np.dtype(np.float64),
                            unit=ureg.elementary_charge * ureg.bohr,
                        ),
                    ]
                ),
            ),
            Quantity(
                'quadrupole',
                r'(quadrupole \(traceless\):[\s\S]+?)\n *\n',
                sub_parser=TextParser(
                    quantities=[
                        Quantity(
                            'q',
                            r'q only:(.+)',
                            dtype=np.dtype(np.float64),
                            unit=ureg.elementary_charge * ureg.bohr**2,
                        ),
                        Quantity(
                            'full',
                            r'full:(.+)',
                            dtype=np.dtype(np.float64),
                            unit=ureg.elementary_charge * ureg.bohr**2,
                        ),
                        Quantity(
                            'q_dip',
                            r'q\+dip:(.+)',
                            dtype=np.dtype(np.float64),
                            unit=ureg.elementary_charge * ureg.bohr**2,
                        ),
                    ]
                ),
            ),
        ]

        geometry_quantities = [
            Quantity('file', r'optimized geometry written to:\s*(\S+)')
        ]

        scf_quantities = (
            common_quantities
            + orbital_quantities
            + [
                Quantity(
                    'model',
                    r'((?:G F N \d+ \- x T B.+\s+\-+\s+|Reference)\s*[\s\S]+?\n *\n)',
                    sub_parser=TextParser(
                        quantities=[
                            Quantity('reference', r'Reference\s*(\S+)'),
                            Quantity(
                                'contribution',
                                r'(\w+:\s*[\s\S]+?)(?:\*|\n *\n)',
                                repeats=True,
                                sub_parser=TextParser(
                                    quantities=[
                                        Quantity('name', r'(\w+):'),
                                        Quantity(
                                            'parameters',
                                            r'\n +(\w.+?  .+)',
                                            str_operation=str_to_parameters,
                                            repeats=True,
                                        ),
                                    ]
                                ),
                            ),
                        ]
                    ),
                ),
                Quantity(
                    'scf_iteration',
                    r'iter\s*E\s*dE.+([\s\S]+?convergence.+)',
                    sub_parser=TextParser(
                        quantities=[
                            Quantity('step', r'(\d+ .+)', repeats=True),
                            Quantity(
                                'converged',
                                r'(\*\*\* convergence criteria.+)',
                                str_operation=lambda x: 'satisfied' in x,
                            ),
                        ]
                    ),
                ),
            ]
        )

        optimization_quantities = [
            Quantity(
                'cycle',
                r'CYCLE +\d([\s\S]+?\n *\n)',
                repeats=True,
                sub_parser=TextParser(
                    quantities=[
                        Quantity(
                            'energy_total',
                            rf'total energy +: +({re_f}) Eh',
                            dtype=np.float64,
                            unit=ureg.hartree,
                        ),
                        Quantity(
                            'energy_change',
                            rf'change +({re_f}) Eh',
                            dtype=np.float64,
                            unit=ureg.hartree,
                        ),
                        Quantity(
                            'scf_iteration',
                            rf'\.+(\s+\d+\s+{re_f}[\s\S]+?)\*',
                            sub_parser=TextParser(
                                quantities=[
                                    Quantity(
                                        'step',
                                        rf'{re_n} +(\d+ +{re_f}.+)',
                                        repeats=True,
                                    ),
                                    Quantity(
                                        'time',
                                        rf'SCC iter\. +\.+ +(\d+) min, +({re_f}) sec',
                                    ),
                                ]
                            ),
                        ),
                    ]
                ),
            ),
            Quantity(
                'converged',
                r'(\*\*\* GEOMETRY OPTIMIZATION.+)',
                str_operation=lambda x: 'CONVERGED' in x,
            ),
            Quantity(
                'final_structure',
                r'final structure:([\s\S]+?\-+\s+\|)',
                sub_parser=TextParser(
                    quantities=[
                        Quantity('atom_labels', r'([A-Z][a-z]?) ', repeats=True),
                        Quantity(
                            'atom_positions',
                            rf'({re_f} +{re_f} +{re_f})',
                            unit=ureg.angstrom,
                            dtype=np.dtype(np.float64),
                        ),
                    ]
                ),
            ),
            Quantity(
                'final_single_point',
                r'(Final Singlepoint +\|[\s\S]+?::::::::::::)',
                sub_parser=TextParser(quantities=scf_quantities),
            ),
        ] + common_quantities

        md_quantities = [
            Quantity('traj_file', r'trajectories on (.+?\.trj)'),
            Quantity(
                'x_xtb_md_time',
                rf'MD time /ps +: +({re_f})',
                dtype=np.float64,
                unit=ureg.ps,
            ),
            Quantity(
                'timestep', rf'dt /fs +: +({re_f})', dtype=np.float64, unit=ureg.fs
            ),
            Quantity(
                'x_xtb_scc_accuracy', rf'SCC accuracy +: +({re_f})', dtype=np.float64
            ),
            Quantity(
                'x_xtb_temperature',
                rf'temperature /K +: +({re_f})',
                dtype=np.float64,
                unit=ureg.K,
            ),
            Quantity('x_xtb_max_steps', rf'max_steps +: +(\d+)', dtype=np.int32),
            Quantity(
                'x_xtb_block_length',
                rf'block length \(av\. \) +: +(\d+)',
                dtype=np.int32,
            ),
            Quantity(
                'x_xtb_dumpstep_trj',
                rf'dumpstep\(trj\) /fs +: +({re_f})',
                dtype=np.float64,
            ),
            Quantity(
                'x_xtb_dumpstep_coords',
                rf'dumpstep\(coords\) /fs +: +({re_f})',
                dtype=np.float64,
            ),
            Quantity(
                'x_xtb_h_atoms_mass',
                rf'H atoms mass \(amu\)  +: +(\d+)',
                dtype=np.float64,
                unit=ureg.amu,
            ),
            Quantity('x_xtb_n_degrees_freedom', rf' +: +(\d+)', dtype=np.float64),
            Quantity(
                'x_xtb_shake_bonds', rf'SHAKE on\. # bonds +: +(\d+)', dtype=np.float64
            ),
            Quantity(
                'x_xtb_berendsen',
                rf'Berendsen THERMOSTAT (\S+)',
                str_operation=lambda x: x == 'on',
            ),
            Quantity(
                'cycle',
                rf'{re_n} +(\d+ +{re_f} +{re_f} +{re_f} +{re_f} +{re_f} +{re_f})',
                dtype=np.dtype(np.float64),
                repeats=True,
            ),
        ]

        self._quantities = [
            Quantity('program_version', r'\* xtb version ([\d\.]+)'),
            Quantity(
                'date_start',
                r'started run on (\d+/\d+/\d+) at (\d+:\d+:\d+\.\d+)',
                dtype=str,
                flatten=False,
            ),
            Quantity(
                'date_end',
                r'finished run on (\d+/\d+/\d+) at (\d+:\d+:\d+\.\d+)',
                dtype=str,
                flatten=False,
            ),
            Quantity(
                'calculation_setup',
                r'Calculation Setup +\|\s*\-+\s*([\s\S]+?)\-+\s+\|',
                sub_parser=TextParser(
                    quantities=[
                        Quantity(
                            'parameter',
                            r'([\w ]+:.+)',
                            str_operation=lambda x: [v.strip() for v in x.split(':')],
                            repeats=True,
                        )
                    ]
                ),
            ),
            Quantity(
                'gfnff',
                r'(G F N - F F[\s\S]+?::::::::::::\n *\n)',
                sub_parser=TextParser(quantities=scf_quantities),
            ),
            Quantity(
                'gfn1',
                r'(G F N 1 - x T B[\s\S]+?::::::::::::\n *\n)',
                sub_parser=TextParser(quantities=scf_quantities),
            ),
            Quantity(
                'gfn2',
                r'(G F N 2 - x T B[\s\S]+?::::::::::::\n *\n)',
                sub_parser=TextParser(quantities=scf_quantities),
            ),
            Quantity(
                'ancopt',
                r'(A N C O P T +\|[\s\S]+?::::::::::::\n *\n)',
                sub_parser=TextParser(quantities=optimization_quantities),
            ),
            Quantity(
                'md',
                r'(Molecular Dynamics +\|[\s\S]+?exit of md)',
                sub_parser=TextParser(quantities=md_quantities),
            ),
            Quantity(
                'property',
                r'(Property Printout +\|[\s\S]+?\-+\s+\|)',
                sub_parser=TextParser(quantities=property_quantities),
            ),
            Quantity(
                'geometry',
                r'(Geometry Summary +\|[\s\S]+?\-+\s+\|)',
                sub_parser=TextParser(quantities=geometry_quantities),
            ),
            Quantity(
                'energy_total',
                rf'\| TOTAL ENERGY\s*({re_f})',
                dtype=np.float64,
                unit=ureg.hartree,
            ),
            Quantity(
                'gradient_norm',
                rf'\| GRADIENT NORM\s*({re_f})',
                dtype=np.float64,
                unit=ureg.hartree / ureg.angstrom,
            ),
            Quantity(
                'hl_gap',
                rf'\| HOMO-LUMO GAP\s*({re_f})',
                dtype=np.float64,
                unit=ureg.eV,
            ),
            Quantity('topo_file', r'Writing topology from bond orders to (.+\.mol)'),
            Quantity(
                'footer',
                r'(\* finished run on [\s\S]+?\Z)',
                sub_parser=TextParser(
                    quantities=[
                        Quantity(
                            'end_time', r'finished run on (\S+) at (\S+)', flatten=False
                        ),
                        Quantity(
                            'wall_time',
                            r'(.+):\s+\* +wall-time: +(\d+) d, +(\d+) h, +(\d+) min, +([\d\.]+) sec',
                            repeats=True,
                            str_operation=str_to_wall_time,
                        ),
                        Quantity(
                            'cpu_time',
                            r'\* +cpu-time: +(\d+) d, +(\d+) h, +(\d+) min, +([\d\.]+) sec',
                            repeats=True,
                        ),
                    ]
                ),
            ),
        ]

    def get_time(self, section=None, index=0):
        start_time = 0
        section_index = 0
        for time in self.get('footer', {}).get('wall_time', []):
            if time[0] == section or section is None:
                if index == section_index:
                    return start_time, time[1]
                section_index += 1
            if time[0] != 'total':
                start_time += time[1]
        return start_time, None


class CoordParser(TextParser):
    def __init__(self):
        super().__init__()

    def init_quantities(self):
        re_f = r'[\d\.\-]+'

        self._quantities = [
            Quantity('coord_unit', r'\$coord(.+)'),
            Quantity(
                'positions_labels',
                rf'({re_f} +{re_f} +{re_f} +[A-Za-z]+\s+)',
                repeats=True,
            ),
            Quantity('periodic', r'\$periodic(.+)'),
            Quantity('lattice_unit', r'\$lattice(.+)'),
            Quantity(
                'lattice',
                rf'({re_f} +{re_f} +{re_f}) *\n',
                repeats=True,
                dtype=np.dtype(np.float64),
            ),
            Quantity('cell_unit', r'\$cell(.+)'),
            Quantity(
                'cell',
                rf'({re_f} +{re_f} +{re_f} +{re_f} +{re_f} +{re_f}) *\n',
                dtype=np.dtype(np.float64),
            ),
        ]

    def get_atoms(self):
        positions = self.get('positions_labels')
        if positions is None:
            return

        lattice_unit = self.get('lattice_unit', '').strip()
        lattice_unit = ureg.angstrom if lattice_unit.startswith('angs') else ureg.bohr
        lattice = self.get('lattice')
        lattice = (
            (lattice * lattice_unit).to('angstrom').magnitude
            if lattice is not None
            else lattice
        )

        cell = self.get('cell')
        if cell is not None:
            cell_unit = self.get('cell_unit')
            cell_unit = ureg.angstrom if cell_unit is not None else ureg.bohr
            cell_abc = (cell[:3] * cell_unit).to('angstrom').magnitude
            lattice = list(cell_abc) + list(cell[3:])

        labels = [p[-1].title() for p in positions]
        positions = [p[:3] for p in positions]
        coord_unit = self.get('coord_unit', '').strip()
        if coord_unit.startswith('frac') and lattice is not None:
            positions = np.dot(positions, lattice)
        elif coord_unit.startswith('angs'):
            positions = positions * ureg.angstrom
        else:
            positions = positions * ureg.bohr
        positions = positions.to('angstrom').magnitude

        pbc = ([True] * int(self.get('periodic', 0))) + [False] * 3

        return aseAtoms(symbols=labels, positions=positions, cell=lattice, pbc=pbc[:3])


class TrajParser(TextParser):
    def __init__(self):
        super().__init__()

    def init_quantities(self):
        re_f = r'[\d\.\-]+'

        self._quantities = [
            Quantity(
                'frame',
                r'energy\:([\s\S]+?(?:\Z|\n *\d+ *\n))',
                repeats=True,
                sub_parser=TextParser(
                    quantities=[
                        Quantity(
                            'positions',
                            rf'({re_f} +{re_f} +{re_f})',
                            repeats=True,
                            dtype=np.dtype(np.float64),
                        ),
                        Quantity('labels', r'\n *([A-Za-z]{1,2}) +', repeats=True),
                    ]
                ),
            )
        ]

    def get_atoms(self, n_frame):
        frames = self.get('frame', [])
        if n_frame >= len(frames):
            return
        frame = self.get('frame')[n_frame]
        labels = [label.title() for label in frame.get('labels', [])]
        # TODO verify if trajectory positions are always printed out in angstroms
        return aseAtoms(symbols=labels, positions=frame.positions)


class XTBParser(MDParser):
    def __init__(self):
        self.out_parser = OutParser()
        self.coord_parser = CoordParser()
        self.traj_parser = TrajParser()
        self.calculation_type = None
        self._metainfo_map = {
            'optimization level': 'optimization_level',
            'max. optcycles': 'max_opt_cycles',
            'ANC micro-cycles': 'anc_micro_cycles',
            'degrees of freedom': 'n_degrees_freedom',
            'RF solver': 'rf_solver',
            'linear?': 'linear',
            'Hlow (freq-cutoff)': 'hlow',
            'Hmax (freq-cutoff)': 'hmax',
            'S6 in model hess.': 's6',
        }
        super().__init__()

    def parse_system(self, source):
        if isinstance(source, int):
            atoms = self.traj_parser.get_atoms(source)
        elif source.endswith('.xyz') or source.endswith('.poscar'):
            atoms = aseread(os.path.join(self.maindir, source))
        else:
            self.coord_parser.mainfile = os.path.join(self.maindir, source)
            atoms = self.coord_parser.get_atoms()

        if atoms is None:
            return

        sec_system = System()
        self.archive.run[0].system.append(sec_system)
        sec_atoms = Atoms()
        sec_system.atoms = sec_atoms
        sec_atoms.labels = atoms.get_chemical_symbols()
        sec_atoms.positions = atoms.get_positions() * ureg.angstrom
        lattice_vectors = np.array(atoms.get_cell())
        if np.count_nonzero(lattice_vectors) > 0:
            sec_atoms.lattice_vectors = lattice_vectors * ureg.angstrom
            sec_atoms.periodic = atoms.get_pbc()

        return sec_system

    def parse_calculation(self, source):
        sec_calc = Calculation()
        self.archive.run[0].calculation.append(sec_calc)
        # total energy
        sec_energy = Energy()
        sec_calc.energy = sec_energy
        sec_energy.total = EnergyEntry(value=source.energy_total)
        sec_energy.change = source.energy_change

        # scf
        for step in source.get('scf_iteration', {}).get('step', []):
            sec_scf = ScfIteration()
            sec_calc.scf_iteration.append(sec_scf)
            sec_scf.energy = Energy(
                total=EnergyEntry(value=step[1] * ureg.hartree),
                change=step[2] * ureg.hartree,
            )

        # summary of calculated properties
        summary = source.get('summary', {})
        for key, val in summary.items():
            if key.startswith('energy_') and val is not None:
                setattr(sec_energy, key.replace('energy_', ''), EnergyEntry(value=val))

        # eigenvalues
        if source.eigenvalues is not None:
            sec_eigs = BandEnergies()
            sec_calc.eigenvalues.append(sec_eigs)
            sec_eigs.occupations = np.reshape(
                source.eigenvalues[0], (1, 1, len(source.eigenvalues[0]))
            )
            sec_eigs.energies = np.reshape(
                source.eigenvalues[1], (1, 1, len(source.eigenvalues[1]))
            )
            sec_eigs.kpoints = np.zeros((1, 3))

        return sec_calc

    def parse_method(self, section):
        model = self.out_parser.get(section, {}).get('model')
        if model is None:
            return

        sec_method = Method()
        self.archive.run[-1].method.append(sec_method)
        parameters = {
            p[0]: p[1]
            for p in self.out_parser.get(section, {})
            .get('setup', {})
            .get('parameter', [])
        }
        sec_tb = TB()
        sec_method.tb = sec_tb
        sec_tb.name = 'xTB'
        sec_tb.x_xtb_setup = parameters
        sec_xtb = xTB()
        sec_tb.xtb = sec_xtb
        sec_xtb.name = section

        if model.get('reference') is not None:
            sec_xtb.reference = model.reference

        for contribution in model.get('contribution', []):
            name = contribution.name.lower()
            if name == 'hamiltonian':
                sec_interaction = Interaction()
                sec_xtb.hamiltonian.append(sec_interaction)
            elif name == 'coulomb':
                sec_interaction = Interaction()
                sec_xtb.coulomb.append(sec_interaction)
            elif name == 'repulsion':
                sec_interaction = Interaction()
                sec_xtb.repulsion.append(sec_interaction)
            else:
                sec_interaction = Interaction()
                sec_xtb.contributions.append(sec_interaction)
                sec_interaction.type = name
            sec_interaction.parameters = {
                p[0]: p[1].tolist() if isinstance(p[1], np.ndarray) else p[1]
                for p in contribution.parameters
            }

    def parse_single_point(self, source, section):
        if source is None:
            return

        total_time = None
        # determine file extension of input structure file
        coord_file = self.archive.run[-1].x_xtb_calculation_setup.get(
            'coordinate file', 'coord'
        )
        if section == 'final_single_point':
            extension = 'coord' if coord_file == 'coord' else coord_file.split('.')[-1]
            coord_file = f'xtbopt.{extension}'
        else:
            self._run_index += 1
            start_time, total_time = self.out_parser.get_time(index=self._run_index)

        sec_system = self.parse_system(coord_file)
        sec_calc = self.parse_calculation(source)
        sec_calc.system_ref = sec_system
        if total_time is not None:
            sec_calc.time_physical = start_time + total_time
            sec_calc.time_calculation = total_time

        return sec_calc

    def parse_gfn(self, section):
        self.parse_method(section)
        self.parse_single_point(self.out_parser.get(section), section)
        self.archive.workflow2 = SinglePoint()

    def parse_opt(self, section):
        module = self.out_parser.get(section)
        if module is None:
            return

        self._run_index += 1

        start_time, total_time = self.out_parser.get_time(section='ANC optimizer')
        time_per_step = (
            total_time / (len(module.get('cycle')) + 1)
            if total_time is not None
            else None
        )
        self.traj_parser.mainfile = os.path.join(self.maindir, 'xtbopt.log')

        for n, cycle in enumerate(module.get('cycle', [])):
            self.parse_system(n)
            sec_scc = self.parse_calculation(cycle)
            if sec_scc is not None and time_per_step is not None:
                sec_scc.time_physical = start_time + time_per_step * (n + 1)
                sec_scc.time_calculation = time_per_step

        # final single point
        sec_scc = self.parse_single_point(
            module.get('final_single_point'), 'final_single_point'
        )
        if sec_scc is not None and time_per_step is not None:
            sec_scc.time_physical = start_time + time_per_step * (
                len(module.get('cycle', [])) + 1
            )
            sec_scc.time_calculation = time_per_step

        # workflow parameters
        workflow = GeometryOptimization(method=GeometryOptimizationMethod())
        for key, val in module.get('setup', {}).get('parameter', []):
            name = self._metainfo_map.get(key)
            if key == 'energy convergence':
                workflow.method.convergence_tolerance_energy_difference = (
                    val * ureg.hartree
                )
            elif key == 'grad. convergence':
                workflow.method.convergence_tolerance_force_maximum = (
                    val * ureg.hartree / ureg.bohr
                )
            elif key == 'maximium RF displ.':
                workflow.method.convergence_tolerance_displacement_maximum = (
                    val * ureg.bohr
                )
            elif name is not None:
                setattr(workflow, f'x_xtb_{name}', val)
        self.archive.workflow2 = workflow

    def parse_md(self, section):
        module = self.out_parser.get(section)
        if module is None:
            return

        self.traj_parser.mainfile = os.path.join(self.maindir, 'xtb.trj')

        # get trj dump frequency to determine which frame to parse in trajectory file
        trj_freq = module.get('x_xtb_dumpstep_trj', 1)

        traj_steps = [
            n * int(trj_freq) for n in range(len(self.traj_parser.get('frame', [])))
        ]
        self.n_atoms = self.archive.run[-1].x_xtb_calculation_setup.get(
            'number of atoms', 0
        )
        self.trajectory_steps = [-1] + traj_steps
        self.thermodynamics_steps = [int(cycle[0]) for cycle in module.get('cycle', [])]

        for step in self.trajectory_steps:
            if step < 0:
                continue
            atoms = self.traj_parser.get_atoms(traj_steps.index(step))
            data = dict(
                labels=atoms.get_chemical_symbols(),
                positions=atoms.get_positions() * ureg.angstrom,
            )
            lattice_vectors = np.array(atoms.get_cell())
            if np.count_nonzero(lattice_vectors) > 0:
                data['lattice_vectors'] = lattice_vectors * ureg.angstrom
            self.parse_trajectory_step(dict(atoms=data))

        time_start, time_calc = self.out_parser.get_time(section='MD')
        time_step = (
            time_calc / (max(self.thermodynamics_steps) + 1)
            if time_calc is not None
            else None
        )

        for n_frame, step in enumerate(self.thermodynamics_steps):
            cycle = module.get('cycle')[n_frame]
            data = dict(
                step=step,
                time_physical=cycle[1] * ureg.ps,
                energy=dict(
                    total=dict(
                        potential=cycle[2] * ureg.hartree,
                        kinetic=cycle[3] * ureg.hartree,
                        value=cycle[6] * ureg.hartree,
                    )
                ),
                temperature=cycle[5] * ureg.kelvin,
            )
            if time_step is not None:
                data['time_physical'] = time_start + time_step * (step + 1)
                data['time_calculation'] = time_step
            self.parse_thermodynamics_step(data)

        # workflow parameters
        self.parse_md_workflow(
            {key: val for key, val in module.items() if key.startswith('x_xtb')}
        )

    def write_to_archive(self) -> None:
        self.out_parser.mainfile = self.mainfile
        self.out_parser.logger = self.logger
        self.coord_parser.logger = self.logger
        self.traj_parser.logger = self.logger
        self.calculation_type = None
        self.maindir = os.path.dirname(self.mainfile)
        self._run_index = 0

        # run parameters
        sec_run = Run()
        self.archive.run.append(sec_run)
        sec_run.program = Program(
            name='xTB', version=self.out_parser.get('program_version')
        )
        sec_run.x_xtb_calculation_setup = {
            p[0]: p[1]
            for p in self.out_parser.get('calculation_setup', {}).get('parameter', [])
        }
        if self.out_parser.date_start is not None:
            sec_run.time_run = TimeRun(
                date_start=datetime.strptime(
                    self.out_parser.date_start, '%Y/%m/%d %H:%M:%S.%f'
                ).timestamp()
            )
            if self.out_parser.date_end is not None:
                sec_run.time_run.date_end = datetime.strptime(
                    self.out_parser.date_end, '%Y/%m/%d %H:%M:%S.%f'
                ).timestamp()

        # modules
        self.parse_gfn('gfnff')
        self.parse_gfn('gfn1')
        self.parse_gfn('gfn2')
        self.parse_opt('ancopt')
        self.parse_md('md')

        # output properties
        properties = self.out_parser.get('property')
        if properties.dipole is not None:
            if sec_run.calculation:
                sec_calc = sec_run.calculation[-1]
            else:
                sec_calc = Calculation()
                sec_run.calculation.append(sec_calc)
            sec_multipoles = Multipoles()
            sec_calc.multipoles.append(sec_multipoles)
            sec_multipoles.dipole = MultipolesEntry(
                total=properties.dipole.full.to('C * m').magnitude,
                x_xtb_q_only=properties.dipole.q.to('C * m').magnitude,
            )
            if properties.quadrupole is not None:
                sec_multipoles.quadrupole = MultipolesEntry(
                    total=properties.quadrupole.full.to('C * m**2').magnitude,
                    x_xtb_q_only=properties.quadrupole.q.to('C * m**2').magnitude,
                    x_xtb_q_plus_dip=properties.quadrupole.q_dip.to(
                        'C * m**2'
                    ).magnitude,
                )
        # TODO implement vibrational properties
