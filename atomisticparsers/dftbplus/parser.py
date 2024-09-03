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
import io
import logging
import numpy as np

from nomad.units import ureg
from nomad.parsing.file_parser import TextParser, Quantity, FileParser, Parser
from runschema.run import Run, Program
from runschema.method import Method, TB
from runschema.system import System, Atoms
from runschema.calculation import (
    Calculation,
    Energy,
    EnergyEntry,
    ScfIteration,
    BandEnergies,
    Multipoles,
    MultipolesEntry,
    Forces,
    ForcesEntry,
)


re_n = r'[\n\r]'
re_f = r'[-+]?\d+\.\d*(?:[DdEe][-+]\d+)?'


class DetailedParser(TextParser):
    def init_quantities(self):
        def to_kpoint(val_in):
            val = np.transpose(
                np.array(
                    [line.split()[1:] for line in val_in.strip().splitlines()],
                    dtype=np.float64,
                )
            )
            eigs = np.array([val[i] for i in range(len(val)) if i % 2 == 0])
            occs = np.array([val[i] for i in range(len(val)) if i % 2 == 0])
            return eigs, occs

        self._quantities = [
            Quantity(
                'coordinates',
                r'Coordinates of moved atoms \(au\):\s+([\d\.\-\+\s]+)',
                dtype=np.dtype(np.float64),
                repeats=True,
            ),
            Quantity(
                'charges',
                r' Net atomic charges \(e\)\s+Atom +Net charge\s+([\d\.\-\+Ee\s]+)',
                dtype=np.dtype(np.float64),
                repeats=True,
            ),
            Quantity(
                'eigenvalues',
                r'Eigenvalues \/H\s+([\d\.\-\+\s]+)',
                dtype=np.dtype(np.float64),
                repeats=True,
            ),
            Quantity(
                'occupations',
                r'Fillings\s+([\d\.\-\+\s]+)',
                dtype=np.dtype(np.float64),
                repeats=True,
            ),
            Quantity(
                'eigenvalues_occupations',
                rf'Eigenvalues \(H\) and fillings \(e\)\s+([\s\S]+?)Eigenvalues',
                sub_parser=TextParser(
                    quantities=[
                        Quantity(
                            'kpoint',
                            r'K-points* \d+\:*\d*\s+([\d\.\-\+\s]+)',
                            str_operation=to_kpoint,
                            repeats=True,
                        )
                    ]
                ),
            ),
            Quantity(
                'fermi_level',
                rf'Fermi level: +({re_f}) H',
                dtype=np.float64,
                unit=ureg.hartree,
            ),
            Quantity(
                'energy_x_dftbp_band',
                rf'Band energy: +({re_f}) H',
                dtype=np.float64,
                unit=ureg.hartree,
            ),
            Quantity(
                'energy_x_dftbp_ts',
                rf'TS: +({re_f}) H',
                dtype=np.float64,
                unit=ureg.hartree,
            ),
            Quantity(
                'energy_x_dftbp_band_free',
                rf'Band free energy \(E\-TS\): +({re_f}) H',
                dtype=np.float64,
                unit=ureg.hartree,
            ),
            Quantity(
                'energy_x_dftbp_band_t0',
                rf'Extrapolated E\(0K\): +({re_f}) H',
                dtype=np.float64,
                unit=ureg.hartree,
            ),
            Quantity(
                'energy_sum_eigenvalues',
                rf'Energy H0: +({re_f}) H',
                dtype=np.float64,
                unit=ureg.hartree,
            ),
            Quantity(
                'energy_x_dftbp_scc',
                rf'Energy SCC: +({re_f}) H',
                dtype=np.float64,
                unit=ureg.hartree,
            ),
            Quantity(
                'energy_electronic',
                rf'Total Electronic energy: +({re_f}) H',
                dtype=np.float64,
                unit=ureg.hartree,
            ),
            Quantity(
                'energy_nuclear_repulsion',
                rf'Repulsive energy: +({re_f}) H',
                dtype=np.float64,
                unit=ureg.hartree,
            ),
            Quantity(
                'energy_x_dftbp_dispersion',
                rf'Dispersion energy: +({re_f}) H',
                dtype=np.float64,
                unit=ureg.hartree,
            ),
            Quantity(
                'energy_total',
                rf'Total energy: +({re_f}) H',
                dtype=np.float64,
                unit=ureg.hartree,
            ),
            Quantity(
                'energy_x_dftbp_total_mermin',
                rf'Total Mermin free energy: +({re_f}) H',
                dtype=np.float64,
                unit=ureg.hartree,
            ),
            Quantity(
                'pressure',
                rf'Pressure: +({re_f}) au',
                dtype=np.float64,
                unit=ureg.hartree / ureg.bohr**3,
            ),
            Quantity(
                'forces',
                r'Total Forces\s+([\d\.\-\+\sE]+)',
                dtype=np.dtype(np.float64),
                repeats=True,
            ),
            Quantity(
                'dipole',
                rf'Dipole moment +: +({re_f} +{re_f} +{re_f})',
                dtype=np.dtype(np.float64),
                repeats=True,
            ),
        ]


class GenParser(FileParser):
    def parse(self, key=None):
        self._results = dict()

        geometry = self.mainfile_obj.read()
        self.mainfile_obj.close()
        if geometry is None:
            return

        input_file = re.search(r'\<\<\< +\"(\S+)\"', geometry)
        if input_file:
            with open(os.path.join(self.maindir, input_file.group(1))) as f:
                geometry = f.read()

        geometry = geometry.strip().splitlines()
        n_atoms, lattice_type = geometry[0].split()
        n_atoms = int(n_atoms)
        elements = geometry[1].split()
        symbols, positions = [], []
        for n in range(2, n_atoms + 2):
            line = geometry[n].split()
            symbols.append(elements[int(line[1]) - 1])
            positions.append(line[2:5])
        lattice_vectors = None
        if lattice_type in ['S', 'F']:
            # line immediately after coordinates is origin
            lattice_vectors = (
                np.array(
                    [v.split() for v in geometry[n_atoms + 3 : n_atoms + 6]],
                    dtype=np.float64,
                )
                * ureg.angstrom
            )
        positions = np.array(positions, dtype=np.float64) * ureg.angstrom
        if lattice_type == 'F':
            # fractional coordinates
            positions = np.dot(positions, lattice_vectors)

        self._results['symbols'] = symbols
        self._results['positions'] = positions
        self._results['lattice_vectors'] = lattice_vectors


class HSDParser(FileParser):
    def __init__(self):
        super().__init__()
        self._re_section_value = re.compile(
            r'(?: *([A-Z]\w+) *= *(\w*) *\{|([A-Z][\w\[\]]+) *= *([^}]+)|(}))'
        )
        self.gen_parser = GenParser()

    @property
    def hsd(self):
        if self._file_handler is None:
            self._file_handler = open(self.mainfile)
        return self._file_handler

    def parse(self, key=None):
        self._results = dict(data=dict())

        line = self.hsd.readline()
        current_section = self._results['data']
        previous_sections = []
        # value as a whole block
        block = ''
        while line:
            matches = self._re_section_value.findall(line)
            for match in matches:
                section, sub_section, key, value, close = match
                if section:
                    current_section[section] = dict()
                    block = ''
                    previous_sections.append(current_section)
                    current_section = current_section[section]
                if sub_section:
                    current_section['_function'] = sub_section
                if key and value:
                    value = value.strip().replace('"', '').replace("'", '')
                    try:
                        value = float(value)
                    except Exception:
                        pass
                    current_section[key] = value
                if close:
                    if block:
                        current_section['_block'] = block
                    current_section = previous_sections[-1]
                    previous_sections.pop(-1)
            if not matches:
                block += line
            line = self.hsd.readline()

        self.gen_parser.mainfile = self.mainfile
        # TODO why is this necessary when you assign mainfile?
        # private variables should not be set
        self.gen_parser._mainfile_obj = io.StringIO(
            self._results.get('data', {}).get('Geometry', {}).get('_block', '')
        )
        self.gen_parser.parse()
        self._results['symbols'] = self.gen_parser.symbols
        self._results['positions'] = self.gen_parser.positions
        self._results['lattice_vectors'] = self.gen_parser.lattice_vectors


class OutParser(TextParser):
    def init_quantities(self):
        self._quantities = [
            Quantity('program_version', r'\|  DFTB\+ +(.+)', flatten=False, dtype=str),
            Quantity('input_file', r'Interpreting input file \'(\S+)\'', dtype=str),
            Quantity(
                'processed_input_file',
                r'Processed input in HSD format written to \'(\S+)\'',
                dtype=str,
            ),
            Quantity(
                'parser_version', r'Parser version: +(.+)', flatten=False, dtype=str
            ),
            Quantity(
                'sk_files',
                r'Reading SK-files:\s+([\s\S]+?)Done',
                str_operation=lambda x: [v.strip() for v in x.strip().splitlines()],
            ),
            Quantity(
                'input_parameters',
                r'Starting initialization\.\.\.\s+\-+\s+([\s\S]+?)\-{50}',
                sub_parser=TextParser(
                    quantities=[
                        Quantity(
                            'key_val',
                            rf'([A-Z].+?\: +[\w\-\+\. \(\)\:]+){re_n}',
                            str_operation=lambda x: [
                                v.strip() for v in x.strip().split(':', 1)
                            ],
                            repeats=True,
                        ),
                        Quantity(
                            'kpoints_weights',
                            rf'K\-points and weights: +([\s\S]+?){re_n} *{re_n}',
                            str_operation=lambda x: [
                                v.strip().split()[1:5] for v in x.strip().splitlines()
                            ],
                            dtype=np.dtype(np.float64),
                        ),
                    ]
                ),
            ),
            Quantity(
                'step',
                r'(Geometry step\: +\d+[\s\S]+?)(?:\-{50}|\Z)',
                repeats=True,
                sub_parser=TextParser(
                    quantities=[
                        Quantity(
                            'scf',
                            rf'\d+ +({re_f} +{re_f} +{re_f})',
                            repeats=True,
                            dtype=np.dtype(np.float64),
                        ),
                        Quantity(
                            'energy_total',
                            rf'Total Energy: +({re_f}) +H',
                            dtype=np.float64,
                            unit=ureg.hartree,
                        ),
                        Quantity(
                            'energy_total_t0',
                            rf'Extrapolated to 0: +({re_f}) +H',
                            dtype=np.float64,
                            unit=ureg.hartree,
                        ),
                        Quantity(
                            'energy_x_dftbp_total_mermin',
                            rf'Total Mermin free energy: +({re_f}) +H',
                            dtype=np.float64,
                            unit=ureg.hartree,
                        ),
                        Quantity(
                            'pressure',
                            rf'Pressure: +({re_f}) +au',
                            dtype=np.float64,
                            unit=ureg.hartree / ureg.bohr**3,
                        ),
                        Quantity(
                            'maximum_force',
                            rf'Maximal force component: +({re_f})',
                            dtype=np.float64,
                            unit=ureg.hartree / ureg.bohr,
                        ),
                    ]
                ),
            ),
        ]


class DFTBPlusParser(Parser):
    def __init__(self):
        self.out_parser = OutParser()
        self.hsd_parser = HSDParser()
        self.detailed_parser = DetailedParser()
        self.gen_parser = GenParser()

    def write_to_archive(self) -> None:
        self.out_parser.mainfile = self.mainfile
        self.out_parser.logger = self.logger
        self.hsd_parser.logger = self.logger
        self.gen_parser.logger = self.logger
        self.detailed_parser.logger = self.logger
        self.maindir = os.path.dirname(self.mainfile)

        sec_run = Run()
        self.archive.run.append(sec_run)
        sec_run.program = Program(
            name='DFTB+', version=self.out_parser.get('program_version')
        )

        def parse_system(source):
            sec_system = System()
            sec_run.system.append(sec_system)
            sec_system.atoms = Atoms(
                labels=source.get('symbols'),
                positions=source.get('positions'),
                lattice_vectors=source.get('lattice_vectors'),
            )

        input_file = self.out_parser.get(
            'processed_input_file', self.out_parser.get('input_file')
        )
        input_parameters = self.out_parser.input_parameters
        if input_file.endswith('.hsd'):
            self.hsd_parser.mainfile = os.path.join(self.maindir, input_file)
            input_parameters = self.hsd_parser.get('data')
            # parse initial structure
            parse_system(self.hsd_parser)

        for key in ['input_file', 'processed_input_file', 'parser_version']:
            setattr(sec_run, f'x_dftbp_{key}', self.out_parser.get(key))

        sec_method = Method()
        sec_run.method.append(sec_method)
        sec_tb = TB()
        sec_method.tb = sec_tb
        sec_tb.name = 'DFTB'
        sec_tb.x_dftbp_input_parameters = input_parameters
        sec_tb.x_dftbp_sk_files = self.out_parser.sk_files

        for step in self.out_parser.get('step', []):
            sec_scc = Calculation()
            sec_run.calculation.append(sec_scc)
            sec_scc.energy = Energy(
                total=EnergyEntry(value=step.energy_total),
                total_t0=EnergyEntry(value=step.energy_total_t0),
            )
            sec_scc.energy.x_dftbp_total_mermin = EnergyEntry(
                value=step.energy_x_dftbp_total_mermin
            )
            sec_scc.pressure = step.pressure
            for scf in step.get('scf', []):
                sec_scf = ScfIteration()
                sec_scc.scf_iteration.append(sec_scf)
                sec_scf.energy = Energy(
                    total=EnergyEntry(value=scf[0] * ureg.hartree),
                    change=scf[1] * ureg.hartree,
                )

        # reference the initial structure
        sec_run.calculation[0].system_ref = sec_run.system[0]

        # parse the final relaxed structure
        self.gen_parser.mainfile = os.path.join(self.maindir, 'geo_end.gen')
        if self.gen_parser.mainfile is not None:
            parse_system(self.gen_parser)
            sec_run.calculation[-1].system_ref = sec_run.system[-1]

        # properties in detailed.out
        # TODO add more properties e.g. charges
        self.detailed_parser.mainfile = os.path.join(self.maindir, 'detailed.out')
        for key, val in self.detailed_parser.items():
            if val is None:
                continue

            if key.startswith('energy_'):
                setattr(
                    sec_scc.energy, key.replace('energy_', ''), EnergyEntry(value=val)
                )
            elif key == 'forces':
                sec_scc.forces = Forces(
                    total=ForcesEntry(value=val * ureg.hartree / ureg.bohr)
                )
            elif key == 'fermi_level':
                sec_scc.energy.fermi = val
            elif key == 'pressure':
                sec_scc.pressure = val
            elif key == 'eigenvalues_occupations':
                sec_eigenvalues = BandEnergies()
                sec_scc.eigenvalues.append(sec_eigenvalues)
                # TODO handle spin polarization
                n_spin = 1
                eigs = np.vstack([kpoint[0] for kpoint in val.get('kpoint', [])])
                sec_eigenvalues.energies = (
                    np.reshape(eigs, (n_spin, *np.shape(eigs))) * ureg.hartree
                )
                occs = np.vstack([kpoint[1] for kpoint in val.get('kpoint', [])])
                sec_eigenvalues.occupations = np.reshape(
                    occs, (n_spin, *np.shape(occs))
                )
            elif key == 'dipole':
                sec_multipole = Multipoles()
                sec_scc.multipoles.append(sec_multipole)
                sec_multipole.dipole = MultipolesEntry(total=val)
