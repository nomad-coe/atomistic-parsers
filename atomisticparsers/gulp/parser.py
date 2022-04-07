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
import logging
import numpy as np
import re

from nomad.units import ureg
from nomad.parsing.file_parser import TextParser, Quantity
from nomad.datamodel.metainfo.simulation.run import Run, Program


re_f = r'[-+]?\d+\.\d*(?:[Ee][-+]\d+)?'
re_n = r'[\n\r]'


class MainfileParser(TextParser):
    def init_quantities(self):
        def to_coordinates(val_in):
            val_in = val_in.replace('*', '')
            data = dict(labels=[], coordinates=[], charges=[], occupancies=[])
            for val in val_in.strip().splitlines():
                val = val.strip().split()
                data['labels'].append(val[1])
                data['coordinates'].append(val[3:6])
                data['charges'].append(val[6])
                data['occupancies'].append(val[7])
            return data

        def to_species(val_in):
            data = dict(
                labels=[], types=[], atomic_numbers=[], atomic_masses=[], charges=[],
                covalent_radii=[], ionic_radii=[], vdw_radii=[]
            )
            for val in val_in.strip().splitlines():
                val = val.strip().split()
                data['labels'].append(val[0])
                data['types'].append(val[1])
                data['atomic_numbers'].append(val[2])
                data['atomic_masses'].append(val[3])
                data['charges'].append(val[4])
                data['covalent_radii'].append(val[5])
                data['ionic_radii'].append(val[6])
                data['vdw_radii'].append(val[7])
            return data

        def to_general_potentials(val_in):
            data = []
            pattern = re.compile(
                rf'([A-Z]\w*) +(\w+) +([A-Z]\w*) +(\w+) +(.+?) +({re_f}) +({re_f}) +({re_f}) +({re_f}) +({re_f}) +({re_f})\s*')
            for val in val_in.strip().splitlines():
                entry = re.match(pattern, val)
                if not entry:
                    continue
                val = entry.groups()
                data.append(dict(
                    atom_labels=[val[0], val[2]], functional_form=val[4], parameters={
                        'A': float(val[5]), 'B': float(val[6]), 'C': float(val[7]), 'D': float(val[8]),
                        'cufoff_min': float(val[9]), 'cutoff_max': float(val[10])
                    }
                ))
            return data

        self._quantities = [
            Quantity(
                'header',
                rf'(Version[\s\S]+?){re_n} *{re_n}',
                sub_parser=TextParser(quantities=[
                    Quantity('program_version', r'Version = (\S+)', dtype=str),
                    Quantity('task', r'\* +(\w+) +\- .+', repeats=True, dtype=str),
                    Quantity(
                        'title', r'\*\*\*\s+\* +(.+?) +\*\s+\*\*\*',
                        dtype=str, flatten=False
                    )

                ])
            ),
            Quantity(
                'input_configuration',
                r'(Input for Configuration.+\s*\*+[\s\S]+?)\*{80}',
                sub_parser=TextParser(quantities=[
                    Quantity('x_gulp_formula', r'Formula = (\S+)', dtype=str),
                    Quantity('x_gulp_pbc', r'Dimensionality = (\d+)', dtype=np.int32),
                    Quantity('x_gulp_space_group', rf'Space group \S+ +\: +(.+?) +{re_n}', dtype=str, flatten=False),
                    Quantity('x_gulp_patterson_group', rf'Patterson group +\: +(.+?) +{re_n}', dtype=str, flatten=False),
                    Quantity(
                        'lattice_vectors',
                        rf'Cartesian lattice vectors \(Angstroms\) \:\s+'
                        rf'((?:{re_f} +{re_f} +{re_f}\s+)+)',
                        dtype=np.dtype(np.float64), shape=[3, 3], unit=ureg.angstrom
                    ),
                    Quantity(
                        'coordinates',
                        rf'No\. + Atomic +x +y +z.+\s+Label.+\s*\-+\s+'
                        rf'((?:\d+ +[A-Z]\w* +\w+ +{re_f}.+\s+)+)',
                        str_operation=to_coordinates
                    )
                ])
            ),
            Quantity(
                'input_information',
                r'(General input information\s+\*\s*\*+[\s\S]+?)\*{80}',
                sub_parser=TextParser(quantities=[
                    Quantity(
                        'species',
                        rf'Species +Type.+\s+Number.+\s*\-+\s+'
                        rf'((?:[A-Z]\w* +\w+ +\d+.+\s+)+)',
                        str_operation=to_species
                    ),
                    # old format
                    # Quantity(
                    #     'general_potentials',
                    #     r'General interatomic potentials +\:\s+\-+\s*'
                    #     r'(Atom +Types +Potential +A +B +C +D +Cutoffs\(Ang\)\s+)'
                    #     r'(1 +2 +Min +Max\s*\-+\s*)'
                    #     r'((?:[A-Z]\w* +\w+ +[A-Z]\w* +\w+ +\w+.+\s*)+)',
                    #     str_operation=to_general_potentials
                    # ),
                    # new format
                    Quantity(
                        'general_potentials',
                        rf'General interatomic potentials +\:\s+\-+\s+'
                        rf'Atom +Types +Potential +Parameter([\s\S]+?){re_n} *{re_n}',
                        sub_parser=TextParser(quantities=[
                            Quantity(
                                'interaction',
                                r'([A-Z]\S* +(?:core|shell|c|s) +[A-Z]\S* +(?:core|shell|c|s) +[\s\S]+?\-{80})',
                                repeats=True, sub_parser=TextParser(quantities=[
                                    Quantity('atom_type', r'([A-Z]\S* +(?:core|shell|c|s))', repeats=True),
                                    Quantity('functional_form', r'([A-Z]\w+)'),
                                    Quantity(
                                        'key_parameter', rf'  ([ \w\-]+?) +((?:{re_f}|\d+))  ',
                                        repeats=True, str_operation=lambda x: [v.strip() for v in x.rsplit(' ', 1)]
                                    )
                                ])
                            )
                        ])
                    )

                    # Quantity(
                    #     'three_body_potentials',
                    #     r''
                    # )
                ])
            )
        ]


class GulpParser:
    def __init__(self):
        self.mainfile_parser = MainfileParser()

    def init_parser(self):
        self.mainfile_parser.logger = self.logger
        self.mainfile_parser.mainfile = self.filepath

    def parse(self, filepath, archive, logger):
        self.filepath = os.path.abspath(filepath)
        self.archive = archive
        self.maindir = os.path.dirname(self.filepath)
        self.logger = logger if logger is not None else logging

        self.init_parser()

        sec_run = self.archive.m_create(Run)
        header = self.mainfile_parser.get('header', {})
        sec_run.program = Program(version=header.get('program_version'))

