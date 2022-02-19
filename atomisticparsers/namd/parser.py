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

import re
import numpy as np

from nomad.units import ureg
from nomad.parsing.file_parser import BasicParser


class NAMDParser(BasicParser):
    def __init__(self):
        re_f = r'\-*\d+\.\d+E*e*\-*\+*\d*'

        def get_positions(val):
            try:
                labels, positions = zip(*re.findall(rf'ATOM +\d+ +([A-Z])\w* +\w+ +\d+ +({re_f} +{re_f} +{re_f})', val))
                positions = np.array([v.split() for v in positions], dtype=np.dtype(np.float64))
                return dict(atom_labels=labels, atom_positions=positions)
            except Exception:
                return dict()

        super().__init__(
            specifications=dict(
                name='parsers/namd', code_name='Namd', domain='dft',
                mainfile_contents_re=r'\s*Info:\s*NAMD\s*[0-9.]+\s*for\s*',
                mainfile_mime_re=r'text/.*'),
            units_mapping=dict(length=ureg.angstrom, energy=ureg.kcal / 6.02214076e+23),
            auxilliary_files=r'Info\: COORDINATE PDB +(\S+\.pdb)',
            # due to auto type conversion, we need to include NAMD to make it string
            program_version=r'Info\: (NAMD [\d\.]+) for',
            atom_labels_atom_positions=(rf'(ATOM +\d+ +\w+ +\w+ +\d+ +{re_f}[\s\S]+?)END', get_positions),
            energy_total=(rf'ENERGY\: +(\d+ +{re_f}.+)', lambda x: x.split()[10]))
