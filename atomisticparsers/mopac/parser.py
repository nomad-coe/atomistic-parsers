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


class MopacParser(BasicParser):
    def __init__(self):
        re_f = r'\-*\d+\.\d+E*e*\-*\+*\d*'

        def get_forces(val):
            try:
                forces = re.findall(rf'({re_f}) +KCAL\/ANGSTROM', val)
                return np.reshape(np.array(
                    forces, dtype=np.dtype(np.float64)), (len(forces) // 3, 3)) * (ureg.kcal / ureg.angstrom / 6.02214076e+23)
            except Exception:
                return []

        super().__init__(
            specifications=dict(
                name='parsers/mopac', code_name='MOPAC', domain='dft',
                mainfile_contents_re=r'\s*\*\*\s*MOPAC\s*([0-9a-zA-Z]*)\s*\*\*\s*',
                mainfile_mime_re=r'text/.*'),
            units_mapping=dict(length=ureg.angstrom, energy=ureg.eV),
            # include code name to distinguish gamess and firefly
            program_version=r'Version ([\w\.]+)',
            atom_labels_atom_positions=r'CARTESIAN COORDINATES\s*(1[\s\S]+?)\n *\n',
            energy_total=rf'TOTAL ENERGY *\= *({re_f})',
            atom_forces=(r'TYPE +VALUE +GRADIENT\s*([\s\S]+?)\n *\n', get_forces))
