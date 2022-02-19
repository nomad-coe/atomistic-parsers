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

from nomad.parsing.file_parser import BasicParser
from nomad.units import ureg


class LibAtomsParser(BasicParser):
    def __init__(self):
        re_f = r'\-*\d+\.\d+E*e*\-*\+*\d*'

        super().__init__(
            specifications=dict(
                name='parsers/lib-atoms', code_name='libAtoms', code_homepage='https://libatoms.github.io/',
                mainfile_contents_re=(r'\s*<GAP_params\s')),
            units_mapping=dict(length=ureg.angstrom, energy=ureg.eV),
            # necessary to include version due to auto type conversion
            program_version=r'(svn\_version\=\"\d+\")',
            atom_labels_atom_positions_atom_forces=r'(\<\!\[CDATA\[[A-Z][a-z]* +[\s\S]+?)(?:\<\!\[CDATA\[\d+\]\]\>|\Z)',
            energy_total=rf'slice\_sample energy\=({re_f})')
