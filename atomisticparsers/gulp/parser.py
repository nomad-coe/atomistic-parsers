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

from nomad.units import ureg
from nomad.parsing.file_parser import BasicParser


class GulpParser(BasicParser):
    def __init__(self):
        re_f = r'\-*\d+\.\d+E*e*\-*\+*\d*'

        super().__init__(
            specifications=dict(
                name='parsers/gulp', code_name='gulp', code_homepage='http://gulp.curtin.edu.au/gulp/',
                mainfile_contents_re=(
                    r'\s*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*'
                    r'\*\*\*\*\*\*\*\*\*\*\*\*\*\s*'
                    r'\s*\*\s*GENERAL UTILITY LATTICE PROGRAM\s*\*\s*')),
            units_mapping=dict(length=ureg.angstrom, energy=ureg.eV),
            # include code name to distinguish gamess and firefly
            program_version=r'Version *\= *([\d\.]+)',
            lattice_vectors=r'Final Cartesian lattice vectors \(Angstroms\) \:\s*([\d\.\-\s]+)',
            atom_labels_atom_positions_scaled=r'Final asymmetric unit coordinates \:\s*\-+\s*.+\s*.+\s*\-+\s*([\s\S]+?)\-{10}',
            energy_total=rf'Final energy *= *({re_f})|Total energy *\(eV\) *\= *({re_f})')
