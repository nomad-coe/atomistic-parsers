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
from simulationparsers.utils import BasicParser


class AmberParser:
    def __init__(self):
        re_f = r'\-*\d+\.\d+E*\-*\+*\d*'
        self._parser = BasicParser(
            'Amber',
            units_mapping=dict(length=ureg.angstrom, energy=ureg.eV),
            program_version=r'Amber\s*(\d+)\s*(\w+)\s*(\d+)',
            # will only read initial coordinates
            auxilliary_files=r'(\S+\.inpcrd|\S+\.prmtop)',
            atom_positions=(
                rf'\d+\s+({re_f} +{re_f} +{re_f} +{re_f} +{re_f} +{re_f}[\s\S]+)',
                lambda x: x.strip().split()),
            atom_atom_number=r'\%FLAG ATOMIC_NUMBER\s*\%FORMAT\(.+\)\s*([\d\s]+)',
            energy_total=rf'NSTEP\s*ENERGY.+\s*\d+\s*({re_f})'
        )

    def parse(self, mainfile, archive, logger=None):
        self._parser.parse(mainfile, archive, logger=None)
