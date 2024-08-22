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

from nomad.units import ureg
from simulationparsers.utils import BasicParser


class GromosParser:
    def __init__(self):
        re_f = r'\-*\d+\.\d+E*e*\-*\+*\d*'

        def get_positions(val):
            try:
                labels, positions = zip(
                    *re.findall(
                        rf'\d+ \w* +([A-Z])\w* +\d+ +({re_f} +{re_f} +{re_f})', val
                    )
                )
                positions = [v.split() for v in positions]
            except Exception:
                labels, positions = [], []
            return dict(atom_labels=labels, atom_positions=positions)

        self._parser = BasicParser(
            'Gromos',
            units_mapping=dict(length=ureg.nm, energy=ureg.kJ / 6.02214076e23),
            # include code name to distinguish gamess and firefly
            program_version=r'version *\: *([\d\.]+)',
            auxilliary_files=r'configuration read from\s*(\S+)',
            atom_labels_atom_positions=(
                rf'POSITION\s*(\d+ +[A-Z]+ +\w+ +\d+ +{re_f} +{re_f} +{re_f}[\s\S]+?)END',
                get_positions,
            ),
            energy_total=rf'E\_Total +\:\ *({re_f})',
            pressure=rf'pressure\: +({re_f})',
            time_step=r'TIMESTEP\s+(\d+)',
        )

    def parse(self, mainfile, archive, logger=None):
        self._parser.parse(mainfile, archive, logger=None)
