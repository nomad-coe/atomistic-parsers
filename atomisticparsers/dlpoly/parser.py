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


class DLPolyParser(BasicParser):
    def __init__(self):
        re_f = r'\-*\d+\.\d+E*\-*\+*\d*'

        def to_traj(val_in):
            labels = re.findall(r'([A-Z][a-z]*)[\+\-]* +\d+', val_in)
            traj = dict(atom_labels=labels)
            try:
                values = re.findall(rf'({re_f} +{re_f} +{re_f})', val_in)
                traj['lattice_vectors'] = np.array([v.split() for v in values[:3]], dtype=np.dtype(np.float64))
                values = np.array([v.split() for v in values[3:]], dtype=np.dtype(np.float64))
                values = np.reshape(values, (len(labels), len(values) // len(labels), 3))
                values = np.transpose(values, (1, 0, 2))
            except Exception:
                pass
            keys = ['atom_positions', 'atom_velocities', 'atom_forces']
            for n, value in enumerate(values):
                traj[keys[n]] = value
            return traj

        super().__init__(
            specifications=dict(
                name='parsers/dl-poly', code_name='DL_POLY',
                code_homepage='https://www.scd.stfc.ac.uk/Pages/DL_POLY.aspx',
                mainfile_contents_re=(r'\*\* DL_POLY \*\*')),
            units_mapping=dict(
                length=ureg.angstrom, energy=10 * ureg.joule / 6.02214076e+23,
                time=ureg.ps, mass=ureg.amu),
            auxilliary_files=r'(HISTORY)',
            program_version=r'\* +version\: +([\d\.]+.+?) +\*',
            lattice_vectors_atom_labels_atom_positions_atom_velocities_atom_forces=(
                r'step +\d+ +\d+ +\d+ +\d+ +.+([\s\S]+?)time',
                to_traj),
            energy_total=rf'\-+\s+\d+ +({re_f}) +(?:{re_f}\s+)+?{re_f}\n *\n')
