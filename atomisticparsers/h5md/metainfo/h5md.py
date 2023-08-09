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
import numpy as np            # pylint: disable=unused-import
import typing                 # pylint: disable=unused-import
from nomad.metainfo import (  # pylint: disable=unused-import
    MSection, MCategory, Category, Package, Quantity, Section, SubSection, SectionProxy,
    Reference
)
from nomad.datamodel.metainfo import simulation


m_package = Package()

class Entry(MSection):
    '''
    Generic section defining a parameter name and value
    '''

    m_def = Section(validate=False)

    kind = Quantity(
        type=str,
        shape=[],
        description='''
        Name of the parameter.
        ''')

    value = Quantity(
        type=str,
        shape=[],
        description='''
        Value of the parameter as a string.
        ''')


class AtomsGroup(simulation.system.AtomsGroup):
    '''
    Describes a group of atoms which may constitute a sub system as in the case of a
    molecule.
    '''

    m_def = Section(validate=False, extends_base_section=True,)

    x_h5md_parameters = SubSection(
        sub_section=Entry.m_def,
        description='''
        Contains additional information about the atom group .
        ''',
        repeats=True)