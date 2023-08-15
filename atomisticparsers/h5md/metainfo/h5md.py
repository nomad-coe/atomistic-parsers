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


class ParamEntry(MSection):
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

    unit = Quantity(
        type=str,
        shape=[],
        description='''
        Unit of the parameter as a string.
        ''')


class CalcEntry(MSection):
    '''
    Section describing a general type of calculation.
    '''

    m_def = Section(validate=False)

    kind = Quantity(
        type=str,
        shape=[],
        description='''
        Kind of the quantity.
        ''')

    value = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        Value of this contribution.
        ''')

    unit = Quantity(
        type=str,
        shape=[],
        description='''
        Unit of the parameter as a string.
        ''')


class AtomsGroup(simulation.system.AtomsGroup):
    '''
    Describes a group of atoms which may constitute a sub system as in the case of a
    molecule.
    '''

    m_def = Section(validate=False, extends_base_section=True,)

    x_h5md_parameters = SubSection(  # TODO should this be called parameters or attributes or what?
        sub_section=ParamEntry.m_def,
        description='''
        Contains additional information about the atom group .
        ''',
        repeats=True)


class Calculation(simulation.calculation.Calculation):

    m_def = Section(validate=False, extends_base_section=True,)

    x_h5md_custom_calculations = SubSection(
        sub_section=ParamEntry.m_def,
        description='''
        Contains other generic custom calculations that are not already defined.
        ''',
        repeats=True)


class Energy(simulation.calculation.Energy):

    m_def = Section(validate=False, extends_base_section=True,)

    x_h5md_energy_contributions = SubSection(
        sub_section=simulation.calculation.EnergyEntry.m_def,
        description='''
        Contains other custom energy contributions that are not already defined.
        ''',
        repeats=True)


class Author(MSection):
    '''
    Contains the specifications of the program.
    '''

    m_def = Section(validate=False)

    name = Quantity(
        type=str,
        shape=[],
        description='''
        Specifies the name of the author who generated the h5md file.
        ''',)

    email = Quantity(
        type=str,
        shape=[],
        description='''
        Author's email.
        ''',)


class Run(simulation.run.Run):

    m_def = Section(validate=False, extends_base_section=True,)

    # TODO Not sure how we are dealing with versioning with H5MD-NOMAD
    x_h5md_version = Quantity(
        type=np.dtype(np.int32),
        shape=[2],
        description='''
        Specifies the version of the h5md schema being followed.
        ''',)

    x_h5md_author = SubSection(sub_section=Author.m_def)

    x_h5md_creator = SubSection(sub_section=simulation.run.Program.m_def)