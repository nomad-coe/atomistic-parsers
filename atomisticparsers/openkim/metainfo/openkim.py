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
    Reference, JSON
)
from nomad.datamodel.metainfo import simulation, workflow


m_package = Package()


class Run(simulation.run.Run):

    m_def = Section(validate=False, extends_base_section=True)

    x_openkim_build_date = Quantity(
        type=str,
        shape=[],
        description='''
        build date as string
        ''')

    x_openkim_src_date = Quantity(
        type=str,
        shape=[],
        description='''
        date of last modification of the source as string
        ''')

    x_openkim_inserted_on = Quantity(
        type=str,
        shape=[],
        description='''
        title for the property
        ''')

    x_openkim_property_id = Quantity(
        type=str,
        shape=[],
        description='''
        unique ID of the property
        ''')

    x_openkim_property_title = Quantity(
        type=str,
        shape=[],
        description='''
        title for the property
        ''')

    x_openkim_property_description = Quantity(
        type=str,
        shape=[],
        description='''
        title for the property
        ''')

    x_openkim_instance_id = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        unique ID of the property
        ''')

    x_openkim_latest = Quantity(
        type=bool,
        shape=[],
        description='''
        ''')

    x_openkim_meta = Quantity(
        type=JSON,
        shape=[],
        description='''
        openkim metadata'''
    )


class System(simulation.system.System):

    m_def = Section(validate=False, extends_base_section=True)

    x_openkim_short_name = Quantity(
        type=str,
        shape=[1],
        description='''
        short name defining the crystal
        ''')

    x_openkim_space_group = Quantity(
        type=str,
        shape=[1],
        description='''
        short name defining the crystal
        ''')


class Elastic(workflow.Elastic):

    m_def = Section(validate=False, extends_base_section=True)

    x_openkim_excess = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='pascal',
        description='''
        total square numerical asymmetry of the calculated elastic constants
        ''')


class Phonon(workflow.Phonon):

    m_def = Section(validate=False, extends_base_section=True)

    x_openkim_wave_number = Quantity(
        type=np.dtype(np.float64),
        shape=['n_spin_channels', 'n_kpoints'],
        unit='1 / m',
        description='''
        wave numbers for each k-point
        ''')


class Workflow(workflow.Workflow):

    m_def = Section(validate=False, extends_base_section=True)

    x_openkim_property = Quantity(
        type=str,
        shape=[],
        description='''
        name of the property to be compared to nomad
        ''')

    x_openkim_nomad_rms_error = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        root mean square difference of the openkim data with respect to nomad
        ''')

    x_openkim_nomad_std = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        standard deviation of the nomad data
        ''')

    x_openkim_n_nomad_data = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        number of nomad entries with property corresponding to x_openkim_property
        ''')
