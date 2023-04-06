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
from nomad.datamodel.metainfo import workflow

m_package = Package()

class System(simulation.system.System):

    m_def = Section(validate=False, extends_base_section=True,)

    x_hoomdblue_orientation = Quantity(
        type=np.dtype(np.float64),
        shape=['number_of_atoms', 4],
        unit='dimensionless',
        description='''
        Orientation of particles with internal degrees of freedom.
        ''',)

    x_hoomdblue_angmom = Quantity(
        type=np.dtype(np.float64),
        shape=['number_of_atoms', 4],
        unit='dimensionless',
        description='''
        Angular momentum of particles with internal degrees of freedom.
        ''',)

    x_hoomdblue_image = Quantity(
        type=np.dtype(np.float64),
        shape=['number_of_atoms', 3],
        unit='dimensionless',
        description='''
        Periodic image for each particle, for unwrapping the simulation, ie xu = x + image * L.
        ''',)


class AtomParameters(simulation.method.AtomParameters):

    m_def = Section(validate=False, extends_base_section=True,)

    x_hoomdblue_body = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        unit='dimensionless',
        description='''
        Indicates the rigid body index when a particle belongs to a rigid body, -1 indicates None.
        ''')

    x_hoomdblue_diameter = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='dimensionless',
        description='''
        Currently only used as a generic per atom quantity for various plugins.
        ''')

    x_hoomdblue_moment_inertia = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='dimensionless',
        description='''
        Defines rotation mass for particle with internal degrees of freedom.
        ''')

    x_hoomdblue_type_shapes = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='dimensionless',
        description='''
        Some parameter for alchemical transformations.
        ''')
