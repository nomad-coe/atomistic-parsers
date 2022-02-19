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

from nomad.metainfo import (  # pylint: disable=unused-import
    MSection, MCategory, Category, Package, Quantity, Section, SubSection, SectionProxy,
    Reference, JSON
)
from nomad.datamodel.metainfo import simulation


m_package = Package()


class AtomParameters(simulation.method.AtomParameters):

    m_def = Section(validate=False, extends_base_section=True)

    atom = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        -
        ''')

    x_dftbp_charge = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        -
        ''')


class System(simulation.system.System):

    m_def = Section(validate=False, extends_base_section=True)

    x_dftbp_atom_positions_X = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        -
        ''')

    x_dftbp_atom_positions_Y = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        -
        ''')

    x_dftbp_atom_positions_Z = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        -
        ''')


class BandEnergies(simulation.calculation.BandEnergies):

    m_def = Section(validate=False, extends_base_section=True)

    x_dftbp_eigenvalues_values = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        -
        ''')

    x_dftbp_eigenvalues_occupation = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        -
        ''')


class Calculation(simulation.calculation.Calculation):

    m_def = Section(validate=False, extends_base_section=True)

    x_dftbp_atom_forces_X = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        -
        ''')

    x_dftbp_atom_forces_Y = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        -
        ''')

    x_dftbp_atom_forces_Z = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        -
        ''')

    x_dftbp_force_max = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        -
        ''')

    x_dftbp_force_max_mov = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        -
        ''')


class Energy(simulation.calculation.Energy):

    m_def = Section(validate=False, extends_base_section=True)

    x_dftbp_total_mermin = SubSection(simulation.calculation.EnergyEntry.m_def)

    x_dftbp_band = SubSection(simulation.calculation.EnergyEntry.m_def)

    x_dftbp_ts = SubSection(simulation.calculation.EnergyEntry.m_def)

    x_dftbp_band_free = SubSection(simulation.calculation.EnergyEntry.m_def)

    x_dftbp_band_t0 = SubSection(simulation.calculation.EnergyEntry.m_def)

    x_dftbp_scc = SubSection(simulation.calculation.EnergyEntry.m_def)

    x_dftbp_dispersion = SubSection(simulation.calculation.EnergyEntry.m_def)

    x_dftbp_force_related = SubSection(simulation.calculation.EnergyEntry.m_def)


class Method(simulation.method.Method):

    m_def = Section(validate=False, extends_base_section=True)

    x_dftbp_input_parameters = Quantity(
        type=JSON,
        shape=[],
        description='''
        -
        ''')


class TB(simulation.method.TB):

    m_def = Section(validate=False, extends_base_section=True)

    x_dftbp_n_sk_files = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        ''')

    x_dftbp_sk_files = Quantity(
        type=str,
        shape=['x_dftbp_n_sk_files'],
        description='''
        ''')


class Run(simulation.run.Run):

    m_def = Section(validate=False, extends_base_section=True)

    x_dftbp_input_file = Quantity(
        type=str,
        shape=[],
        description='''
        ''')

    x_dftbp_processed_input_file = Quantity(
        type=str,
        shape=[],
        description='''
        ''')

    x_dftbp_parser_version = Quantity(
        type=str,
        shape=[],
        description='''
        ''')
