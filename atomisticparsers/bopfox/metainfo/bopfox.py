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
import numpy as np  # pylint: disable=unused-import

from nomad.metainfo import (  # pylint: disable=unused-import
    MSection,
    MCategory,
    Category,
    Package,
    Quantity,
    Section,
    SubSection,
    SectionProxy,
    Reference,
    JSON,
)
import runschema.run  # pylint: disable=unused-import
import runschema.calculation  # pylint: disable=unused-import
import runschema.method  # pylint: disable=unused-import
import runschema.system  # pylint: disable=unused-import


m_package = Package()


class Method(runschema.calculation.Method):
    m_def = Section(validate=False, extends_base_section=True)

    x_bopfox_simulation_parameters = Quantity(
        type=JSON,
        shape=[],
        description="""
        """,
    )


class Energy(runschema.calculation.Energy):
    m_def = Section(validate=False, extends_base_section=True)

    x_bopfox_bond = SubSection(
        sub_section=runschema.calculation.EnergyEntry.m_def,
        description="""
        Contains the value and information regarding the bond energy.
        """,
    )

    x_bopfox_prom = SubSection(
        sub_section=runschema.calculation.EnergyEntry.m_def,
        description="""
        Contains the value and information regarding the promotion energy.
        """,
    )

    x_bopfox_rep1 = SubSection(
        sub_section=runschema.calculation.EnergyEntry.m_def,
        description="""
        Contains the value and information regarding the first repulsion energy.
        """,
    )

    x_bopfox_rep2 = SubSection(
        sub_section=runschema.calculation.EnergyEntry.m_def,
        description="""
        Contains the value and information regarding the second repulsion energy.
        """,
    )

    x_bopfox_rep3 = SubSection(
        sub_section=runschema.calculation.EnergyEntry.m_def,
        description="""
        Contains the value and information regarding the third repulsion energy.
        """,
    )


class Forces(runschema.calculation.Forces):
    m_def = Section(validate=False, extends_base_section=True)

    x_bopfox_analytic = SubSection(
        sub_section=runschema.calculation.ForcesEntry.m_def,
        description="""
        Contains the value and information regarding the analytic forces.
        """,
    )

    x_bopfox_rep1 = SubSection(
        sub_section=runschema.calculation.ForcesEntry.m_def,
        description="""
        Contains the value and information regarding the first analytic forces.
        """,
    )

    x_bopfox_rep2 = SubSection(
        sub_section=runschema.calculation.ForcesEntry.m_def,
        description="""
        Contains the value and information regarding the second analytic forces.
        """,
    )

    x_bopfox_rep3 = SubSection(
        sub_section=runschema.calculation.ForcesEntry.m_def,
        description="""
        Contains the value and information regarding the third analytic forces.
        """,
    )


class x_bopfox_onsite_levels_value(runschema.calculation.AtomicValues):
    m_def = Section(validate=False)

    value = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description="""
        Value of the onsite level projected on orbital and spin channel.
        """,
    )


class x_bopfox_onsite_levels(runschema.calculation.Atomic):
    m_def = Section(validate=False)

    orbital_projected = SubSection(
        sub_section=x_bopfox_onsite_levels_value.m_def, repeats=True
    )


class Calculation(runschema.calculation.Calculation):
    m_def = Section(validate=False, extends_base_section=True)

    x_bopfox_onsite_levels = SubSection(
        sub_section=x_bopfox_onsite_levels.m_def, repeats=True
    )


class Interaction(runschema.method.Interaction):
    m_def = Section(validate=False, extends_base_section=True)

    x_bopfox_valence = Quantity(
        type=str,
        shape=['n_atoms'],
        description="""
        Valence of the atoms described by the interaction.
        """,
    )

    x_bopfox_chargetransfer = Quantity(
        type=np.dtype(np.float64),
        shape=['*'],
        description="""
        Charge transfer parameters.
        """,
    )

    x_bopfox_cutoff = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description="""
        Cutoff distance for the interaction.
        """,
    )

    x_bopfox_dcutoff = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description="""
        Distance from cutoff where the cutoff function is applied.
        """,
    )


class Model(runschema.method.Model):
    m_def = Section(validate=False, extends_base_section=True)

    x_bopfox_parameters = Quantity(
        type=JSON,
        shape=[],
        description="""
        """,
    )


class xTB(runschema.method.xTB):
    m_def = Section(validate=False, extends_base_section=True)

    x_bopfox_parameters = Quantity(
        type=JSON,
        shape=[],
        description="""
        """,
    )


class AtomParameters(runschema.method.AtomParameters):
    m_def = Section(validate=False, extends_base_section=True)

    x_bopfox_valenceorbitals = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description="""
        """,
    )

    x_bopfox_stonerintegral = Quantity(
        type=np.dtype(np.float64),
        shape=[3],
        description="""
        """,
    )
