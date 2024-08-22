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
    Package,
    Quantity,
    Section,
    SubSection,
    JSON,
)
import runschema.run  # pylint: disable=unused-import
import runschema.calculation  # pylint: disable=unused-import
import runschema.method  # pylint: disable=unused-import
import runschema.system  # pylint: disable=unused-import
import simulationworkflowschema


m_package = Package()


class Run(runschema.method.Method):
    m_def = Section(validate=False, extends_base_section=True)

    x_xtb_calculation_setup = Quantity(
        type=JSON,
        shape=[],
        description="""
        """,
    )


class TB(runschema.method.TB):
    m_def = Section(validate=False, extends_base_section=True)

    x_xtb_setup = Quantity(
        type=JSON,
        shape=[],
        description="""
        """,
    )


class Energy(runschema.calculation.Energy):
    m_def = Section(validate=False, extends_base_section=True)

    x_xtb_scc = SubSection(sub_section=runschema.calculation.EnergyEntry.m_def)

    x_xtb_isotropic_es = SubSection(sub_section=runschema.calculation.EnergyEntry.m_def)

    x_xtb_anisotropic_es = SubSection(
        sub_section=runschema.calculation.EnergyEntry.m_def
    )

    x_xtb_anistropic_xc = SubSection(
        sub_section=runschema.calculation.EnergyEntry.m_def
    )

    x_xtb_dispersion = SubSection(sub_section=runschema.calculation.EnergyEntry.m_def)

    x_xtb_repulsion = SubSection(sub_section=runschema.calculation.EnergyEntry.m_def)

    x_xtb_halogen_bond_corr = SubSection(
        sub_section=runschema.calculation.EnergyEntry.m_def
    )

    x_xtb_add_restraining = SubSection(
        sub_section=runschema.calculation.EnergyEntry.m_def
    )


class MultipolesEntry(runschema.calculation.MultipolesEntry):
    m_def = Section(validate=False, extends_base_section=True)

    x_xtb_q_only = Quantity(
        type=np.float64,
        shape=['n_multipoles'],
        description="""
        """,
    )

    x_xtb_q_plus_dip = Quantity(
        type=np.float64,
        shape=['n_multipoles'],
        description="""
        """,
    )


class GeometryOptimization(simulationworkflowschema.GeometryOptimization):
    m_def = Section(validate=False, extends_base_section=True)

    x_xtb_optimization_level = Quantity(
        type=str,
        shape=[],
        description="""
        """,
    )

    x_xtb_max_opt_cycles = Quantity(
        type=np.int32,
        shape=[],
        description="""
        """,
    )

    x_xtb_anc_micro_cycles = Quantity(
        type=np.int32,
        shape=[],
        description="""
        """,
    )

    x_xtb_n_degrees_freedom = Quantity(
        type=np.int32,
        shape=[],
        description="""
        """,
    )

    x_xtb_rf_solver = Quantity(
        type=str,
        shape=[],
        description="""
        """,
    )

    x_xtb_linear = Quantity(
        type=str,
        shape=[],
        description="""
        """,
    )

    x_xtb_hlow = Quantity(
        type=np.float64,
        shape=[],
        description="""
        """,
    )

    x_xtb_hmax = Quantity(
        type=np.float64,
        shape=[],
        description="""
        """,
    )

    x_xtb_s6 = Quantity(
        type=np.float64,
        shape=[],
        description="""
        """,
    )


class MolecularDynamics(simulationworkflowschema.MolecularDynamics):
    m_def = Section(validate=False, extends_base_section=True)

    x_xtb_md_time = Quantity(
        type=np.float64,
        shape=[],
        unit='s',
        description="""
        """,
    )

    x_xtb_scc_accuracy = Quantity(
        type=np.float64,
        shape=[],
        description="""
        """,
    )

    x_xtb_temperature = Quantity(
        type=np.float64,
        shape=[],
        unit='K',
        description="""
        """,
    )

    x_xtb_max_steps = Quantity(
        type=np.int32,
        shape=[],
        description="""
        """,
    )

    x_xtb_max_block_length = Quantity(
        type=np.int32,
        shape=[],
        description="""
        """,
    )

    x_xtb_dumpstep_trj = Quantity(
        type=np.float64,
        shape=[],
        unit='s',
        description="""
        """,
    )

    x_xtb_dumpstep_coords = Quantity(
        type=np.float64,
        shape=[],
        unit='s',
        description="""
        """,
    )

    x_xtb_h_atoms_mass = Quantity(
        type=np.float64,
        shape=[],
        unit='kg',
        description="""
        """,
    )

    x_xtb_n_degrees_freedom = Quantity(
        type=np.int32,
        shape=[],
        description="""
        """,
    )

    x_xtb_shake_bonds = Quantity(
        type=np.int32,
        shape=[],
        description="""
        """,
    )

    x_xtb_berendsen = Quantity(
        type=bool,
        shape=[],
        description="""
        """,
    )
