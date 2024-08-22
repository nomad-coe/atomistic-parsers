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
import typing  # pylint: disable=unused-import
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
import simulationworkflowschema


m_package = Package()


class x_dl_poly_section_md_molecule_type(MSection):
    """
    Section to store molecule type information
    """

    m_def = Section(validate=False)

    x_dl_poly_md_molecule_type_id = Quantity(
        type=np.int32,
        shape=[],
        description="""
        Molecule type id
        """,
    )

    x_dl_poly_md_molecule_type_name = Quantity(
        type=str,
        shape=[],
        description="""
        Molecule type name
        """,
    )


class x_dl_poly_section_md_topology(MSection):
    """
    Section modelling the MD topology
    """

    m_def = Section(validate=False)

    x_dl_poly_md_molecular_types = Quantity(
        type=np.int32,
        shape=[],
        description="""
        Number of molecular types in topology
        """,
    )

    x_dl_poly_section_md_molecule_type = SubSection(
        sub_section=SectionProxy('x_dl_poly_section_md_molecule_type'), repeats=True
    )


class MolecularDynamics(simulationworkflowschema.MolecularDynamics):
    m_def = Section(validate=False, extends_base_section=True)

    x_dl_poly_barostat_target_pressure = Quantity(
        type=np.float64,
        shape=[],
        unit='pascal',
        description="""
        MD barostat target pressure.
        """,
    )

    x_dl_poly_barostat_tau = Quantity(
        type=np.float64,
        shape=[],
        unit='second',
        description="""
        MD barostat relaxation time.
        """,
    )

    x_dl_poly_integrator_dt = Quantity(
        type=np.float64,
        shape=[],
        unit='second',
        description="""
        MD integration time step.
        """,
    )

    x_dl_poly_integrator_type = Quantity(
        type=str,
        shape=[],
        description="""
        MD integrator type, valid values are defined in the integrator_type wiki page.
        """,
    )

    x_dl_poly_number_of_steps_requested = Quantity(
        type=np.float64,
        shape=[],
        description="""
        Number of requested MD integration time steps.
        """,
    )

    x_dl_poly_thermostat_target_temperature = Quantity(
        type=np.float64,
        shape=[],
        unit='kelvin',
        description="""
        MD thermostat target temperature.
        """,
    )

    x_dl_poly_thermostat_tau = Quantity(
        type=np.float64,
        shape=[],
        unit='second',
        description="""
        MD thermostat relaxation time.
        """,
    )


class Run(runschema.run.Run):
    m_def = Section(validate=False, extends_base_section=True)

    x_dl_poly_program_version_date = Quantity(
        type=str,
        shape=[],
        description="""
        Program version date
        """,
    )

    x_dl_poly_system_description = Quantity(
        type=str,
        shape=[],
        description="""
        Simulation run title
        """,
    )


class System(runschema.system.System):
    m_def = Section(validate=False, extends_base_section=True)

    x_dl_poly_section_md_topology = SubSection(
        sub_section=SectionProxy('x_dl_poly_section_md_topology'), repeats=True
    )


class Method(runschema.method.Method):
    m_def = Section(validate=False, extends_base_section=True)

    x_dl_poly_step_number_equilibration = Quantity(
        type=np.int32,
        shape=[],
        description="""
        MD equilibration step number
        """,
    )

    x_dl_poly_step_number = Quantity(
        type=np.int32,
        shape=[],
        description="""
        MD total step number
        """,
    )

    x_dl_poly_thermostat_temperature = Quantity(
        type=np.float64,
        shape=[],
        description="""
        Thermostat coupling temperature
        """,
    )

    x_dl_poly_control_parameters = Quantity(
        type=JSON,
        shape=[],
        description="""
        """,
    )


class AtomParameters(runschema.method.AtomParameters):
    m_def = Section(validate=False, extends_base_section=True)

    x_dl_poly_nrept = Quantity(
        type=np.int32,
        shape=[],
        description="""
        """,
    )

    x_dl_poly_ifrz = Quantity(
        type=np.int32,
        shape=[],
        description="""
        """,
    )

    x_dl_poly_igrp = Quantity(
        type=np.int32,
        shape=[],
        description="""
        """,
    )


class Calculation(runschema.calculation.Calculation):
    m_def = Section(validate=False, extends_base_section=True)

    x_dl_poly_virial_configurational = Quantity(
        type=np.float64,
        shape=[],
        description="""
        """,
    )

    x_dl_poly_virial_short_range = Quantity(
        type=np.float64,
        shape=[],
        description="""
        """,
    )

    x_dl_poly_virial_coulomb = Quantity(
        type=np.float64,
        shape=[],
        description="""
        """,
    )

    x_dl_poly_virial_bond = Quantity(
        type=np.float64,
        shape=[],
        description="""
        """,
    )

    x_dl_poly_virial_angle = Quantity(
        type=np.float64,
        shape=[],
        description="""
        """,
    )

    x_dl_poly_virial_constraint = Quantity(
        type=np.float64,
        shape=[],
        description="""
        """,
    )

    x_dl_poly_virial_tethering = Quantity(
        type=np.float64,
        shape=[],
        description="""
        """,
    )

    x_dl_poly_volume = Quantity(
        type=np.float64,
        unit='m ** 3',
        shape=[],
        description="""
        """,
    )

    x_dl_poly_core_shell = Quantity(
        type=np.float64,
        shape=[],
        description="""
        """,
    )

    x_dl_poly_virial_potential_mean_force = Quantity(
        type=np.float64,
        shape=[],
        description="""
        """,
    )
