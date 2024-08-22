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
)
import runschema.run  # pylint: disable=unused-import
import runschema.calculation  # pylint: disable=unused-import
import runschema.method  # pylint: disable=unused-import
import runschema.system  # pylint: disable=unused-import
import simulationworkflowschema


m_package = Package()


class x_amber_mdin_method(MCategory):
    """
    Parameters of mdin belonging to section method.
    """

    m_def = Category()


class x_amber_mdout_single_configuration_calculation(MCategory):
    """
    Parameters of mdout belonging to section_single_configuration_calculation.
    """

    m_def = Category()


class x_amber_mdout_method(MCategory):
    """
    Parameters of mdin belonging to section method.
    """

    m_def = Category()


class x_amber_mdout_run(MCategory):
    """
    Parameters of mdin belonging to settings run.
    """

    m_def = Category()


class x_amber_mdin_run(MCategory):
    """
    Parameters of mdin belonging to settings run.
    """

    m_def = Category()


class x_amber_section_input_output_files(MSection):
    """
    Temperory variable to store input and output file keywords
    """

    m_def = Section(validate=False)


class x_amber_section_single_configuration_calculation(MSection):
    """
    section for gathering values for MD steps
    """

    m_def = Section(validate=False)


class System(runschema.system.System):
    m_def = Section(validate=False, extends_base_section=True)

    x_amber_atom_positions_image_index = Quantity(
        type=np.int32,
        shape=['number_of_atoms', 3],
        unit='dimensionless',
        description="""
        PBC image flag index.
        """,
    )

    x_amber_atom_positions_scaled = Quantity(
        type=np.float64,
        shape=['number_of_atoms', 3],
        unit='dimensionless',
        description="""
        Position of the atoms in a scaled format [0, 1].
        """,
    )

    x_amber_atom_positions_wrapped = Quantity(
        type=np.float64,
        shape=['number_of_atoms', 3],
        unit='meter',
        description="""
        Position of the atoms wrapped back to the periodic box.
        """,
    )

    x_amber_dummy = Quantity(
        type=str,
        shape=[],
        description="""
        dummy
        """,
    )

    x_amber_mdin_finline = Quantity(
        type=str,
        shape=[],
        description="""
        finline in mdin
        """,
    )

    x_amber_traj_timestep_store = Quantity(
        type=str,
        shape=[],
        description="""
        tmp
        """,
    )

    x_amber_traj_number_of_atoms_store = Quantity(
        type=str,
        shape=[],
        description="""
        tmp
        """,
    )

    x_amber_traj_box_bound_store = Quantity(
        type=str,
        shape=[],
        description="""
        tmp
        """,
    )

    x_amber_traj_box_bounds_store = Quantity(
        type=str,
        shape=[],
        description="""
        tmp
        """,
    )

    x_amber_traj_variables_store = Quantity(
        type=str,
        shape=[],
        description="""
        tmp
        """,
    )

    x_amber_traj_atoms_store = Quantity(
        type=str,
        shape=[],
        description="""
        tmp
        """,
    )


class MolecularDynamics(simulationworkflowschema.MolecularDynamics):
    m_def = Section(validate=False, extends_base_section=True)

    x_amber_barostat_target_pressure = Quantity(
        type=np.float64,
        shape=[],
        unit='pascal',
        description="""
        MD barostat target pressure.
        """,
    )

    x_amber_barostat_tau = Quantity(
        type=np.float64,
        shape=[],
        unit='second',
        description="""
        MD barostat relaxation time.
        """,
    )

    x_amber_barostat_type = Quantity(
        type=str,
        shape=[],
        description="""
        MD barostat type, valid values are defined in the barostat_type wiki page.
        """,
    )

    x_amber_integrator_dt = Quantity(
        type=np.float64,
        shape=[],
        unit='second',
        description="""
        MD integration time step.
        """,
    )

    x_amber_integrator_type = Quantity(
        type=str,
        shape=[],
        description="""
        MD integrator type, valid values are defined in the integrator_type wiki page.
        """,
    )

    x_amber_periodicity_type = Quantity(
        type=str,
        shape=[],
        description="""
        Periodic boundary condition type in the sampling (non-PBC or PBC).
        """,
    )

    x_amber_langevin_gamma = Quantity(
        type=np.float64,
        shape=[],
        unit='second',
        description="""
        Langevin thermostat damping factor.
        """,
    )

    x_amber_number_of_steps_requested = Quantity(
        type=np.float64,
        shape=[],
        description="""
        Number of requested MD integration time steps.
        """,
    )

    x_amber_thermostat_level = Quantity(
        type=str,
        shape=[],
        description="""
        MD thermostat level (see wiki: single, multiple, regional).
        """,
    )

    x_amber_thermostat_target_temperature = Quantity(
        type=np.float64,
        shape=[],
        unit='kelvin',
        description="""
        MD thermostat target temperature.
        """,
    )

    x_amber_thermostat_tau = Quantity(
        type=np.float64,
        shape=[],
        unit='second',
        description="""
        MD thermostat relaxation time.
        """,
    )

    x_amber_thermostat_type = Quantity(
        type=str,
        shape=[],
        description="""
        MD thermostat type, valid values are defined in the thermostat_type wiki page.
        """,
    )


class AtomParameters(runschema.method.AtomParameters):
    m_def = Section(validate=False, extends_base_section=True)

    x_amber_atom_type_element = Quantity(
        type=str,
        shape=[],
        description="""
        Element symbol of an atom type.
        """,
    )

    x_amber_atom_type_radius = Quantity(
        type=np.float64,
        shape=[],
        description="""
        van der Waals radius of an atom type.
        """,
    )


class Interaction(runschema.method.Interaction):
    m_def = Section(validate=False, extends_base_section=True)

    x_amber_interaction_atom_to_atom_type_ref = Quantity(
        type=runschema.method.AtomParameters,
        shape=['number_of_atoms_per_interaction'],
        description="""
        Reference to the atom type of each interaction atoms.
        """,
    )

    x_amber_number_of_defined_pair_interactions = Quantity(
        type=np.int32,
        shape=[],
        description="""
        Number of defined pair interactions (L-J pairs).
        """,
    )

    x_amber_pair_interaction_atom_type_ref = Quantity(
        type=runschema.method.AtomParameters,
        shape=[
            'x_amber_number_of_defined_pair_interactions',
            'number_of_atoms_per_interaction',
        ],
        description="""
        Reference to the atom type for pair interactions.
        """,
    )

    x_amber_pair_interaction_parameters = Quantity(
        type=np.float64,
        shape=['x_amber_number_of_defined_pair_interactions', 2],
        description="""
        Pair interactions parameters.
        """,
    )

    x_amber_molecule_interaction_atom_to_atom_type_ref = Quantity(
        type=runschema.method.AtomParameters,
        shape=['number_of_atoms_per_interaction'],
        description="""
        Reference to the atom type of each molecule interaction atoms.
        """,
    )

    x_amber_number_of_defined_molecule_pair_interactions = Quantity(
        type=np.int32,
        shape=[],
        description="""
        Number of defined pair interactions within a molecule (L-J pairs).
        """,
    )

    x_amber_pair_molecule_interaction_parameters = Quantity(
        type=np.float64,
        shape=['number_of_defined_molecule_pair_interactions', 2],
        description="""
        Molecule pair interactions parameters.
        """,
    )

    x_amber_pair_molecule_interaction_to_atom_type_ref = Quantity(
        type=runschema.method.AtomParameters,
        shape=[
            'x_amber_number_of_defined_pair_interactions',
            'number_of_atoms_per_interaction',
        ],
        description="""
        Reference to the atom type for pair interactions within a molecule.
        """,
    )


class Run(runschema.run.Run):
    m_def = Section(validate=False, extends_base_section=True)

    x_amber_program_version_date = Quantity(
        type=str,
        shape=[],
        description="""
        Program version date.
        """,
    )

    x_amber_xlo_xhi = Quantity(
        type=str,
        shape=[],
        description="""
        test
        """,
    )

    x_amber_data_file_store = Quantity(
        type=str,
        shape=[],
        description="""
        Filename of data file
        """,
    )

    x_amber_program_working_path = Quantity(
        type=str,
        shape=[],
        description="""
        tmp
        """,
    )

    x_amber_program_execution_host = Quantity(
        type=str,
        shape=[],
        description="""
        tmp
        """,
    )

    x_amber_program_execution_path = Quantity(
        type=str,
        shape=[],
        description="""
        tmp
        """,
    )

    x_amber_program_module = Quantity(
        type=str,
        shape=[],
        description="""
        tmp
        """,
    )

    x_amber_program_execution_date = Quantity(
        type=str,
        shape=[],
        description="""
        tmp
        """,
    )

    x_amber_program_execution_time = Quantity(
        type=str,
        shape=[],
        description="""
        tmp
        """,
    )

    x_amber_mdin_header = Quantity(
        type=str,
        shape=[],
        description="""
        tmp
        """,
    )

    x_amber_mdin_wt = Quantity(
        type=str,
        shape=[],
        description="""
        tmp
        """,
    )

    x_amber_section_input_output_files = SubSection(
        sub_section=SectionProxy('x_amber_section_input_output_files'), repeats=False
    )


class Calculation(runschema.calculation.Calculation):
    m_def = Section(validate=False, extends_base_section=True)

    x_amber_section_single_configuration_calculation = SubSection(
        sub_section=SectionProxy('x_amber_section_single_configuration_calculation'),
        repeats=True,
    )
