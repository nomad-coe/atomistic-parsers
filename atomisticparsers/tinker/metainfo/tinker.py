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
import simulationworkflowschema


m_package = Package()


class x_tinker_mdin_input_output_files(MCategory):
    """
    Parameters of mdin belonging to x_tinker_section_control_parameters.
    """

    m_def = Category()


class x_tinker_mdin_control_parameters(MCategory):
    """
    Parameters of mdin belonging to x_tinker_section_control_parameters.
    """

    m_def = Category()


class x_tinker_mdin_method(MCategory):
    """
    Parameters of mdin belonging to section method.
    """

    m_def = Category()


class x_tinker_mdout_single_configuration_calculation(MCategory):
    """
    Parameters of mdout belonging to section_single_configuration_calculation.
    """

    m_def = Category()


class x_tinker_mdout_method(MCategory):
    """
    Parameters of mdin belonging to section method.
    """

    m_def = Category()


class x_tinker_mdout_run(MCategory):
    """
    Parameters of mdin belonging to settings run.
    """

    m_def = Category()


class x_tinker_mdin_run(MCategory):
    """
    Parameters of mdin belonging to settings run.
    """

    m_def = Category()


class x_tinker_section_input_output_files(MSection):
    """
    Section to store input and output file names
    """

    m_def = Section(validate=False)


class x_tinker_section_control_parameters(MSection):
    """
    Section to store the input and output control parameters
    """

    m_def = Section(validate=False)

    x_tinker_inout_file_structure = Quantity(
        type=str,
        shape=[],
        description="""
        tinker input topology file.
        """,
    )

    x_tinker_inout_file_trajectory = Quantity(
        type=str,
        shape=[],
        description="""
        tinker output trajectory file.
        """,
    )

    x_tinker_inout_file_traj_coord = Quantity(
        type=str,
        shape=[],
        description="""
        tinker output trajectory file.
        """,
    )

    x_tinker_inout_file_traj_vel = Quantity(
        type=str,
        shape=[],
        description="""
        tinker output file for velocities in the trajectory.
        """,
    )

    x_tinker_inout_file_traj_force = Quantity(
        type=str,
        shape=[],
        description="""
        tinker output file for forces in the trajectory.
        """,
    )

    x_tinker_inout_file_output_coord = Quantity(
        type=str,
        shape=[],
        description="""
        tinker output coordinates file.
        """,
    )

    x_tinker_inout_file_output_vel = Quantity(
        type=str,
        shape=[],
        description="""
        tinker output velocities file.
        """,
    )

    x_tinker_inout_file_output_force = Quantity(
        type=str,
        shape=[],
        description="""
        tinker output forces file.
        """,
    )

    x_tinker_inout_file_input_coord = Quantity(
        type=str,
        shape=[],
        description="""
        tinker input coordinates file.
        """,
    )

    x_tinker_inout_file_input_vel = Quantity(
        type=str,
        shape=[],
        description="""
        tinker input velocities file.
        """,
    )

    x_tinker_inout_file_restart_coord = Quantity(
        type=str,
        shape=[],
        description="""
        tinker restart coordinates file.
        """,
    )

    x_tinker_inout_file_restart_vel = Quantity(
        type=str,
        shape=[],
        description="""
        tinker restart velocities file.
        """,
    )

    x_tinker_inout_file_output_log = Quantity(
        type=str,
        shape=[],
        description="""
        tinker MD output log file.
        """,
    )

    x_tinker_inout_control_number_of_steps = Quantity(
        type=str,
        shape=[],
        description="""
        tinker running environment and control parameters.
        """,
    )

    x_tinker_inout_control_polar_eps = Quantity(
        type=str,
        shape=[],
        description="""
        tinker running environment and control parameters.
        """,
    )

    x_tinker_inout_control_initial_temperature = Quantity(
        type=str,
        shape=[],
        description="""
        tinker running environment and control parameters.
        """,
    )

    x_tinker_inout_control_dielectric = Quantity(
        type=str,
        shape=[],
        description="""
        tinker running environment and control parameters.
        """,
    )

    x_tinker_inout_control_minimization = Quantity(
        type=str,
        shape=[],
        description="""
        tinker running environment and control parameters.
        """,
    )

    x_tinker_inout_control_integrator = Quantity(
        type=str,
        shape=[],
        description="""
        tinker running environment and control parameters.
        """,
    )

    x_tinker_inout_control_parameters = Quantity(
        type=str,
        shape=[],
        description="""
        tinker running environment and control parameters.
        """,
    )

    x_tinker_inout_control_verbose = Quantity(
        type=bool,
        shape=[],
        description="""
        tinker running environment and control parameters.
        """,
    )

    x_tinker_inout_control_a_axis = Quantity(
        type=str,
        shape=[],
        description="""
        tinker running environment and control parameters.
        """,
    )

    x_tinker_inout_control_b_axis = Quantity(
        type=str,
        shape=[],
        description="""
        tinker running environment and control parameters.
        """,
    )

    x_tinker_inout_control_c_axis = Quantity(
        type=str,
        shape=[],
        description="""
        tinker running environment and control parameters.
        """,
    )

    x_tinker_inout_control_alpha = Quantity(
        type=str,
        shape=[],
        description="""
        tinker running environment and control parameters.
        """,
    )

    x_tinker_inout_control_beta = Quantity(
        type=str,
        shape=[],
        description="""
        tinker running environment and control parameters.
        """,
    )

    x_tinker_inout_control_gamma = Quantity(
        type=str,
        shape=[],
        description="""
        tinker running environment and control parameters.
        """,
    )

    x_tinker_inout_control_tau_pressure = Quantity(
        type=str,
        shape=[],
        description="""
        tinker running environment and control parameters.
        """,
    )

    x_tinker_inout_control_tau_temperature = Quantity(
        type=str,
        shape=[],
        description="""
        tinker running environment and control parameters.
        """,
    )

    x_tinker_inout_control_debug = Quantity(
        type=bool,
        shape=[],
        description="""
        tinker running environment and control parameters.
        """,
    )

    x_tinker_inout_control_group = Quantity(
        type=bool,
        shape=[],
        description="""
        tinker running environment and control parameters.
        """,
    )

    x_tinker_inout_control_group_inter = Quantity(
        type=str,
        shape=[],
        description="""
        tinker running environment and control parameters.
        """,
    )

    x_tinker_inout_control_vib_roots = Quantity(
        type=str,
        shape=[],
        description="""
        tinker running environment and control parameters.
        """,
    )

    x_tinker_inout_control_spacegroup = Quantity(
        type=str,
        shape=[],
        description="""
        tinker running environment and control parameters.
        """,
    )

    x_tinker_inout_control_digits = Quantity(
        type=str,
        shape=[],
        description="""
        tinker running environment and control parameters.
        """,
    )

    x_tinker_inout_control_printout = Quantity(
        type=str,
        shape=[],
        description="""
        tinker running environment and control parameters.
        """,
    )

    x_tinker_inout_control_enforce_chirality = Quantity(
        type=bool,
        shape=[],
        description="""
        tinker running environment and control parameters.
        """,
    )

    x_tinker_inout_control_neighbor_list = Quantity(
        type=bool,
        shape=[],
        description="""
        tinker running environment and control parameters.
        """,
    )

    x_tinker_inout_control_vdw_cutoff = Quantity(
        type=str,
        shape=[],
        description="""
        tinker running environment and control parameters.
        """,
    )

    x_tinker_inout_control_vdw_correction = Quantity(
        type=bool,
        shape=[],
        description="""
        tinker running environment and control parameters.
        """,
    )

    x_tinker_inout_control_ewald = Quantity(
        type=bool,
        shape=[],
        description="""
        tinker running environment and control parameters.
        """,
    )

    x_tinker_inout_control_ewald_cutoff = Quantity(
        type=str,
        shape=[],
        description="""
        tinker running environment and control parameters.
        """,
    )

    x_tinker_inout_control_archive = Quantity(
        type=bool,
        shape=[],
        description="""
        tinker running environment and control parameters.
        """,
    )

    x_tinker_inout_control_barostat = Quantity(
        type=str,
        shape=[],
        description="""
        tinker running environment and control parameters.
        """,
    )

    x_tinker_inout_control_aniso_pressure = Quantity(
        type=bool,
        shape=[],
        description="""
        tinker running environment and control parameters.
        """,
    )

    x_tinker_inout_control_lights = Quantity(
        type=bool,
        shape=[],
        description="""
        tinker running environment and control parameters.
        """,
    )

    x_tinker_inout_control_randomseed = Quantity(
        type=str,
        shape=[],
        description="""
        tinker running environment and control parameters.
        """,
    )

    x_tinker_inout_control_saddlepoint = Quantity(
        type=bool,
        shape=[],
        description="""
        tinker running environment and control parameters.
        """,
    )

    x_tinker_inout_control_vdwtype = Quantity(
        type=str,
        shape=[],
        description="""
        tinker running environment and control parameters.
        """,
    )

    x_tinker_inout_control_title = Quantity(
        type=str,
        shape=[],
        description="""
        tinker running environment and control parameters.
        """,
    )

    x_tinker_inout_control_step_t = Quantity(
        type=str,
        shape=[],
        description="""
        tinker running environment and control parameters.
        """,
    )

    x_tinker_inout_control_step_dt = Quantity(
        type=str,
        shape=[],
        description="""
        tinker running environment and control parameters.
        """,
    )

    x_tinker_inout_control_random_number_generator_seed = Quantity(
        type=str,
        shape=[],
        description="""
        tinker running environment and control parameters.
        """,
    )

    x_tinker_inout_control_radiusrule = Quantity(
        type=str,
        shape=[],
        description="""
        tinker running environment and control parameters.
        """,
    )

    x_tinker_inout_control_radiustype = Quantity(
        type=str,
        shape=[],
        description="""
        tinker running environment and control parameters.
        """,
    )

    x_tinker_inout_control_radiussize = Quantity(
        type=str,
        shape=[],
        description="""
        tinker running environment and control parameters.
        """,
    )

    x_tinker_inout_control_epsilonrule = Quantity(
        type=str,
        shape=[],
        description="""
        tinker running environment and control parameters.
        """,
    )

    x_tinker_inout_control_rattle = Quantity(
        type=str,
        shape=[],
        description="""
        tinker running environment and control parameters.
        """,
    )

    x_tinker_inout_control_lambda = Quantity(
        type=str,
        shape=[],
        description="""
        tinker running environment and control parameters.
        """,
    )

    x_tinker_inout_control_mutate = Quantity(
        type=str,
        shape=[],
        description="""
        tinker running environment and control parameters.
        """,
    )

    x_tinker_inout_control_basin = Quantity(
        type=str,
        shape=[],
        description="""
        tinker running environment and control parameters.
        """,
    )

    x_tinker_inout_control_pme_grid = Quantity(
        type=str,
        shape=[],
        description="""
        tinker running environment and control parameters.
        """,
    )

    x_tinker_inout_control_pme_order = Quantity(
        type=str,
        shape=[],
        description="""
        tinker running environment and control parameters.
        """,
    )

    x_tinker_inout_control_nstep = Quantity(
        type=np.int32,
        shape=[],
        description="""
        tinker running environment and control parameters.
        """,
    )

    x_tinker_inout_control_initial_configuration_file = Quantity(
        type=str,
        shape=[],
        description="""
        tinker running environment and control parameters.
        """,
    )

    x_tinker_inout_control_final_configuration_file = Quantity(
        type=str,
        shape=[],
        description="""
        tinker running environment and control parameters.
        """,
    )

    x_tinker_inout_control_initial_trajectory_file = Quantity(
        type=str,
        shape=[],
        description="""
        tinker running environment and control parameters.
        """,
    )

    x_tinker_inout_control_restart_file = Quantity(
        type=str,
        shape=[],
        description="""
        tinker running environment and control parameters.
        """,
    )

    x_tinker_inout_control_archive_file = Quantity(
        type=str,
        shape=[],
        description="""
        tinker running environment and control parameters.
        """,
    )

    x_tinker_inout_control_force_field_file = Quantity(
        type=str,
        shape=[],
        description="""
        tinker running environment and control parameters.
        """,
    )

    x_tinker_inout_control_key_file = Quantity(
        type=str,
        shape=[],
        description="""
        tinker running environment and control parameters.
        """,
    )

    x_tinker_inout_control_coordinate_file_list = Quantity(
        type=str,
        shape=[],
        description="""
        tinker running environment and control parameters.
        """,
    )

    x_tinker_inout_control_structure_file = Quantity(
        type=str,
        shape=[],
        description="""
        tinker running environment and control parameters.
        """,
    )

    x_tinker_inout_control_parameter_file = Quantity(
        type=str,
        shape=[],
        description="""
        tinker running environment and control parameters.
        """,
    )

    x_tinker_inout_control_input_file = Quantity(
        type=str,
        shape=[],
        description="""
        tinker running environment and control parameters.
        """,
    )

    x_tinker_inout_control_topology_file = Quantity(
        type=str,
        shape=[],
        description="""
        tinker running environment and control parameters.
        """,
    )

    x_tinker_inout_control_configuration_file = Quantity(
        type=str,
        shape=[],
        description="""
        tinker running environment and control parameters.
        """,
    )

    x_tinker_inout_control_number_of_parameter_files = Quantity(
        type=np.int32,
        shape=[],
        description="""
        tinker running environment and control parameters.
        """,
    )

    x_tinker_inout_control_parameter_files = Quantity(
        type=str,
        shape=['x_tinker_inout_control_number_of_parameter_files'],
        description="""
        tinker running environment and control parameters.
        """,
    )

    x_tinker_section_input_output_files = SubSection(
        sub_section=SectionProxy('x_tinker_section_input_output_files'), repeats=True
    )


class x_tinker_section_atom_to_atom_type_ref(MSection):
    """
    Section to store atom label to atom type definition list
    """

    m_def = Section(validate=False)

    x_tinker_atom_to_atom_type_ref = Quantity(
        type=np.int64,
        shape=['number_of_atoms_per_type'],
        description="""
        Reference to the atoms of each atom type.
        """,
    )


class x_tinker_section_single_configuration_calculation(MSection):
    """
    section for gathering values for MD steps
    """

    m_def = Section(validate=False)


class System(runschema.system.System):
    m_def = Section(validate=False, extends_base_section=True)

    x_tinker_atom_positions_image_index = Quantity(
        type=np.int32,
        shape=['number_of_atoms', 3],
        unit='dimensionless',
        description="""
        PBC image flag index.
        """,
    )

    x_tinker_atom_positions_scaled = Quantity(
        type=np.float64,
        shape=['number_of_atoms', 3],
        unit='dimensionless',
        description="""
        Position of the atoms in a scaled format [0, 1].
        """,
    )

    x_tinker_atom_positions_wrapped = Quantity(
        type=np.float64,
        shape=['number_of_atoms', 3],
        unit='meter',
        description="""
        Position of the atoms wrapped back to the periodic box.
        """,
    )

    x_tinker_lattice_lengths = Quantity(
        type=np.float64,
        shape=[3],
        description="""
        Lattice dimensions in a vector. Vector includes [a, b, c] lengths.
        """,
    )

    x_tinker_lattice_angles = Quantity(
        type=np.float64,
        shape=[3],
        description="""
        Angles of lattice vectors. Vector includes [alpha, beta, gamma] in degrees.
        """,
    )

    x_tinker_dummy = Quantity(
        type=str,
        shape=[],
        description="""
        dummy
        """,
    )

    x_tinker_mdin_finline = Quantity(
        type=str,
        shape=[],
        description="""
        finline in mdin
        """,
    )

    x_tinker_traj_timestep_store = Quantity(
        type=str,
        shape=[],
        description="""
        tmp
        """,
    )

    x_tinker_traj_number_of_atoms_store = Quantity(
        type=str,
        shape=[],
        description="""
        tmp
        """,
    )

    x_tinker_traj_box_bound_store = Quantity(
        type=str,
        shape=[],
        description="""
        tmp
        """,
    )

    x_tinker_traj_box_bounds_store = Quantity(
        type=str,
        shape=[],
        description="""
        tmp
        """,
    )

    x_tinker_traj_variables_store = Quantity(
        type=str,
        shape=[],
        description="""
        tmp
        """,
    )

    x_tinker_traj_atoms_store = Quantity(
        type=str,
        shape=[],
        description="""
        tmp
        """,
    )


class MolecularDynamics(simulationworkflowschema.MolecularDynamics):
    m_def = Section(validate=False, extends_base_section=True)

    x_tinker_barostat_target_pressure = Quantity(
        type=np.float64,
        shape=[],
        unit='pascal',
        description="""
        MD barostat target pressure.
        """,
    )

    x_tinker_barostat_tau = Quantity(
        type=np.float64,
        shape=[],
        description="""
        MD barostat relaxation time.
        """,
    )

    x_tinker_barostat_type = Quantity(
        type=str,
        shape=[],
        description="""
        MD barostat type, valid values are defined in the barostat_type wiki page.
        """,
    )

    x_tinker_integrator_dt = Quantity(
        type=np.float64,
        shape=[],
        unit='second',
        description="""
        MD integration time step.
        """,
    )

    x_tinker_integrator_type = Quantity(
        type=str,
        shape=[],
        description="""
        MD integrator type, valid values are defined in the integrator_type wiki page.
        """,
    )

    x_tinker_periodicity_type = Quantity(
        type=str,
        shape=[],
        description="""
        Periodic boundary condition type in the sampling (non-PBC or PBC).
        """,
    )

    x_tinker_langevin_gamma = Quantity(
        type=np.float64,
        shape=[],
        unit='second',
        description="""
        Langevin thermostat damping factor.
        """,
    )

    x_tinker_number_of_steps_requested = Quantity(
        type=np.float64,
        shape=[],
        description="""
        Number of requested MD integration time steps.
        """,
    )

    x_tinker_thermostat_level = Quantity(
        type=str,
        shape=[],
        description="""
        MD thermostat level (see wiki: single, multiple, regional).
        """,
    )

    x_tinker_thermostat_target_temperature = Quantity(
        type=np.float64,
        shape=[],
        unit='kelvin',
        description="""
        MD thermostat target temperature.
        """,
    )

    x_tinker_thermostat_tau = Quantity(
        type=np.float64,
        shape=[],
        unit='second',
        description="""
        MD thermostat relaxation time.
        """,
    )

    x_tinker_thermostat_type = Quantity(
        type=str,
        shape=[],
        description="""
        MD thermostat type, valid values are defined in the thermostat_type wiki page.
        """,
    )


class GeometryOptimization(simulationworkflowschema.GeometryOptimization):
    m_def = Section(validate=False, extends_base_section=True)

    x_tiner_final_function_value = Quantity(
        type=np.float64,
        shape=[],
        unit='newton',
        description="""
        Final value of the energy.
        """,
    )

    x_tinker_final_rms_gradient = Quantity(
        type=np.float64,
        shape=[],
        unit='newton',
        description="""
        Tolerance value of the RMS gradient for structure minimization.
        """,
    )

    x_tinker_final_gradient_norm = Quantity(
        type=np.float64,
        shape=[],
        unit='newton',
        description="""
        Tolerance value of the RMS gradient for structure minimization.
        """,
    )

    x_tinker_final_gradient_norm = Quantity(
        type=np.float64,
        shape=[],
        unit='newton',
        description="""
        Tolerance value of the RMS gradient for structure minimization.
        """,
    )


class AtomParameters(runschema.method.AtomParameters):
    m_def = Section(validate=False, extends_base_section=True)

    x_tinker_atom_name = Quantity(
        type=str,
        shape=[],
        description="""
        Atom name of an atom in topology definition.
        """,
    )

    x_tinker_atom_type = Quantity(
        type=str,
        shape=[],
        description="""
        Atom type of an atom in topology definition.
        """,
    )

    x_tinker_atom_resid = Quantity(
        type=np.int32,
        shape=[],
        description="""
        """,
    )

    x_tinker_atom_element = Quantity(
        type=str,
        shape=[],
        description="""
        Atom type of an atom in topology definition.
        """,
    )

    x_tinker_atom_type_element = Quantity(
        type=str,
        shape=[],
        description="""
        Element symbol of an atom type.
        """,
    )

    x_tinker_atom_type_radius = Quantity(
        type=np.float64,
        shape=[],
        description="""
        van der Waals radius of an atom type.
        """,
    )

    number_of_atoms_per_type = Quantity(
        type=int,
        shape=[],
        description="""
        Number of atoms involved in this type.
        """,
    )


class Interaction(runschema.method.Interaction):
    m_def = Section(validate=False, extends_base_section=True)

    x_tinker_interaction_atom_to_atom_type_ref = Quantity(
        type=runschema.method.AtomParameters,
        shape=['number_of_atoms_per_interaction'],
        description="""
        Reference to the atom type of each interaction atoms.
        """,
    )

    x_tinker_number_of_defined_pair_interactions = Quantity(
        type=np.int32,
        shape=[],
        description="""
        Number of defined pair interactions (L-J pairs).
        """,
    )

    x_tinker_pair_interaction_atom_type_ref = Quantity(
        type=runschema.method.AtomParameters,
        shape=[
            'x_tinker_number_of_defined_pair_interactions',
            'number_of_atoms_per_interaction',
        ],
        description="""
        Reference to the atom type for pair interactions.
        """,
    )

    x_tinker_pair_interaction_parameters = Quantity(
        type=np.float64,
        shape=['x_tinker_number_of_defined_pair_interactions', 2],
        description="""
        Pair interactions parameters.
        """,
    )

    x_tinker_molecule_interaction_atom_to_atom_type_ref = Quantity(
        type=runschema.method.AtomParameters,
        shape=['number_of_atoms_per_interaction'],
        description="""
        Reference to the atom type of each molecule interaction atoms.
        """,
    )

    x_tinker_number_of_defined_molecule_pair_interactions = Quantity(
        type=np.int32,
        shape=[],
        description="""
        Number of defined pair interactions within a molecule (L-J pairs).
        """,
    )

    x_tinker_pair_molecule_interaction_parameters = Quantity(
        type=np.float64,
        shape=['number_of_defined_molecule_pair_interactions', 2],
        description="""
        Molecule pair interactions parameters.
        """,
    )

    x_tinker_pair_molecule_interaction_to_atom_type_ref = Quantity(
        type=runschema.method.AtomParameters,
        shape=[
            'x_tinker_number_of_defined_pair_interactions',
            'number_of_atoms_per_interaction',
        ],
        description="""
        Reference to the atom type for pair interactions within a molecule.
        """,
    )


class Run(runschema.run.Run):
    m_def = Section(validate=False, extends_base_section=True)

    x_tinker_program_version_date = Quantity(
        type=str,
        shape=[],
        description="""
        Program version date.
        """,
    )

    x_tinker_parallel_task_nr = Quantity(
        type=np.float64,
        shape=[],
        description="""
        Program task no.
        """,
    )

    x_tinker_build_osarch = Quantity(
        type=str,
        shape=[],
        description="""
        Program Build OS/ARCH
        """,
    )

    x_tinker_output_created_by_user = Quantity(
        type=str,
        shape=[],
        description="""
        Output file creator
        """,
    )

    x_tinker_most_severe_warning_level = Quantity(
        type=str,
        shape=[],
        description="""
        Highest tinker warning level in the run.
        """,
    )

    x_tinker_program_build_date = Quantity(
        type=str,
        shape=[],
        description="""
        Program Build date
        """,
    )

    x_tinker_program_citation = Quantity(
        type=str,
        shape=[],
        description="""
        Program citations
        """,
    )

    x_tinker_program_copyright = Quantity(
        type=str,
        shape=[],
        description="""
        Program copyright
        """,
    )

    x_tinker_number_of_tasks = Quantity(
        type=np.float64,
        shape=[],
        description="""
        Number of tasks in parallel program (MPI).
        """,
    )

    x_tinker_program_module_version = Quantity(
        type=str,
        shape=[],
        description="""
        tinker program module version.
        """,
    )

    x_tinker_program_license = Quantity(
        type=str,
        shape=[],
        description="""
        tinker program license.
        """,
    )

    x_tinker_xlo_xhi = Quantity(
        type=str,
        shape=[],
        description="""
        test
        """,
    )

    x_tinker_data_file_store = Quantity(
        type=str,
        shape=[],
        description="""
        Filename of data file
        """,
    )

    x_tinker_program_working_path = Quantity(
        type=str,
        shape=[],
        description="""
        tmp
        """,
    )

    x_tinker_program_execution_host = Quantity(
        type=str,
        shape=[],
        description="""
        tmp
        """,
    )

    x_tinker_program_execution_path = Quantity(
        type=str,
        shape=[],
        description="""
        tmp
        """,
    )

    x_tinker_program_module = Quantity(
        type=str,
        shape=[],
        description="""
        tmp
        """,
    )

    x_tinker_program_execution_date = Quantity(
        type=str,
        shape=[],
        description="""
        tmp
        """,
    )

    x_tinker_program_execution_time = Quantity(
        type=str,
        shape=[],
        description="""
        tmp
        """,
    )

    x_tinker_mdin_header = Quantity(
        type=str,
        shape=[],
        description="""
        tmp
        """,
    )

    x_tinker_mdin_wt = Quantity(
        type=str,
        shape=[],
        description="""
        tmp
        """,
    )

    x_tinker_section_control_parameters = SubSection(
        sub_section=SectionProxy('x_tinker_section_control_parameters'), repeats=True
    )

    x_tinker_control_parameters = Quantity(
        type=JSON,
        shape=[],
        description="""
        Parameters read from key file""",
    )


class Calculation(runschema.calculation.Calculation):
    m_def = Section(validate=False, extends_base_section=True)

    x_tinker_section_single_configuration_calculation = SubSection(
        sub_section=SectionProxy('x_tinker_section_single_configuration_calculation'),
        repeats=True,
    )


class VibrationalFrequencies(runschema.calculation.VibrationalFrequencies):
    m_def = Section(validate=False, extends_base_section=True)

    x_tinker_eigenvalues = Quantity(
        type=np.float64,
        shape=['n_frequencies'],
        description="""
        """,
    )
