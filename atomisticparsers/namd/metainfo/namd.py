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


class x_namd_mdin_input_output_files(MCategory):
    """
    Parameters of mdin belonging to x_namd_section_control_parameters.
    """

    m_def = Category()


class x_namd_mdin_control_parameters(MCategory):
    """
    Parameters of mdin belonging to x_namd_section_control_parameters.
    """

    m_def = Category()


class x_namd_mdin_method(MCategory):
    """
    Parameters of mdin belonging to section method.
    """

    m_def = Category()


class x_namd_mdout_single_configuration_calculation(MCategory):
    """
    Parameters of mdout belonging to section_single_configuration_calculation.
    """

    m_def = Category()


class x_namd_mdout_method(MCategory):
    """
    Parameters of mdin belonging to section method.
    """

    m_def = Category()


class x_namd_mdout_run(MCategory):
    """
    Parameters of mdin belonging to settings run.
    """

    m_def = Category()


class x_namd_mdin_run(MCategory):
    """
    Parameters of mdin belonging to settings run.
    """

    m_def = Category()


class x_namd_section_input_output_files(MSection):
    """
    Section to store input and output file names
    """

    m_def = Section(validate=False)

    x_namd_inout_file_structure = Quantity(
        type=str,
        shape=[],
        description="""
        NAMD input topology file.
        """,
    )

    x_namd_inout_file_traj_coord = Quantity(
        type=str,
        shape=[],
        description="""
        NAMD output trajectory file.
        """,
    )

    x_namd_inout_file_traj_vel = Quantity(
        type=str,
        shape=[],
        description="""
        NAMD output file for velocities in the trajectory.
        """,
    )

    x_namd_inout_file_traj_force = Quantity(
        type=str,
        shape=[],
        description="""
        NAMD output file for forces in the trajectory.
        """,
    )

    x_namd_inout_file_output_coord = Quantity(
        type=str,
        shape=[],
        description="""
        NAMD output coordinates file.
        """,
    )

    x_namd_inout_file_output_vel = Quantity(
        type=str,
        shape=[],
        description="""
        NAMD output velocities file.
        """,
    )

    x_namd_inout_file_output_force = Quantity(
        type=str,
        shape=[],
        description="""
        NAMD output forces file.
        """,
    )

    x_namd_inout_file_input_coord = Quantity(
        type=str,
        shape=[],
        description="""
        NAMD input coordinates file.
        """,
    )

    x_namd_inout_file_input_vel = Quantity(
        type=str,
        shape=[],
        description="""
        NAMD input velocities file.
        """,
    )

    x_namd_inout_file_restart_coord = Quantity(
        type=str,
        shape=[],
        description="""
        NAMD restart coordinates file.
        """,
    )

    x_namd_inout_file_restart_vel = Quantity(
        type=str,
        shape=[],
        description="""
        NAMD restart velocities file.
        """,
    )

    x_namd_inout_file_fftw_datafile = Quantity(
        type=str,
        shape=[],
        description="""
        NAMD FFTW data file.
        """,
    )

    x_namd_inout_file_mdlog = Quantity(
        type=str,
        shape=[],
        description="""
        NAMD MD output log file.
        """,
    )


class x_namd_section_control_parameters(MSection):
    """
    Section to store the input and output control parameters
    """

    m_def = Section(validate=False)

    x_namd_inout_control_timestep = Quantity(
        type=str,
        shape=[],
        description="""
        NAMD running environment and control parameters.
        """,
    )

    x_namd_inout_control_number_of_steps = Quantity(
        type=str,
        shape=[],
        description="""
        NAMD running environment and control parameters.
        """,
    )

    x_namd_inout_control_steps_per_cycle = Quantity(
        type=str,
        shape=[],
        description="""
        NAMD running environment and control parameters.
        """,
    )

    x_namd_inout_control_periodic_cell_basis_1 = Quantity(
        type=str,
        shape=[],
        description="""
        NAMD running environment and control parameters.
        """,
    )

    x_namd_inout_control_periodic_cell_basis_2 = Quantity(
        type=str,
        shape=[],
        description="""
        NAMD running environment and control parameters.
        """,
    )

    x_namd_inout_control_periodic_cell_basis_3 = Quantity(
        type=str,
        shape=[],
        description="""
        NAMD running environment and control parameters.
        """,
    )

    x_namd_inout_control_periodic_cell_center = Quantity(
        type=str,
        shape=[],
        description="""
        NAMD running environment and control parameters.
        """,
    )

    x_namd_inout_control_load_balancer = Quantity(
        type=str,
        shape=[],
        description="""
        NAMD running environment and control parameters.
        """,
    )

    x_namd_inout_control_load_balancing_strategy = Quantity(
        type=str,
        shape=[],
        description="""
        NAMD running environment and control parameters.
        """,
    )

    x_namd_inout_control_ldb_period = Quantity(
        type=str,
        shape=[],
        description="""
        NAMD running environment and control parameters.
        """,
    )

    x_namd_inout_control_first_ldb_timestep = Quantity(
        type=str,
        shape=[],
        description="""
        NAMD running environment and control parameters.
        """,
    )

    x_namd_inout_control_last_ldb_timestep = Quantity(
        type=str,
        shape=[],
        description="""
        NAMD running environment and control parameters.
        """,
    )

    x_namd_inout_control_ldb_background_scaling = Quantity(
        type=str,
        shape=[],
        description="""
        NAMD running environment and control parameters.
        """,
    )

    x_namd_inout_control_hom_background_scaling = Quantity(
        type=str,
        shape=[],
        description="""
        NAMD running environment and control parameters.
        """,
    )

    x_namd_inout_control_pme_background_scaling = Quantity(
        type=str,
        shape=[],
        description="""
        NAMD running environment and control parameters.
        """,
    )

    x_namd_inout_control_min_atoms_per_patch = Quantity(
        type=str,
        shape=[],
        description="""
        NAMD running environment and control parameters.
        """,
    )

    x_namd_inout_control_initial_temperature = Quantity(
        type=str,
        shape=[],
        description="""
        NAMD running environment and control parameters.
        """,
    )

    x_namd_inout_control_center_of_mass_moving_initially = Quantity(
        type=str,
        shape=[],
        description="""
        NAMD running environment and control parameters.
        """,
    )

    x_namd_inout_control_dielectric = Quantity(
        type=str,
        shape=[],
        description="""
        NAMD running environment and control parameters.
        """,
    )

    x_namd_inout_control_excluded_species_or_groups = Quantity(
        type=str,
        shape=[],
        description="""
        NAMD running environment and control parameters.
        """,
    )

    x_namd_inout_control_1_4_electrostatics_scale = Quantity(
        type=str,
        shape=[],
        description="""
        NAMD running environment and control parameters.
        """,
    )

    x_namd_inout_control_traj_dcd_filename = Quantity(
        type=str,
        shape=[],
        description="""
        NAMD running environment and control parameters.
        """,
    )

    x_namd_inout_control_traj_dcd_frequency = Quantity(
        type=str,
        shape=[],
        description="""
        NAMD running environment and control parameters.
        """,
    )

    x_namd_inout_control_traj_dcd_first_step = Quantity(
        type=str,
        shape=[],
        description="""
        NAMD running environment and control parameters.
        """,
    )

    x_namd_inout_control_velocity_dcd_filename = Quantity(
        type=str,
        shape=[],
        description="""
        NAMD running environment and control parameters.
        """,
    )

    x_namd_inout_control_velocity_dcd_frequency = Quantity(
        type=str,
        shape=[],
        description="""
        NAMD running environment and control parameters.
        """,
    )

    x_namd_inout_control_velocity_dcd_first_step = Quantity(
        type=str,
        shape=[],
        description="""
        NAMD running environment and control parameters.
        """,
    )

    x_namd_inout_control_force_dcd_filename = Quantity(
        type=str,
        shape=[],
        description="""
        NAMD running environment and control parameters.
        """,
    )

    x_namd_inout_control_force_dcd_frequency = Quantity(
        type=str,
        shape=[],
        description="""
        NAMD running environment and control parameters.
        """,
    )

    x_namd_inout_control_force_dcd_first_step = Quantity(
        type=str,
        shape=[],
        description="""
        NAMD running environment and control parameters.
        """,
    )

    x_namd_inout_control_output_filename = Quantity(
        type=str,
        shape=[],
        description="""
        NAMD running environment and control parameters.
        """,
    )

    x_namd_inout_control_binary_output = Quantity(
        type=str,
        shape=[],
        description="""
        NAMD running environment and control parameters.
        """,
    )

    x_namd_inout_control_restart_filename = Quantity(
        type=str,
        shape=[],
        description="""
        NAMD running environment and control parameters.
        """,
    )

    x_namd_inout_control_restart_frequency = Quantity(
        type=str,
        shape=[],
        description="""
        NAMD running environment and control parameters.
        """,
    )

    x_namd_inout_control_binary_restart = Quantity(
        type=str,
        shape=[],
        description="""
        NAMD running environment and control parameters.
        """,
    )

    x_namd_inout_control_switching = Quantity(
        type=str,
        shape=[],
        description="""
        NAMD running environment and control parameters.
        """,
    )

    x_namd_inout_control_switching_on = Quantity(
        type=str,
        shape=[],
        description="""
        NAMD running environment and control parameters.
        """,
    )

    x_namd_inout_control_switching_off = Quantity(
        type=str,
        shape=[],
        description="""
        NAMD running environment and control parameters.
        """,
    )

    x_namd_inout_control_pairlist_distance = Quantity(
        type=str,
        shape=[],
        description="""
        NAMD running environment and control parameters.
        """,
    )

    x_namd_inout_control_pairlist_shrink_rate = Quantity(
        type=str,
        shape=[],
        description="""
        NAMD running environment and control parameters.
        """,
    )

    x_namd_inout_control_pairlist_grow_rate = Quantity(
        type=str,
        shape=[],
        description="""
        NAMD running environment and control parameters.
        """,
    )

    x_namd_inout_control_pairlist_trigger = Quantity(
        type=str,
        shape=[],
        description="""
        NAMD running environment and control parameters.
        """,
    )

    x_namd_inout_control_pairlists_per_cycle = Quantity(
        type=str,
        shape=[],
        description="""
        NAMD running environment and control parameters.
        """,
    )

    x_namd_inout_control_pairlists = Quantity(
        type=str,
        shape=[],
        description="""
        NAMD running environment and control parameters.
        """,
    )

    x_namd_inout_control_margin = Quantity(
        type=str,
        shape=[],
        description="""
        NAMD running environment and control parameters.
        """,
    )

    x_namd_inout_control_hydrogen_group_cutoff = Quantity(
        type=str,
        shape=[],
        description="""
        NAMD running environment and control parameters.
        """,
    )

    x_namd_inout_control_patch_dimension = Quantity(
        type=str,
        shape=[],
        description="""
        NAMD running environment and control parameters.
        """,
    )

    x_namd_inout_control_energy_output_steps = Quantity(
        type=str,
        shape=[],
        description="""
        NAMD running environment and control parameters.
        """,
    )

    x_namd_inout_control_crossterm_energy = Quantity(
        type=str,
        shape=[],
        description="""
        NAMD running environment and control parameters.
        """,
    )

    x_namd_inout_control_timing_output_steps = Quantity(
        type=str,
        shape=[],
        description="""
        NAMD running environment and control parameters.
        """,
    )

    x_namd_inout_control_velocity_rescale_freq = Quantity(
        type=str,
        shape=[],
        description="""
        NAMD running environment and control parameters.
        """,
    )

    x_namd_inout_control_velocity_rescale_temp = Quantity(
        type=str,
        shape=[],
        description="""
        NAMD running environment and control parameters.
        """,
    )

    x_namd_inout_control_velocity_reassignment_freq = Quantity(
        type=str,
        shape=[],
        description="""
        NAMD running environment and control parameters.
        """,
    )

    x_namd_inout_control_velocity_reassignment_temp = Quantity(
        type=str,
        shape=[],
        description="""
        NAMD running environment and control parameters.
        """,
    )

    x_namd_inout_control_velocity_reassignment_incr = Quantity(
        type=str,
        shape=[],
        description="""
        NAMD running environment and control parameters.
        """,
    )

    x_namd_inout_control_velocity_reassignment_hold = Quantity(
        type=str,
        shape=[],
        description="""
        NAMD running environment and control parameters.
        """,
    )

    x_namd_inout_control_lowe_andersen_dynamics = Quantity(
        type=str,
        shape=[],
        description="""
        NAMD running environment and control parameters.
        """,
    )

    x_namd_inout_control_lowe_andersen_temperature = Quantity(
        type=str,
        shape=[],
        description="""
        NAMD running environment and control parameters.
        """,
    )

    x_namd_inout_control_lowe_andersen_rate = Quantity(
        type=str,
        shape=[],
        description="""
        NAMD running environment and control parameters.
        """,
    )

    x_namd_inout_control_lowe_andersen_cutoff = Quantity(
        type=str,
        shape=[],
        description="""
        NAMD running environment and control parameters.
        """,
    )

    x_namd_inout_control_langevin_dynamics = Quantity(
        type=str,
        shape=[],
        description="""
        NAMD running environment and control parameters.
        """,
    )

    x_namd_inout_control_langevin_temperature = Quantity(
        type=str,
        shape=[],
        description="""
        NAMD running environment and control parameters.
        """,
    )

    x_namd_inout_control_langevin_integrator = Quantity(
        type=str,
        shape=[],
        description="""
        NAMD running environment and control parameters.
        """,
    )

    x_namd_inout_control_langevin_damping_file = Quantity(
        type=str,
        shape=[],
        description="""
        NAMD running environment and control parameters.
        """,
    )

    x_namd_inout_control_langevin_damping_column = Quantity(
        type=str,
        shape=[],
        description="""
        NAMD running environment and control parameters.
        """,
    )

    x_namd_inout_control_langevin_damping_coefficient_unit = Quantity(
        type=str,
        shape=[],
        description="""
        NAMD running environment and control parameters.
        """,
    )

    x_namd_inout_control_langevin_dynamics_not_applied_to = Quantity(
        type=str,
        shape=[],
        description="""
        NAMD running environment and control parameters.
        """,
    )

    x_namd_inout_control_temperature_coupling = Quantity(
        type=str,
        shape=[],
        description="""
        NAMD running environment and control parameters.
        """,
    )

    x_namd_inout_control_coupling_temperature = Quantity(
        type=str,
        shape=[],
        description="""
        NAMD running environment and control parameters.
        """,
    )

    x_namd_inout_control_berendsen_pressure_coupling = Quantity(
        type=str,
        shape=[],
        description="""
        NAMD running environment and control parameters.
        """,
    )

    x_namd_inout_control_berendsen_compressibility_estimate = Quantity(
        type=str,
        shape=[],
        description="""
        NAMD running environment and control parameters.
        """,
    )

    x_namd_inout_control_berendsen_relaxation_time = Quantity(
        type=str,
        shape=[],
        description="""
        NAMD running environment and control parameters.
        """,
    )

    x_namd_inout_control_berendsen_coupling_frequency = Quantity(
        type=str,
        shape=[],
        description="""
        NAMD running environment and control parameters.
        """,
    )

    x_namd_inout_control_langevin_piston_pressure_control = Quantity(
        type=str,
        shape=[],
        description="""
        NAMD running environment and control parameters.
        """,
    )

    x_namd_inout_control_target_pressure = Quantity(
        type=str,
        shape=[],
        description="""
        NAMD running environment and control parameters.
        """,
    )

    x_namd_inout_control_langevin_oscillation_period = Quantity(
        type=str,
        shape=[],
        description="""
        NAMD running environment and control parameters.
        """,
    )

    x_namd_inout_control_langevin_decay_time = Quantity(
        type=str,
        shape=[],
        description="""
        NAMD running environment and control parameters.
        """,
    )

    x_namd_inout_control_langevin_piston_temperature = Quantity(
        type=str,
        shape=[],
        description="""
        NAMD running environment and control parameters.
        """,
    )

    x_namd_inout_control_pressure_control = Quantity(
        type=str,
        shape=[],
        description="""
        NAMD running environment and control parameters.
        """,
    )

    x_namd_inout_control_initial_strain_rate = Quantity(
        type=str,
        shape=[],
        description="""
        NAMD running environment and control parameters.
        """,
    )

    x_namd_inout_control_cell_fluctuation = Quantity(
        type=str,
        shape=[],
        description="""
        NAMD running environment and control parameters.
        """,
    )

    x_namd_inout_control_particle_mesh_ewald = Quantity(
        type=str,
        shape=[],
        description="""
        NAMD running environment and control parameters.
        """,
    )

    x_namd_inout_control_pme_tolerance = Quantity(
        type=str,
        shape=[],
        description="""
        NAMD running environment and control parameters.
        """,
    )

    x_namd_inout_control_pme_ewald_coefficient = Quantity(
        type=str,
        shape=[],
        description="""
        NAMD running environment and control parameters.
        """,
    )

    x_namd_inout_control_pme_interpolation_order = Quantity(
        type=str,
        shape=[],
        description="""
        NAMD running environment and control parameters.
        """,
    )

    x_namd_inout_control_pme_grid_dimensions = Quantity(
        type=str,
        shape=[],
        description="""
        NAMD running environment and control parameters.
        """,
    )

    x_namd_inout_control_pme_maximum_grid_spacing = Quantity(
        type=str,
        shape=[],
        description="""
        NAMD running environment and control parameters.
        """,
    )

    x_namd_inout_control_fftw_data_file = Quantity(
        type=str,
        shape=[],
        description="""
        NAMD running environment and control parameters.
        """,
    )

    x_namd_inout_control_full_electrostatic_evaluation_frequency = Quantity(
        type=str,
        shape=[],
        description="""
        NAMD running environment and control parameters.
        """,
    )

    x_namd_inout_control_minimization = Quantity(
        type=str,
        shape=[],
        description="""
        NAMD running environment and control parameters.
        """,
    )

    x_namd_inout_control_velocity_quenching = Quantity(
        type=str,
        shape=[],
        description="""
        NAMD running environment and control parameters.
        """,
    )

    x_namd_inout_control_verlet_integrator = Quantity(
        type=str,
        shape=[],
        description="""
        NAMD running environment and control parameters.
        """,
    )

    x_namd_inout_control_random_number_seed = Quantity(
        type=str,
        shape=[],
        description="""
        NAMD running environment and control parameters.
        """,
    )

    x_namd_inout_control_use_hydrogen_bonds = Quantity(
        type=str,
        shape=[],
        description="""
        NAMD running environment and control parameters.
        """,
    )

    x_namd_inout_control_coordinate_pdb = Quantity(
        type=str,
        shape=[],
        description="""
        NAMD running environment and control parameters.
        """,
    )

    x_namd_inout_control_structure_file = Quantity(
        type=str,
        shape=[],
        description="""
        NAMD running environment and control parameters.
        """,
    )

    x_namd_inout_control_parameter_file = Quantity(
        type=str,
        shape=[],
        description="""
        NAMD running environment and control parameters.
        """,
    )

    x_namd_inout_control_number_of_parameters = Quantity(
        type=np.int32,
        shape=[],
        description="""
        NAMD running environment and control parameters.
        """,
    )

    x_namd_inout_control_parameters = Quantity(
        type=str,
        shape=['x_namd_inout_control_number_of_parameters'],
        description="""
        NAMD running environment and control parameters.
        """,
    )


class x_namd_section_atom_to_atom_type_ref(MSection):
    """
    Section to store atom label to atom type definition list
    """

    m_def = Section(validate=False)

    x_namd_atom_to_atom_type_ref = Quantity(
        type=np.int64,
        shape=['number_of_atoms_per_type'],
        description="""
        Reference to the atoms of each atom type.
        """,
    )


class x_namd_section_single_configuration_calculation(MSection):
    """
    section for gathering values for MD steps
    """

    m_def = Section(validate=False)


class System(runschema.system.System):
    m_def = Section(validate=False, extends_base_section=True)

    x_namd_atom_positions_image_index = Quantity(
        type=np.int32,
        shape=['number_of_atoms', 3],
        unit='dimensionless',
        description="""
        PBC image flag index.
        """,
    )

    x_namd_atom_positions_scaled = Quantity(
        type=np.float64,
        shape=['number_of_atoms', 3],
        unit='dimensionless',
        description="""
        Position of the atoms in a scaled format [0, 1].
        """,
    )

    x_namd_atom_positions_wrapped = Quantity(
        type=np.float64,
        shape=['number_of_atoms', 3],
        unit='meter',
        description="""
        Position of the atoms wrapped back to the periodic box.
        """,
    )

    x_namd_lattice_lengths = Quantity(
        type=np.float64,
        shape=[3],
        description="""
        Lattice dimensions in a vector. Vector includes [a, b, c] lengths.
        """,
    )

    x_namd_lattice_angles = Quantity(
        type=np.float64,
        shape=[3],
        description="""
        Angles of lattice vectors. Vector includes [alpha, beta, gamma] in degrees.
        """,
    )

    x_namd_dummy = Quantity(
        type=str,
        shape=[],
        description="""
        dummy
        """,
    )

    x_namd_mdin_finline = Quantity(
        type=str,
        shape=[],
        description="""
        finline in mdin
        """,
    )

    x_namd_traj_timestep_store = Quantity(
        type=str,
        shape=[],
        description="""
        tmp
        """,
    )

    x_namd_traj_number_of_atoms_store = Quantity(
        type=str,
        shape=[],
        description="""
        tmp
        """,
    )

    x_namd_traj_box_bound_store = Quantity(
        type=str,
        shape=[],
        description="""
        tmp
        """,
    )

    x_namd_traj_box_bounds_store = Quantity(
        type=str,
        shape=[],
        description="""
        tmp
        """,
    )

    x_namd_traj_variables_store = Quantity(
        type=str,
        shape=[],
        description="""
        tmp
        """,
    )

    x_namd_traj_atoms_store = Quantity(
        type=str,
        shape=[],
        description="""
        tmp
        """,
    )


class MolecularDynamics(simulationworkflowschema.MolecularDynamics):
    m_def = Section(validate=False, extends_base_section=True)

    x_namd_barostat_target_pressure = Quantity(
        type=np.float64,
        shape=[],
        unit='pascal',
        description="""
        MD barostat target pressure.
        """,
    )

    x_namd_barostat_tau = Quantity(
        type=np.float64,
        shape=[],
        unit='second',
        description="""
        MD barostat relaxation time.
        """,
    )

    x_namd_barostat_type = Quantity(
        type=str,
        shape=[],
        description="""
        MD barostat type, valid values are defined in the barostat_type wiki page.
        """,
    )

    x_namd_integrator_dt = Quantity(
        type=np.float64,
        shape=[],
        unit='second',
        description="""
        MD integration time step.
        """,
    )

    x_namd_integrator_type = Quantity(
        type=str,
        shape=[],
        description="""
        MD integrator type, valid values are defined in the integrator_type wiki page.
        """,
    )

    x_namd_periodicity_type = Quantity(
        type=str,
        shape=[],
        description="""
        Periodic boundary condition type in the sampling (non-PBC or PBC).
        """,
    )

    x_namd_langevin_gamma = Quantity(
        type=np.float64,
        shape=[],
        unit='second',
        description="""
        Langevin thermostat damping factor.
        """,
    )

    x_namd_number_of_steps_requested = Quantity(
        type=np.float64,
        shape=[],
        description="""
        Number of requested MD integration time steps.
        """,
    )

    x_namd_thermostat_level = Quantity(
        type=str,
        shape=[],
        description="""
        MD thermostat level (see wiki: single, multiple, regional).
        """,
    )

    x_namd_thermostat_target_temperature = Quantity(
        type=np.float64,
        shape=[],
        unit='kelvin',
        description="""
        MD thermostat target temperature.
        """,
    )

    x_namd_thermostat_tau = Quantity(
        type=np.float64,
        shape=[],
        unit='second',
        description="""
        MD thermostat relaxation time.
        """,
    )

    x_namd_thermostat_type = Quantity(
        type=str,
        shape=[],
        description="""
        MD thermostat type, valid values are defined in the thermostat_type wiki page.
        """,
    )


class AtomParameters(runschema.method.AtomParameters):
    m_def = Section(validate=False, extends_base_section=True)

    x_namd_atom_name = Quantity(
        type=str,
        shape=[],
        description="""
        Atom name of an atom in topology definition.
        """,
    )

    x_namd_atom_type = Quantity(
        type=str,
        shape=[],
        description="""
        Atom type of an atom in topology definition.
        """,
    )

    x_namd_atom_element = Quantity(
        type=str,
        shape=[],
        description="""
        Atom type of an atom in topology definition.
        """,
    )

    x_namd_atom_type_element = Quantity(
        type=str,
        shape=[],
        description="""
        Element symbol of an atom type.
        """,
    )

    x_namd_atom_type_radius = Quantity(
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

    x_namd_interaction_atom_to_atom_type_ref = Quantity(
        type=runschema.method.AtomParameters,
        shape=['number_of_atoms_per_interaction'],
        description="""
        Reference to the atom type of each interaction atoms.
        """,
    )

    x_namd_number_of_defined_pair_interactions = Quantity(
        type=np.int32,
        shape=[],
        description="""
        Number of defined pair interactions (L-J pairs).
        """,
    )

    x_namd_pair_interaction_atom_type_ref = Quantity(
        type=runschema.method.AtomParameters,
        shape=[
            'x_namd_number_of_defined_pair_interactions',
            'number_of_atoms_per_interaction',
        ],
        description="""
        Reference to the atom type for pair interactions.
        """,
    )

    x_namd_pair_interaction_parameters = Quantity(
        type=np.float64,
        shape=['x_namd_number_of_defined_pair_interactions', 2],
        description="""
        Pair interactions parameters.
        """,
    )

    x_namd_molecule_interaction_atom_to_atom_type_ref = Quantity(
        type=runschema.method.AtomParameters,
        shape=['number_of_atoms_per_interaction'],
        description="""
        Reference to the atom type of each molecule interaction atoms.
        """,
    )

    x_namd_number_of_defined_molecule_pair_interactions = Quantity(
        type=np.int32,
        shape=[],
        description="""
        Number of defined pair interactions within a molecule (L-J pairs).
        """,
    )

    x_namd_pair_molecule_interaction_parameters = Quantity(
        type=np.float64,
        shape=['number_of_defined_molecule_pair_interactions', 2],
        description="""
        Molecule pair interactions parameters.
        """,
    )

    x_namd_pair_molecule_interaction_to_atom_type_ref = Quantity(
        type=runschema.method.AtomParameters,
        shape=[
            'x_namd_number_of_defined_pair_interactions',
            'number_of_atoms_per_interaction',
        ],
        description="""
        Reference to the atom type for pair interactions within a molecule.
        """,
    )


class Run(runschema.run.Run):
    m_def = Section(validate=False, extends_base_section=True)

    x_namd_program_version_date = Quantity(
        type=str,
        shape=[],
        description="""
        Program version date.
        """,
    )

    x_namd_parallel_task_nr = Quantity(
        type=np.float64,
        shape=[],
        description="""
        Program task no.
        """,
    )

    x_namd_program_build_date = Quantity(
        type=str,
        shape=[],
        description="""
        Program Build date
        """,
    )

    x_namd_program_citation = Quantity(
        type=str,
        shape=[],
        description="""
        Program citations
        """,
    )

    x_namd_number_of_tasks = Quantity(
        type=np.float64,
        shape=[],
        description="""
        Number of tasks in parallel program (MPI).
        """,
    )

    x_namd_program_module_version = Quantity(
        type=str,
        shape=[],
        description="""
        NAMD program module version.
        """,
    )

    x_namd_program_license = Quantity(
        type=str,
        shape=[],
        description="""
        NAMD program license.
        """,
    )

    x_namd_xlo_xhi = Quantity(
        type=str,
        shape=[],
        description="""
        test
        """,
    )

    x_namd_data_file_store = Quantity(
        type=str,
        shape=[],
        description="""
        Filename of data file
        """,
    )

    x_namd_program_working_path = Quantity(
        type=str,
        shape=[],
        description="""
        tmp
        """,
    )

    x_namd_program_execution_host = Quantity(
        type=str,
        shape=[],
        description="""
        tmp
        """,
    )

    x_namd_program_execution_path = Quantity(
        type=str,
        shape=[],
        description="""
        tmp
        """,
    )

    x_namd_program_module = Quantity(
        type=str,
        shape=[],
        description="""
        tmp
        """,
    )

    x_namd_program_execution_date = Quantity(
        type=str,
        shape=[],
        description="""
        tmp
        """,
    )

    x_namd_program_execution_time = Quantity(
        type=str,
        shape=[],
        description="""
        tmp
        """,
    )

    x_namd_mdin_header = Quantity(
        type=str,
        shape=[],
        description="""
        tmp
        """,
    )

    x_namd_mdin_wt = Quantity(
        type=str,
        shape=[],
        description="""
        tmp
        """,
    )

    x_namd_section_input_output_files = SubSection(
        sub_section=SectionProxy('x_namd_section_input_output_files'), repeats=True
    )

    x_namd_section_control_parameters = SubSection(
        sub_section=SectionProxy('x_namd_section_control_parameters'), repeats=True
    )


class Calculation(runschema.calculation.Calculation):
    m_def = Section(validate=False, extends_base_section=True)

    x_namd_temperature_average = Quantity(
        type=np.float64,
        shape=[],
        unit='kelvin',
        description="""
        """,
    )

    x_namd_gpressure = Quantity(
        type=np.float64,
        shape=[],
        unit='pascal',
        description="""
        """,
    )

    x_namd_pressure_average = Quantity(
        type=np.float64,
        shape=[],
        unit='pascal',
        description="""
        """,
    )

    x_namd_gpressure_average = Quantity(
        type=np.float64,
        shape=[],
        unit='pascal',
        description="""
        """,
    )

    x_namd_volume = Quantity(
        type=np.float64,
        shape=[],
        unit='meter ** 3',
        description="""
        """,
    )


class Energy(runschema.calculation.Energy):
    m_def = Section(validate=False, extends_base_section=True)

    x_namd_total3 = SubSection(sub_section=runschema.calculation.EnergyEntry.m_def)


class Program(runschema.run.Program):
    m_def = Section(validate=False, extends_base_section=True)

    x_namd_build_osarch = Quantity(
        type=str,
        shape=[],
        description="""
        Program Build OS/ARCH
        """,
    )


class Method(runschema.method.Method):
    m_def = Section(validate=False, extends_base_section=True)

    x_namd_input_parameters = Quantity(
        type=JSON,
        shape=[],
        description="""
        Input parameters read from the configuration file.
        """,
    )

    x_namd_simulation_parameters = Quantity(
        type=JSON,
        shape=[],
        description="""
        Simulation parameters used by the program.
        """,
    )
