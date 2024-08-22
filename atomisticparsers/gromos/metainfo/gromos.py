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


class x_gromos_mdin_input_output_files(MCategory):
    """
    Parameters of mdin belonging to x_gromos_section_control_parameters.
    """

    m_def = Category()


class x_gromos_mdin_control_parameters(MCategory):
    """
    Parameters of mdin belonging to x_gromos_section_control_parameters.
    """

    m_def = Category()


class x_gromos_mdin_method(MCategory):
    """
    Parameters of mdin belonging to section method.
    """

    m_def = Category()


class x_gromos_mdout_single_configuration_calculation(MCategory):
    """
    Parameters of mdout belonging to section_single_configuration_calculation.
    """

    m_def = Category()


class x_gromos_mdout_method(MCategory):
    """
    Parameters of mdin belonging to section method.
    """

    m_def = Category()


class x_gromos_mdout_run(MCategory):
    """
    Parameters of mdin belonging to settings run.
    """

    m_def = Category()


class x_gromos_mdin_run(MCategory):
    """
    Parameters of mdin belonging to settings run.
    """

    m_def = Category()


class x_gromos_section_input_output_files(MSection):
    """
    Section to store input and output file names
    """

    m_def = Section(validate=False)


class x_gromos_section_control_parameters(MSection):
    """
    Section to store the input and output control parameters
    """

    m_def = Section(validate=False)

    x_gromos_inout_file_structure = Quantity(
        type=str,
        shape=[],
        description="""
        gromos input topology file.
        """,
    )

    x_gromos_inout_file_trajectory = Quantity(
        type=str,
        shape=[],
        description="""
        gromos output trajectory file.
        """,
    )

    x_gromos_inout_file_traj_coord = Quantity(
        type=str,
        shape=[],
        description="""
        gromos output trajectory file.
        """,
    )

    x_gromos_inout_file_traj_vel = Quantity(
        type=str,
        shape=[],
        description="""
        gromos output file for velocities in the trajectory.
        """,
    )

    x_gromos_inout_file_traj_force = Quantity(
        type=str,
        shape=[],
        description="""
        gromos output file for forces in the trajectory.
        """,
    )

    x_gromos_inout_file_output_coord = Quantity(
        type=str,
        shape=[],
        description="""
        gromos output coordinates file.
        """,
    )

    x_gromos_inout_file_output_vel = Quantity(
        type=str,
        shape=[],
        description="""
        gromos output velocities file.
        """,
    )

    x_gromos_inout_file_output_force = Quantity(
        type=str,
        shape=[],
        description="""
        gromos output forces file.
        """,
    )

    x_gromos_inout_file_input_coord = Quantity(
        type=str,
        shape=[],
        description="""
        gromos input coordinates file.
        """,
    )

    x_gromos_inout_file_input_vel = Quantity(
        type=str,
        shape=[],
        description="""
        gromos input velocities file.
        """,
    )

    x_gromos_inout_file_restart_coord = Quantity(
        type=str,
        shape=[],
        description="""
        gromos restart coordinates file.
        """,
    )

    x_gromos_inout_file_restart_vel = Quantity(
        type=str,
        shape=[],
        description="""
        gromos restart velocities file.
        """,
    )

    x_gromos_inout_file_output_log = Quantity(
        type=str,
        shape=[],
        description="""
        gromos MD output log file.
        """,
    )

    x_gromos_inout_control_number_of_steps = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_steps_per_cycle = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_initial_temperature = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_dielectric = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_minimization = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_verlet_integrator = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_topology_parameters = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_topology_type = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_resname = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_number_of_atom_types = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_number_of_atoms_solute = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_lennard_jones_exceptions = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_number_of_h_bonds_at_constraint = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_number_of_bonds_at_constraint = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_bondangles_containing_hydrogens = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_bondangles_not_containing_hydrogens = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_improper_dihedrals_not_containing_hydrogens = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_improper_dihedrals_containing_hydrogens = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_dihedrals_not_containing_hydrogens = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_dihedrals_containing_hydrogens = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_crossdihedrals_not_containing_hydrogens = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_crossdihedrals_containing_hydrogens = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_number_of_solvent_atoms = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_number_of_solvent_constraints = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_number_of_solvents_added = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_number_of_molecules_in_solute = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_number_of_temperature_groups_for_solute = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_number_of_pressure_groups_for_solute = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_bond_angle_interaction_in_force_field = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_improper_dihedral_interaction_in_force_field = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_dihedral_interaction_in_force_field = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_nonbonded_definitions_in_force_field = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_pairlist_algorithm = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_periodic_boundary_conditions = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_virial = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_cutoff_type = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_shortrange_cutoff = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_longrange_cutoff = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_pairlist_update_step_frequency = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_reactionfield_cutoff = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_force_field_epsilon = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_reactionfield_epsilon = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_force_field_kappa = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_force_field_perturbation = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_title = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_emin_ntem = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_emin_ncyc = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_emin_dele = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_emin_dx0 = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_emin_dxm = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_emin_nmin = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_emin_flim = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_sys_npm = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_sys_nsm = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_init_ntivel = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_init_ntishk = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_init_ntinht = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_init_ntinhb = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_init_ntishi = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_init_ntirtc = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_init_nticom = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_init_ntisti = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_init_ig = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_init_tempi = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_step_nstlim = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_step_t = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_step_dt = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_bcnd_ntb = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_bcnd_ndfmin = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_bath_alg = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_bath_num = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_bath_nbaths = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_bath_temp = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_bath_tau = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_bath_dofset = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_bath_last = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_bath_combath = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_bath_irbath = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_pres_couple = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_pres_scale = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_pres_comp = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_pres_taup = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_pres_virial = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_pres_aniso = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_pres_init0 = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_covf_ntbbh = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_covf_ntbah = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_covf_ntbdn = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_solm_nspm = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_solm_nsp = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_comt_nscm = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_prnt_ntpr = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_prnt_ntpp = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_writ_ntwx = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_writ_ntwse = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_writ_ntwv = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_writ_ntwf = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_writ_ntwe = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_writ_ntwg = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_writ_ntwb = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_cnst_ntc = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_cnst_ntcp = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_cnst_ntcp0 = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_cnst_ntcs = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_cnst_ntcs0 = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_forc_bonds = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_forc_angs = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_forc_imps = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_forc_dihs = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_forc_elec = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_forc_vdw = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_forc_negr = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_forc_nre = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_pair_alg = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_pair_nsnb = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_pair_rcutp = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_pair_rcutl = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_pair_size = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_pair_type = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_nonb_nlrele = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_nonb_appak = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_nonb_rcrf = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_nonb_epsrf = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_nonb_nslfexcl = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_nonb_nshape = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_nonb_ashape = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_nonb_na2clc = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_nonb_tola2 = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_nonb_epsls = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_nonb_nkx = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_nonb_nky = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_nonb_nkz = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_nonb_kcut = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_nonb_ngx = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_nonb_ngy = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_nonb_ngz = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_nonb_nasord = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_nonb_nfdord = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_nonb_nalias = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_nonb_nspord = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_nonb_nqeval = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_nonb_faccur = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_nonb_nrdgrd = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_nonb_nwrgrd = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_nonb_nlrlj = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_nonb_slvdns = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_structure_file = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_parameter_file = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_input_file = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_topology_file = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_configuration_file = Quantity(
        type=str,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_number_of_parameters = Quantity(
        type=np.int32,
        shape=[],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_inout_control_parameters = Quantity(
        type=str,
        shape=['x_gromos_inout_control_number_of_parameters'],
        description="""
        gromos running environment and control parameters.
        """,
    )

    x_gromos_section_input_output_files = SubSection(
        sub_section=SectionProxy('x_gromos_section_input_output_files'), repeats=True
    )


class x_gromos_section_atom_to_atom_type_ref(MSection):
    """
    Section to store atom label to atom type definition list
    """

    m_def = Section(validate=False)

    x_gromos_atom_to_atom_type_ref = Quantity(
        type=np.int64,
        shape=['number_of_atoms_per_type'],
        description="""
        Reference to the atoms of each atom type.
        """,
    )


class x_gromos_section_single_configuration_calculation(MSection):
    """
    section for gathering values for MD steps
    """

    m_def = Section(validate=False)


class System(runschema.system.System):
    m_def = Section(validate=False, extends_base_section=True)

    x_gromos_atom_positions_image_index = Quantity(
        type=np.int32,
        shape=['number_of_atoms', 3],
        unit='dimensionless',
        description="""
        PBC image flag index.
        """,
    )

    x_gromos_atom_positions_scaled = Quantity(
        type=np.float64,
        shape=['number_of_atoms', 3],
        unit='dimensionless',
        description="""
        Position of the atoms in a scaled format [0, 1].
        """,
    )

    x_gromos_atom_positions_wrapped = Quantity(
        type=np.float64,
        shape=['number_of_atoms', 3],
        unit='meter',
        description="""
        Position of the atoms wrapped back to the periodic box.
        """,
    )

    x_gromos_lattice_lengths = Quantity(
        type=np.float64,
        shape=[3],
        description="""
        Lattice dimensions in a vector. Vector includes [a, b, c] lengths.
        """,
    )

    x_gromos_lattice_angles = Quantity(
        type=np.float64,
        shape=[3],
        description="""
        Angles of lattice vectors. Vector includes [alpha, beta, gamma] in degrees.
        """,
    )

    x_gromos_dummy = Quantity(
        type=str,
        shape=[],
        description="""
        dummy
        """,
    )

    x_gromos_mdin_finline = Quantity(
        type=str,
        shape=[],
        description="""
        finline in mdin
        """,
    )

    x_gromos_traj_timestep_store = Quantity(
        type=str,
        shape=[],
        description="""
        tmp
        """,
    )

    x_gromos_traj_number_of_atoms_store = Quantity(
        type=str,
        shape=[],
        description="""
        tmp
        """,
    )

    x_gromos_traj_box_bound_store = Quantity(
        type=str,
        shape=[],
        description="""
        tmp
        """,
    )

    x_gromos_traj_box_bounds_store = Quantity(
        type=str,
        shape=[],
        description="""
        tmp
        """,
    )

    x_gromos_traj_variables_store = Quantity(
        type=str,
        shape=[],
        description="""
        tmp
        """,
    )

    x_gromos_traj_atoms_store = Quantity(
        type=str,
        shape=[],
        description="""
        tmp
        """,
    )


class MolecularDynamics(simulationworkflowschema.MolecularDynamics):
    m_def = Section(validate=False, extends_base_section=True)

    x_gromos_barostat_target_pressure = Quantity(
        type=np.float64,
        shape=[],
        unit='pascal',
        description="""
        MD barostat target pressure.
        """,
    )

    x_gromos_barostat_tau = Quantity(
        type=np.float64,
        shape=[],
        unit='second',
        description="""
        MD barostat relaxation time.
        """,
    )

    x_gromos_barostat_type = Quantity(
        type=str,
        shape=[],
        description="""
        MD barostat type, valid values are defined in the barostat_type wiki page.
        """,
    )

    x_gromos_integrator_dt = Quantity(
        type=np.float64,
        shape=[],
        unit='second',
        description="""
        MD integration time step.
        """,
    )

    x_gromos_integrator_type = Quantity(
        type=str,
        shape=[],
        description="""
        MD integrator type, valid values are defined in the integrator_type wiki page.
        """,
    )

    x_gromos_periodicity_type = Quantity(
        type=str,
        shape=[],
        description="""
        Periodic boundary condition type in the sampling (non-PBC or PBC).
        """,
    )

    x_gromos_langevin_gamma = Quantity(
        type=np.float64,
        shape=[],
        unit='second',
        description="""
        Langevin thermostat damping factor.
        """,
    )

    x_gromos_number_of_steps_requested = Quantity(
        type=np.float64,
        shape=[],
        description="""
        Number of requested MD integration time steps.
        """,
    )

    x_gromos_thermostat_level = Quantity(
        type=str,
        shape=[],
        description="""
        MD thermostat level (see wiki: single, multiple, regional).
        """,
    )

    x_gromos_thermostat_target_temperature = Quantity(
        type=np.float64,
        shape=[],
        unit='kelvin',
        description="""
        MD thermostat target temperature.
        """,
    )

    x_gromos_thermostat_tau = Quantity(
        type=np.float64,
        shape=[],
        unit='second',
        description="""
        MD thermostat relaxation time.
        """,
    )

    x_gromos_thermostat_type = Quantity(
        type=str,
        shape=[],
        description="""
        MD thermostat type, valid values are defined in the thermostat_type wiki page.
        """,
    )


class AtomParameters(runschema.method.AtomParameters):
    m_def = Section(validate=False, extends_base_section=True)

    x_gromos_atom_name = Quantity(
        type=str,
        shape=[],
        description="""
        Atom name of an atom in topology definition.
        """,
    )

    x_gromos_atom_type = Quantity(
        type=str,
        shape=[],
        description="""
        Atom type of an atom in topology definition.
        """,
    )

    x_gromos_atom_element = Quantity(
        type=str,
        shape=[],
        description="""
        Atom type of an atom in topology definition.
        """,
    )

    x_gromos_atom_type_element = Quantity(
        type=str,
        shape=[],
        description="""
        Element symbol of an atom type.
        """,
    )

    x_gromos_atom_type_radius = Quantity(
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

    x_gromos_interaction_atom_to_atom_type_ref = Quantity(
        type=runschema.method.AtomParameters,
        shape=['number_of_atoms_per_interaction'],
        description="""
        Reference to the atom type of each interaction atoms.
        """,
    )

    x_gromos_number_of_defined_pair_interactions = Quantity(
        type=np.int32,
        shape=[],
        description="""
        Number of defined pair interactions (L-J pairs).
        """,
    )

    x_gromos_pair_interaction_atom_type_ref = Quantity(
        type=runschema.method.AtomParameters,
        shape=[
            'x_gromos_number_of_defined_pair_interactions',
            'number_of_atoms_per_interaction',
        ],
        description="""
        Reference to the atom type for pair interactions.
        """,
    )

    x_gromos_pair_interaction_parameters = Quantity(
        type=np.float64,
        shape=['x_gromos_number_of_defined_pair_interactions', 2],
        description="""
        Pair interactions parameters.
        """,
    )

    x_gromos_molecule_interaction_atom_to_atom_type_ref = Quantity(
        type=runschema.method.AtomParameters,
        shape=['number_of_atoms_per_interaction'],
        description="""
        Reference to the atom type of each molecule interaction atoms.
        """,
    )

    x_gromos_number_of_defined_molecule_pair_interactions = Quantity(
        type=np.int32,
        shape=[],
        description="""
        Number of defined pair interactions within a molecule (L-J pairs).
        """,
    )

    x_gromos_pair_molecule_interaction_parameters = Quantity(
        type=np.float64,
        shape=['number_of_defined_molecule_pair_interactions', 2],
        description="""
        Molecule pair interactions parameters.
        """,
    )

    x_gromos_pair_molecule_interaction_to_atom_type_ref = Quantity(
        type=runschema.method.AtomParameters,
        shape=[
            'x_gromos_number_of_defined_pair_interactions',
            'number_of_atoms_per_interaction',
        ],
        description="""
        Reference to the atom type for pair interactions within a molecule.
        """,
    )


class Run(runschema.run.Run):
    m_def = Section(validate=False, extends_base_section=True)

    x_gromos_program_version_date = Quantity(
        type=str,
        shape=[],
        description="""
        Program version date.
        """,
    )

    x_gromos_parallel_task_nr = Quantity(
        type=np.float64,
        shape=[],
        description="""
        Program task no.
        """,
    )

    x_gromos_build_osarch = Quantity(
        type=str,
        shape=[],
        description="""
        Program Build OS/ARCH
        """,
    )

    x_gromos_output_created_by_user = Quantity(
        type=str,
        shape=[],
        description="""
        Output file creator
        """,
    )

    x_gromos_most_severe_warning_level = Quantity(
        type=str,
        shape=[],
        description="""
        Highest gromos warning level in the run.
        """,
    )

    x_gromos_program_build_date = Quantity(
        type=str,
        shape=[],
        description="""
        Program Build date
        """,
    )

    x_gromos_program_citation = Quantity(
        type=str,
        shape=[],
        description="""
        Program citations
        """,
    )

    x_gromos_program_copyright = Quantity(
        type=str,
        shape=[],
        description="""
        Program copyright
        """,
    )

    x_gromos_number_of_tasks = Quantity(
        type=np.float64,
        shape=[],
        description="""
        Number of tasks in parallel program (MPI).
        """,
    )

    x_gromos_program_module_version = Quantity(
        type=str,
        shape=[],
        description="""
        gromos program module version.
        """,
    )

    x_gromos_program_license = Quantity(
        type=str,
        shape=[],
        description="""
        gromos program license.
        """,
    )

    x_gromos_xlo_xhi = Quantity(
        type=str,
        shape=[],
        description="""
        test
        """,
    )

    x_gromos_data_file_store = Quantity(
        type=str,
        shape=[],
        description="""
        Filename of data file
        """,
    )

    x_gromos_program_working_path = Quantity(
        type=str,
        shape=[],
        description="""
        tmp
        """,
    )

    x_gromos_program_execution_host = Quantity(
        type=str,
        shape=[],
        description="""
        tmp
        """,
    )

    x_gromos_program_execution_path = Quantity(
        type=str,
        shape=[],
        description="""
        tmp
        """,
    )

    x_gromos_program_module = Quantity(
        type=str,
        shape=[],
        description="""
        tmp
        """,
    )

    x_gromos_program_execution_date = Quantity(
        type=str,
        shape=[],
        description="""
        tmp
        """,
    )

    x_gromos_program_execution_time = Quantity(
        type=str,
        shape=[],
        description="""
        tmp
        """,
    )

    x_gromos_mdin_header = Quantity(
        type=str,
        shape=[],
        description="""
        tmp
        """,
    )

    x_gromos_mdin_wt = Quantity(
        type=str,
        shape=[],
        description="""
        tmp
        """,
    )

    x_gromos_section_control_parameters = SubSection(
        sub_section=SectionProxy('x_gromos_section_control_parameters'), repeats=True
    )


class Calculation(runschema.calculation.Calculation):
    m_def = Section(validate=False, extends_base_section=True)

    x_gromos_section_single_configuration_calculation = SubSection(
        sub_section=SectionProxy('x_gromos_section_single_configuration_calculation'),
        repeats=True,
    )
