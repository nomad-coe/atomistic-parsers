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
)
import runschema.run  # pylint: disable=unused-import
import runschema.calculation  # pylint: disable=unused-import
import runschema.method  # pylint: disable=unused-import
import runschema.system  # pylint: disable=unused-import
import simulationworkflowschema


m_package = Package()


class x_gulp_section_main_keyword(MSection):
    """
    Section for GULP calculation mode input variable
    """

    m_def = Section(validate=False)

    x_gulp_main_keyword = Quantity(
        type=str,
        shape=[],
        description="""
        GULP calculation mode input variable
        """,
    )


class x_gulp_section_forcefield(MSection):
    """
    Section for GULP force field specification
    """

    m_def = Section(validate=False)

    x_gulp_forcefield_species_1 = Quantity(
        type=str,
        shape=[],
        description="""
        GULP force field species 1
        """,
    )

    x_gulp_forcefield_species_2 = Quantity(
        type=str,
        shape=[],
        description="""
        GULP force field species 2
        """,
    )

    x_gulp_forcefield_species_3 = Quantity(
        type=str,
        shape=[],
        description="""
        GULP force field species 3
        """,
    )

    x_gulp_forcefield_species_4 = Quantity(
        type=str,
        shape=[],
        description="""
        GULP force field species 4
        """,
    )

    x_gulp_forcefield_speciestype_1 = Quantity(
        type=str,
        shape=[],
        description="""
        GULP force field speciestype 1
        """,
    )

    x_gulp_forcefield_speciestype_2 = Quantity(
        type=str,
        shape=[],
        description="""
        GULP force field speciestype 2
        """,
    )

    x_gulp_forcefield_speciestype_3 = Quantity(
        type=str,
        shape=[],
        description="""
        GULP force field speciestype 3
        """,
    )

    x_gulp_forcefield_speciestype_4 = Quantity(
        type=str,
        shape=[],
        description="""
        GULP force field speciestype 4
        """,
    )

    x_gulp_forcefield_potential_name = Quantity(
        type=str,
        shape=[],
        description="""
        GULP force field potential name
        """,
    )

    x_gulp_forcefield_parameter_a = Quantity(
        type=np.float64,
        shape=[],
        description="""
        GULP force field parameter A
        """,
    )

    x_gulp_forcefield_parameter_b = Quantity(
        type=np.float64,
        shape=[],
        description="""
        GULP force field parameter B
        """,
    )

    x_gulp_forcefield_parameter_c = Quantity(
        type=np.float64,
        shape=[],
        description="""
        GULP force field parameter C
        """,
    )

    x_gulp_forcefield_parameter_d = Quantity(
        type=np.float64,
        shape=[],
        description="""
        GULP force field parameter D
        """,
    )

    x_gulp_forcefield_cutoff_min = Quantity(
        type=str,
        shape=[],
        description="""
        GULP force field cutoff min (can also be a string like 3Bond for some reason)
        """,
    )

    x_gulp_forcefield_cutoff_max = Quantity(
        type=np.float64,
        shape=[],
        description="""
        GULP force field cutoff max
        """,
    )

    x_gulp_forcefield_threebody_1 = Quantity(
        type=str,
        shape=[],
        description="""
        GULP 3-body force field parameter 1
        """,
    )

    x_gulp_forcefield_threebody_2 = Quantity(
        type=np.float64,
        shape=[],
        description="""
        GULP 3-body force field parameter 2
        """,
    )

    x_gulp_forcefield_threebody_3 = Quantity(
        type=np.float64,
        shape=[],
        description="""
        GULP 3-body force field parameter 3
        """,
    )

    x_gulp_forcefield_threebody_theta = Quantity(
        type=np.float64,
        shape=[],
        description="""
        GULP 3-body force field parameter theta
        """,
    )

    x_gulp_forcefield_fourbody_force_constant = Quantity(
        type=np.float64,
        shape=[],
        description="""
        GULP 4-body force field parameter force constant
        """,
    )

    x_gulp_forcefield_fourbody_sign = Quantity(
        type=str,
        shape=[],
        description="""
        GULP 4-body force field parameter sign
        """,
    )

    x_gulp_forcefield_fourbody_phase = Quantity(
        type=np.float64,
        shape=[],
        description="""
        GULP 4-body force field parameter phase
        """,
    )

    x_gulp_forcefield_fourbody_phi0 = Quantity(
        type=np.float64,
        shape=[],
        description="""
        GULP 4-body force field parameter phi0
        """,
    )


class System(runschema.system.System):
    m_def = Section(validate=False, extends_base_section=True)

    x_gulp_patterson_group = Quantity(
        type=str,
        shape=[],
        description="""
        Patterson group
        """,
    )

    x_gulp_space_group = Quantity(
        type=str,
        shape=[],
        description="""
        Space group
        """,
    )

    x_gulp_formula = Quantity(
        type=str,
        shape=[],
        description="""
        GULP chemical formula
        """,
    )

    x_gulp_cell_alpha = Quantity(
        type=np.float64,
        shape=[],
        description="""
        grrr
        """,
    )

    x_gulp_cell_beta = Quantity(
        type=np.float64,
        shape=[],
        description="""
        grrr
        """,
    )

    x_gulp_cell_gamma = Quantity(
        type=np.float64,
        shape=[],
        description="""
        grrr
        """,
    )

    x_gulp_cell_a = Quantity(
        type=np.float64,
        shape=[],
        description="""
        grrr
        """,
    )

    x_gulp_cell_b = Quantity(
        type=np.float64,
        shape=[],
        description="""
        grrr
        """,
    )

    x_gulp_cell_c = Quantity(
        type=np.float64,
        shape=[],
        description="""
        grrr
        """,
    )

    x_gulp_prim_cell_alpha = Quantity(
        type=np.float64,
        shape=[],
        description="""
        grrr
        """,
    )

    x_gulp_prim_cell_beta = Quantity(
        type=np.float64,
        shape=[],
        description="""
        grrr
        """,
    )

    x_gulp_prim_cell_gamma = Quantity(
        type=np.float64,
        shape=[],
        description="""
        grrr
        """,
    )

    x_gulp_prim_cell_a = Quantity(
        type=np.float64,
        shape=[],
        description="""
        grrr
        """,
    )

    x_gulp_prim_cell_b = Quantity(
        type=np.float64,
        shape=[],
        description="""
        grrr
        """,
    )

    x_gulp_prim_cell_c = Quantity(
        type=np.float64,
        shape=[],
        description="""
        grrr
        """,
    )

    x_gulp_pbc = Quantity(
        type=np.int32,
        shape=[],
        description="""
        grrr
        """,
    )


class Run(runschema.run.Run):
    m_def = Section(validate=False, extends_base_section=True)

    x_gulp_title = Quantity(
        type=str,
        shape=[],
        description="""
        Title of GULP calculation
        """,
    )

    x_gulp_n_cpu = Quantity(
        type=np.int32,
        shape=[],
        description="""
        """,
    )

    x_gulp_host_name = Quantity(
        type=str,
        shape=[],
        description="""
        """,
    )

    x_gulp_total_n_configurations_input = Quantity(
        type=np.int32,
        shape=[],
        description="""
        """,
    )

    x_gulp_section_main_keyword = SubSection(
        sub_section=SectionProxy('x_gulp_section_main_keyword'), repeats=True
    )


class Method(runschema.method.Method):
    m_def = Section(validate=False, extends_base_section=True)

    x_gulp_number_of_species = Quantity(
        type=int,
        shape=[],
        description="""
        Number of species in GULP
        """,
    )

    x_gulp_species_charge = Quantity(
        type=np.float64,
        shape=['x_gulp_number_of_species'],
        description="""
        Number of species in GULP
        """,
    )

    x_gulp_section_forcefield = SubSection(
        sub_section=SectionProxy('x_gulp_section_forcefield'), repeats=True
    )


class AtomParameters(runschema.method.AtomParameters):
    m_def = Section(validate=False, extends_base_section=True)

    x_gulp_type = Quantity(
        type=str,
        shape=[],
        description="""
        """,
    )

    x_gulp_covalent_radius = Quantity(
        type=np.float64,
        shape=[],
        unit='m',
        description="""
        """,
    )

    x_gulp_ionic_radius = Quantity(
        type=np.float64,
        shape=[],
        unit='m',
        description="""
        """,
    )

    x_gulp_vdw_radius = Quantity(
        type=np.float64,
        shape=[],
        unit='m',
        description="""
        """,
    )


class Energy(runschema.calculation.Energy):
    m_def = Section(validate=False, extends_base_section=True)

    x_gulp_attachment_energy = Quantity(
        type=np.float64,
        shape=[],
        unit='joule',
        description="""
        GULP energy term for attachment_energy
        """,
    )

    x_gulp_attachment_unit = Quantity(
        type=np.float64,
        shape=[],
        unit='joule',
        description="""
        GULP energy term for attachment_energy_unit
        """,
    )

    x_gulp_bond_order_potentials = Quantity(
        type=np.float64,
        shape=[],
        unit='joule',
        description="""
        GULP energy term for bond_order_potentials
        """,
    )

    x_gulp_brenner_potentials = Quantity(
        type=np.float64,
        shape=[],
        unit='joule',
        description="""
        GULP energy term for brenner_potentials
        """,
    )

    x_gulp_bulk = Quantity(
        type=np.float64,
        shape=[],
        unit='joule',
        description="""
        GULP energy term for bulk_energy
        """,
    )

    x_gulp_dispersion_real_recip = Quantity(
        type=np.float64,
        shape=[],
        unit='joule',
        description="""
        GULP energy term for dispersion_real_recip
        """,
    )

    x_gulp_electric_field_times_distance = Quantity(
        type=np.float64,
        shape=[],
        unit='joule',
        description="""
        GULP energy term for electric_field_times_distance
        """,
    )

    x_gulp_shift = Quantity(
        type=np.float64,
        shape=[],
        unit='joule',
        description="""
        GULP energy term for energy_shift
        """,
    )

    x_gulp_four_body_potentials = Quantity(
        type=np.float64,
        shape=[],
        unit='joule',
        description="""
        GULP energy term for four_body_potentials
        """,
    )

    x_gulp_improper_torsions = Quantity(
        type=np.float64,
        shape=[],
        unit='joule',
        description="""
        GULP energy term for improper_torsions
        """,
    )

    x_gulp_interatomic_potentials = Quantity(
        type=np.float64,
        shape=[],
        unit='joule',
        description="""
        GULP energy term for interatomic_potentials
        """,
    )

    x_gulp_many_body_potentials = Quantity(
        type=np.float64,
        shape=[],
        unit='joule',
        description="""
        GULP energy term for many_body_potentials
        """,
    )

    x_gulp_monopole_monopole_real = Quantity(
        type=np.float64,
        shape=[],
        unit='joule',
        description="""
        GULP energy term for monopole_monopole_real
        """,
    )

    x_gulp_monopole_monopole_recip = Quantity(
        type=np.float64,
        shape=[],
        unit='joule',
        description="""
        GULP energy term for monopole_monopole_recip
        """,
    )

    x_gulp_monopole_monopole_total = Quantity(
        type=np.float64,
        shape=[],
        unit='joule',
        description="""
        GULP energy term for monopole_monopole_total
        """,
    )

    x_gulp_neutralising = Quantity(
        type=np.float64,
        shape=[],
        unit='joule',
        description="""
        GULP energy term for neutralising_energy
        """,
    )

    x_gulp_non_primitive_unit_cell = Quantity(
        type=np.float64,
        shape=[],
        unit='joule',
        description="""
        GULP energy term for non_primitive_unit_cell
        """,
    )

    x_gulp_out_of_plane_potentials = Quantity(
        type=np.float64,
        shape=[],
        unit='joule',
        description="""
        GULP energy term for out_of_plane_potentials
        """,
    )

    x_gulp_primitive_unit_cell = Quantity(
        type=np.float64,
        shape=[],
        unit='joule',
        description="""
        GULP energy term for primitive_unit_cell
        """,
    )

    x_gulp_reaxff_force_field = Quantity(
        type=np.float64,
        shape=[],
        unit='joule',
        description="""
        GULP energy term for reaxff_force_field
        """,
    )

    x_gulp_region_1_2_interaction = Quantity(
        type=np.float64,
        shape=[],
        unit='joule',
        description="""
        GULP energy term for region_1_2_interaction
        """,
    )

    x_gulp_region_2_2_interaction = Quantity(
        type=np.float64,
        shape=[],
        unit='joule',
        description="""
        GULP energy term for region_2_2_interaction
        """,
    )

    x_gulp_self_eem_qeq_sm = Quantity(
        type=np.float64,
        shape=[],
        unit='joule',
        description="""
        GULP energy term for self_energy_eem_qeq_sm
        """,
    )

    x_gulp_sm_coulomb_correction = Quantity(
        type=np.float64,
        shape=[],
        unit='joule',
        description="""
        GULP energy term for sm_coulomb_correction
        """,
    )

    x_gulp_solvation = Quantity(
        type=np.float64,
        shape=[],
        unit='joule',
        description="""
        GULP energy term for solvation_energy
        """,
    )

    x_gulp_three_body_potentials = Quantity(
        type=np.float64,
        shape=[],
        unit='joule',
        description="""
        GULP energy term for three_body_potentials
        """,
    )

    x_gulp_total_averaged = SubSection(
        sub_section=runschema.calculation.EnergyEntry.m_def
    )


class x_gulp_bulk_optimisation_cycle(MSection):
    m_def = Section(validate=False)

    x_gulp_energy = Quantity(
        type=np.float64,
        shape=[],
        unit='joule',
        description="""
        """,
    )

    x_gulp_gnorm = Quantity(
        type=np.float64,
        shape=[],
        description="""
        """,
    )

    x_gulp_cpu_time = Quantity(
        type=np.float64,
        shape=[],
        unit='s',
        description="""
        """,
    )


class x_gulp_bulk_optimisation(MSection):
    m_def = Section(validate=False)

    x_gulp_n_variables = Quantity(
        type=np.int32,
        shape=[],
        description="""
        """,
    )

    x_gulp_max_n_calculations = Quantity(
        type=np.int32,
        shape=[],
        description="""
        """,
    )

    x_gulp_max_hessian_update_interval = Quantity(
        type=np.int32,
        shape=[],
        description="""
        """,
    )

    x_gulp_max_step_size = Quantity(
        type=np.float64,
        shape=[],
        description="""
        """,
    )

    x_gulp_max_parameter_tolerance = Quantity(
        type=np.float64,
        shape=[],
        description="""
        """,
    )

    x_gulp_max_function_tolerance = Quantity(
        type=np.float64,
        shape=[],
        description="""
        """,
    )

    x_gulp_max_gradient_tolerance = Quantity(
        type=np.float64,
        shape=[],
        description="""
        """,
    )

    x_gulp_max_gradient_component = Quantity(
        type=np.float64,
        shape=[],
        description="""
        """,
    )

    x_gulp_optimiser = Quantity(
        type=str,
        shape=[],
        description="""
        """,
    )

    x_gulp_hessian_updater = Quantity(
        type=np.float64,
        shape=[],
        description="""
        """,
    )

    x_gulp_bulk_optimisation_cycle = SubSection(
        sub_section=x_gulp_bulk_optimisation_cycle, repeats=True
    )


class Calculation(runschema.calculation.Calculation):
    m_def = Section(validate=False, extends_base_section=True)

    x_gulp_md_time = Quantity(
        type=np.float64,
        shape=[],
        description="""
        GULP molecular dynamics time
        """,
    )

    x_gulp_md_kinetic_energy = Quantity(
        type=np.float64,
        shape=[],
        description="""
        GULP molecular dynamics kinetic energy
        """,
    )

    x_gulp_md_potential_energy = Quantity(
        type=np.float64,
        shape=[],
        description="""
        GULP molecular dynamics potential energy
        """,
    )

    x_gulp_md_total_energy = Quantity(
        type=np.float64,
        shape=[],
        description="""
        GULP molecular dynamics total energy
        """,
    )

    x_gulp_md_temperature = Quantity(
        type=np.float64,
        shape=[],
        description="""
        GULP molecular dynamics temperature
        """,
    )

    x_gulp_md_pressure = Quantity(
        type=np.float64,
        shape=[],
        description="""
        GULP molecular dynamics pressure
        """,
    )

    x_gulp_temperature_averaged = Quantity(
        type=np.float64,
        shape=[],
        unit='kelvin',
        description="""
        """,
    )

    x_gulp_pressure_averaged = Quantity(
        type=np.float64,
        shape=[],
        unit='pascal',
        description="""
        """,
    )

    x_gulp_piezoelectric_strain_matrix = Quantity(
        type=np.float64,
        shape=[3, 6],
        unit='C / m**2',
        description="""
        """,
    )

    x_gulp_piezoelectric_stress_matrix = Quantity(
        type=np.float64,
        shape=[3, 6],
        unit='C / N',
        description="""
        """,
    )

    x_gulp_static_dielectric_constant_tensor = Quantity(
        type=np.float64,
        shape=[3, 3],
        description="""
        """,
    )

    x_gulp_high_frequency_dielectric_constant_tensor = Quantity(
        type=np.float64,
        shape=[3, 3],
        description="""
        """,
    )

    x_gulp_static_refractive_indices = Quantity(
        type=np.float64,
        shape=[3],
        description="""
        """,
    )

    x_gulp_high_frequency_refractive_indices = Quantity(
        type=np.float64,
        shape=[3],
        description="""
        """,
    )

    x_gulp_bulk_optimisation = SubSection(sub_section=x_gulp_bulk_optimisation.m_def)


class ElasticResults(simulationworkflowschema.ElasticResults):
    m_def = Section(validate=False, extends_base_section=True)

    x_gulp_velocity_s_wave_reuss = Quantity(
        type=np.float64,
        shape=[],
        unit='m / s',
        description="""
        """,
    )

    x_gulp_velocity_s_wave_voigt = Quantity(
        type=np.float64,
        shape=[],
        unit='m / s',
        description="""
        """,
    )

    x_gulp_velocity_s_wave_hill = Quantity(
        type=np.float64,
        shape=[],
        unit='m / s',
        description="""
        """,
    )

    x_gulp_velocity_p_wave_reuss = Quantity(
        type=np.float64,
        shape=[],
        unit='m / s',
        description="""
        """,
    )

    x_gulp_velocity_p_wave_voigt = Quantity(
        type=np.float64,
        shape=[],
        unit='m / s',
        description="""
        """,
    )

    x_gulp_velocity_p_wave_hill = Quantity(
        type=np.float64,
        shape=[],
        unit='m / s',
        description="""
        """,
    )

    x_gulp_compressibility = Quantity(
        type=np.float64,
        shape=[],
        unit='1 / pascal',
        description="""
        """,
    )

    # TODO determine if these values can be transformed to the top level definitions
    x_gulp_youngs_modulus_x = Quantity(
        type=np.float64,
        shape=[],
        unit='pascal',
        description="""
        """,
    )

    x_gulp_youngs_modulus_y = Quantity(
        type=np.float64,
        shape=[],
        unit='pascal',
        description="""
        """,
    )

    x_gulp_youngs_modulus_z = Quantity(
        type=np.float64,
        shape=[],
        unit='pascal',
        description="""
        """,
    )

    x_gulp_poissons_ratio = Quantity(
        type=np.float64,
        shape=[3, 3],
        description="""
        """,
    )


class MolecularDynamics(simulationworkflowschema.MolecularDynamics):
    m_def = Section(validate=False, extends_base_section=True)

    x_gulp_friction_temperature_bath = Quantity(
        type=np.float64,
        shape=[],
        description="""
        """,
    )

    x_gulp_n_mobile_ions = Quantity(
        type=np.int32,
        shape=[],
        description="""
        """,
    )

    x_gulp_n_degrees_of_freedom = Quantity(
        type=np.int32,
        shape=[],
        description="""
        """,
    )

    x_gulp_equilibration_time = Quantity(
        type=np.float64,
        shape=[],
        unit='s',
        description="""
        """,
    )

    x_gulp_production_time = Quantity(
        type=np.float64,
        shape=[],
        unit='s',
        description="""
        """,
    )

    x_gulp_scaling_time = Quantity(
        type=np.float64,
        shape=[],
        unit='s',
        description="""
        """,
    )

    x_gulp_scaling_frequency = Quantity(
        type=np.float64,
        shape=[],
        unit='s',
        description="""
        """,
    )

    x_gulp_sampling_frequency = Quantity(
        type=np.float64,
        shape=[],
        unit='s',
        description="""
        """,
    )

    x_gulp_write_frequency = Quantity(
        type=np.float64,
        shape=[],
        unit='s',
        description="""
        """,
    )

    x_gulp_td_force_start_time = Quantity(
        type=np.float64,
        shape=[],
        unit='s',
        description="""
        """,
    )

    x_gulp_td_field_start_time = Quantity(
        type=np.float64,
        shape=[],
        unit='s',
        description="""
        """,
    )
