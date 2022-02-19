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


m_package = Package()


class x_gulp_section_main_keyword(MSection):
    '''
    Section for GULP calculation mode input variable
    '''

    m_def = Section(validate=False)

    x_gulp_main_keyword = Quantity(
        type=str,
        shape=[],
        description='''
        GULP calculation mode input variable
        ''')


class x_gulp_section_forcefield(MSection):
    '''
    Section for GULP force field specification
    '''

    m_def = Section(validate=False)

    x_gulp_forcefield_species_1 = Quantity(
        type=str,
        shape=[],
        description='''
        GULP force field species 1
        ''')

    x_gulp_forcefield_species_2 = Quantity(
        type=str,
        shape=[],
        description='''
        GULP force field species 2
        ''')

    x_gulp_forcefield_species_3 = Quantity(
        type=str,
        shape=[],
        description='''
        GULP force field species 3
        ''')

    x_gulp_forcefield_species_4 = Quantity(
        type=str,
        shape=[],
        description='''
        GULP force field species 4
        ''')

    x_gulp_forcefield_speciestype_1 = Quantity(
        type=str,
        shape=[],
        description='''
        GULP force field speciestype 1
        ''')

    x_gulp_forcefield_speciestype_2 = Quantity(
        type=str,
        shape=[],
        description='''
        GULP force field speciestype 2
        ''')

    x_gulp_forcefield_speciestype_3 = Quantity(
        type=str,
        shape=[],
        description='''
        GULP force field speciestype 3
        ''')

    x_gulp_forcefield_speciestype_4 = Quantity(
        type=str,
        shape=[],
        description='''
        GULP force field speciestype 4
        ''')

    x_gulp_forcefield_potential_name = Quantity(
        type=str,
        shape=[],
        description='''
        GULP force field potential name
        ''')

    x_gulp_forcefield_parameter_a = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        GULP force field parameter A
        ''')

    x_gulp_forcefield_parameter_b = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        GULP force field parameter B
        ''')

    x_gulp_forcefield_parameter_c = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        GULP force field parameter C
        ''')

    x_gulp_forcefield_parameter_d = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        GULP force field parameter D
        ''')

    x_gulp_forcefield_cutoff_min = Quantity(
        type=str,
        shape=[],
        description='''
        GULP force field cutoff min (can also be a string like 3Bond for some reason)
        ''')

    x_gulp_forcefield_cutoff_max = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        GULP force field cutoff max
        ''')

    x_gulp_forcefield_threebody_1 = Quantity(
        type=str,
        shape=[],
        description='''
        GULP 3-body force field parameter 1
        ''')

    x_gulp_forcefield_threebody_2 = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        GULP 3-body force field parameter 2
        ''')

    x_gulp_forcefield_threebody_3 = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        GULP 3-body force field parameter 3
        ''')

    x_gulp_forcefield_threebody_theta = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        GULP 3-body force field parameter theta
        ''')

    x_gulp_forcefield_fourbody_force_constant = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        GULP 4-body force field parameter force constant
        ''')

    x_gulp_forcefield_fourbody_sign = Quantity(
        type=str,
        shape=[],
        description='''
        GULP 4-body force field parameter sign
        ''')

    x_gulp_forcefield_fourbody_phase = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        GULP 4-body force field parameter phase
        ''')

    x_gulp_forcefield_fourbody_phi0 = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        GULP 4-body force field parameter phi0
        ''')


class System(simulation.system.System):

    m_def = Section(validate=False, extends_base_section=True)

    x_gulp_patterson_group = Quantity(
        type=str,
        shape=[],
        description='''
        Patterson group
        ''')

    x_gulp_space_group = Quantity(
        type=str,
        shape=[],
        description='''
        Space group
        ''')

    x_gulp_formula = Quantity(
        type=str,
        shape=[],
        description='''
        GULP chemical formula
        ''')

    x_gulp_cell_alpha = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        grrr
        ''')

    x_gulp_cell_beta = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        grrr
        ''')

    x_gulp_cell_gamma = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        grrr
        ''')

    x_gulp_cell_a = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        grrr
        ''')

    x_gulp_cell_b = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        grrr
        ''')

    x_gulp_cell_c = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        grrr
        ''')

    x_gulp_prim_cell_alpha = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        grrr
        ''')

    x_gulp_prim_cell_beta = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        grrr
        ''')

    x_gulp_prim_cell_gamma = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        grrr
        ''')

    x_gulp_prim_cell_a = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        grrr
        ''')

    x_gulp_prim_cell_b = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        grrr
        ''')

    x_gulp_prim_cell_c = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        grrr
        ''')

    x_gulp_pbc = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        grrr
        ''')


class Run(simulation.run.Run):

    m_def = Section(validate=False, extends_base_section=True)

    x_gulp_title = Quantity(
        type=str,
        shape=[],
        description='''
        Title of GULP calculation
        ''')

    x_gulp_section_main_keyword = SubSection(
        sub_section=SectionProxy('x_gulp_section_main_keyword'),
        repeats=True)


class Method(simulation.method.Method):

    m_def = Section(validate=False, extends_base_section=True)

    x_gulp_number_of_species = Quantity(
        type=int,
        shape=[],
        description='''
        Number of species in GULP
        ''')

    x_gulp_species_charge = Quantity(
        type=np.dtype(np.float64),
        shape=['x_gulp_number_of_species'],
        description='''
        Number of species in GULP
        ''')

    x_gulp_section_forcefield = SubSection(
        sub_section=SectionProxy('x_gulp_section_forcefield'),
        repeats=True)


class Calculation(simulation.calculation.Calculation):

    m_def = Section(validate=False, extends_base_section=True)

    x_gulp_energy_attachment_energy = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        GULP energy term for attachment_energy
        ''')

    x_gulp_energy_attachment_energy_unit = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        GULP energy term for attachment_energy_unit
        ''')

    x_gulp_energy_bond_order_potentials = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        GULP energy term for bond_order_potentials
        ''')

    x_gulp_energy_brenner_potentials = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        GULP energy term for brenner_potentials
        ''')

    x_gulp_energy_bulk_energy = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        GULP energy term for bulk_energy
        ''')

    x_gulp_energy_dispersion_real_recip = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        GULP energy term for dispersion_real_recip
        ''')

    x_gulp_energy_electric_field_times_distance = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        GULP energy term for electric_field_times_distance
        ''')

    x_gulp_energy_energy_shift = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        GULP energy term for energy_shift
        ''')

    x_gulp_energy_four_body_potentials = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        GULP energy term for four_body_potentials
        ''')

    x_gulp_energy_improper_torsions = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        GULP energy term for improper_torsions
        ''')

    x_gulp_energy_interatomic_potentials = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        GULP energy term for interatomic_potentials
        ''')

    x_gulp_energy_many_body_potentials = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        GULP energy term for many_body_potentials
        ''')

    x_gulp_energy_monopole_monopole_real = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        GULP energy term for monopole_monopole_real
        ''')

    x_gulp_energy_monopole_monopole_recip = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        GULP energy term for monopole_monopole_recip
        ''')

    x_gulp_energy_monopole_monopole_total = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        GULP energy term for monopole_monopole_total
        ''')

    x_gulp_energy_neutralising_energy = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        GULP energy term for neutralising_energy
        ''')

    x_gulp_energy_non_primitive_unit_cell = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        GULP energy term for non_primitive_unit_cell
        ''')

    x_gulp_energy_out_of_plane_potentials = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        GULP energy term for out_of_plane_potentials
        ''')

    x_gulp_energy_primitive_unit_cell = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        GULP energy term for primitive_unit_cell
        ''')

    x_gulp_energy_reaxff_force_field = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        GULP energy term for reaxff_force_field
        ''')

    x_gulp_energy_region_1_2_interaction = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        GULP energy term for region_1_2_interaction
        ''')

    x_gulp_energy_region_2_2_interaction = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        GULP energy term for region_2_2_interaction
        ''')

    x_gulp_energy_self_energy_eem_qeq_sm = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        GULP energy term for self_energy_eem_qeq_sm
        ''')

    x_gulp_energy_sm_coulomb_correction = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        GULP energy term for sm_coulomb_correction
        ''')

    x_gulp_energy_solvation_energy = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        GULP energy term for solvation_energy
        ''')

    x_gulp_energy_three_body_potentials = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        GULP energy term for three_body_potentials
        ''')

    x_gulp_energy_total_lattice_energy = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        GULP energy term for total_lattice_energy
        ''')

    x_gulp_md_time = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        GULP molecular dynamics time
        ''')

    x_gulp_md_kinetic_energy = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        GULP molecular dynamics kinetic energy
        ''')

    x_gulp_md_potential_energy = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        GULP molecular dynamics potential energy
        ''')

    x_gulp_md_total_energy = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        GULP molecular dynamics total energy
        ''')

    x_gulp_md_temperature = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        GULP molecular dynamics temperature
        ''')

    x_gulp_md_pressure = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        GULP molecular dynamics pressure
        ''')
