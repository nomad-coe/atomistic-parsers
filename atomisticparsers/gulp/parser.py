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
import os
import logging
import numpy as np
from datetime import datetime
from ase.atoms import Atoms as ase_Atoms
from ase.spacegroup import crystal as ase_crystal

from nomad.units import ureg
from nomad.parsing.file_parser import TextParser, Quantity, Parser
from runschema.run import Run, Program, TimeRun
from runschema.method import Method, ForceField, Model, Interaction, AtomParameters
from runschema.system import System, Atoms
from runschema.calculation import Calculation, Energy, EnergyEntry
from simulationworkflowschema import (
    Elastic,
    ElasticMethod,
    ElasticResults,
    MolecularDynamics,
    MolecularDynamicsMethod,
)
from atomisticparsers.gulp.metainfo.gulp import (
    x_gulp_bulk_optimisation,
    x_gulp_bulk_optimisation_cycle,
)


re_f = r'[-+]?\d+\.\d*(?:[Ee][-+]\d+)?'
re_n = r'[\n\r]'


class MainfileParser(TextParser):
    def init_quantities(self):
        def to_species(val_in):
            data = dict(
                label=[],
                x_gulp_type=[],
                atom_number=[],
                mass=[],
                charge=[],
                x_gulp_covalent_radius=[],
                x_gulp_ionic_radius=[],
                x_gulp_vdw_radius=[],
            )
            for val in val_in.strip().splitlines():
                val = val.strip().split()
                data['label'].append(val[0])
                data['x_gulp_type'].append(val[1].lower())
                data['atom_number'].append(int(val[2]))
                data['mass'].append(float(val[3]) * ureg.amu)
                data['charge'].append(float(val[4]) * ureg.elementary_charge)
                data['x_gulp_covalent_radius'].append(float(val[5]) * ureg.angstrom)
                data['x_gulp_ionic_radius'].append(float(val[6]) * ureg.angstrom)
                data['x_gulp_vdw_radius'].append(float(val[7]) * ureg.angstrom)
            return data

        def to_cell_parameters(val_in):
            val = val_in.strip().split()
            return np.array(
                [val[0], val[2], val[4], val[1], val[3], val[5]], np.dtype(np.float64)
            )

        coordinates_quantities = [
            Quantity('unit', r'Label +\((\w+)\)', dtype=str),
            Quantity(
                'auxilliary_keys', r'x +y +z +(.+)', str_operation=lambda x: x.split()
            ),
            Quantity(
                'atom',
                rf'\d+ +([A-Z][a-z]*)\S* +(\w+ +{re_f}.+)',
                repeats=True,
                str_operation=lambda x: x.strip().replace('*', '').split(),
            ),
        ]

        calc_quantities = [
            Quantity(
                'energy_components',
                rf'Components of .*energy \:\s+([\s\S]+?){re_n} *{re_n}',
                sub_parser=TextParser(
                    quantities=[
                        Quantity(
                            'key_val',
                            rf'(.+) *= +({re_f}) eV',
                            repeats=True,
                            str_operation=lambda x: [
                                v.strip() for v in x.rsplit(' ', 1)
                            ],
                        )
                    ]
                ),
            ),
            Quantity(
                'bulk_optimisation',
                rf'(Number of variables += +\d+[\s\S]+?Start of bulk optimisation \:\s+[\s\S]+?){re_n} *{re_n}',
                sub_parser=TextParser(
                    quantities=[
                        Quantity(
                            'x_gulp_n_variables',
                            r'Number of variables += +(\d+)',
                            dtype=np.int32,
                        ),
                        Quantity(
                            'x_gulp_max_n_calculations',
                            r'Maximum number of calculations += +(\d+)',
                            dtype=np.int32,
                        ),
                        Quantity(
                            'x_gulp_max_hessian_update_interval',
                            r'Maximum Hessian update interval += +(\d+)',
                            dtype=np.int32,
                        ),
                        Quantity(
                            'x_gulp_max_step_size',
                            rf'Maximum step size  += +({re_f})',
                            dtype=np.float64,
                        ),
                        Quantity(
                            'x_gulp_max_parameter_tolerance',
                            rf'Maximum parameter tolerance += +({re_f})',
                            dtype=np.float64,
                        ),
                        Quantity(
                            'x_gulp_max_function_tolerance',
                            rf'Maximum function +tolerance += +({re_f})',
                            dtype=np.float64,
                        ),
                        Quantity(
                            'x_gulp_max_gradient_tolerance',
                            rf'Maximum gradient +tolerance += +({re_f})',
                            dtype=np.float64,
                        ),
                        Quantity(
                            'x_gulp_max_gradient_component',
                            rf'Maximum gradient +component += +({re_f})',
                            dtype=np.float64,
                        ),
                        Quantity(
                            'cycle',
                            rf'Cycle\: +\d+ +Energy\: +({re_f}) +Gnorm\: +({re_f}) +CPU\: +({re_f})',
                            repeats=True,
                            dtype=np.dtype(np.float64),
                        ),
                    ]
                ),
            ),
            Quantity(
                'coordinates',
                r'Final.+coordinates.+\:\s+\-+\s+'
                r'No\. +Atomic +(x +y +z +.+\s+Label .+)\s*\-+\s*([\s\S]+?)\-{50}',
                sub_parser=TextParser(quantities=coordinates_quantities),
            ),
            Quantity(
                'lattice_vectors',
                rf'Final Cartesian lattice vectors \(Angstroms\) \:\s+((?:{re_f}\s+)+)',
                dtype=np.dtype(np.float64),
                shape=(3, 3),
            ),
            Quantity(
                'cell_parameters_primitive',
                rf'Final cell parameters and derivatives \:\s+\-+\s+'
                rf'((?:\w+ +{re_f}.+\s+)+)',
                str_operation=lambda x: np.array(
                    [v.split()[1] for v in x.strip().splitlines()], np.dtype(np.float64)
                ),
            ),
            Quantity(
                'cell_parameters',
                r'Non\-primitive lattice parameters \:\s+'
                rf'a += +({re_f}) +b += +({re_f}) +c += +({re_f})\s+'
                rf'alpha *= +({re_f}) +beta *= +({re_f}) +gamma *= +({re_f})\s+',
                dtype=np.dtype(np.float64),
            ),
            Quantity(
                'elastic_constants',
                r'Elastic Constant Matrix.+\s+\-+\s+Indices.+\s+\-+\s+'
                rf'((?:\d+ +{re_f} +{re_f} +{re_f} +{re_f} +{re_f} +{re_f}\s+)+)',
                str_operation=lambda x: np.array(
                    [v.split()[1:7] for v in x.strip().splitlines()],
                    np.dtype(np.float64),
                )
                * ureg.GPa,
            ),
            Quantity(
                'elastic_compliance',
                r'Elastic Compliance Matrix.+\s+\-+\s+Indices.+\s+\-+\s+'
                rf'((?:\d+ +{re_f} +{re_f} +{re_f} +{re_f} +{re_f} +{re_f}\s+)+)',
                str_operation=lambda x: np.array(
                    [v.split()[1:7] for v in x.strip().splitlines()],
                    np.dtype(np.float64),
                )
                * 1
                / ureg.GPa,
            ),
            Quantity(
                'mechanical_properties',
                rf'Mechanical properties \:\s+([\s\S]+?){re_n} *{re_n}',
                sub_parser=TextParser(
                    quantities=[
                        Quantity(
                            'bulk_modulus',
                            rf'Bulk +Modulus \(GPa\) += +({re_f} +{re_f} +{re_f})',
                            dtype=np.dtype(np.float64),
                            unit=ureg.GPa,
                        ),
                        Quantity(
                            'shear_modulus',
                            rf'Shear +Modulus \(GPa\) += +({re_f} +{re_f} +{re_f})',
                            dtype=np.dtype(np.float64),
                            unit=ureg.GPa,
                        ),
                        Quantity(
                            'x_gulp_velocity_s_wave',
                            rf'Velocity S\-wave \(km/s\) += +({re_f} +{re_f} +{re_f})',
                            dtype=np.dtype(np.float64),
                            unit=ureg.km / ureg.s,
                        ),
                        Quantity(
                            'x_gulp_velocity_p_wave',
                            rf'Velocity P\-wave \(km/s\) += +({re_f} +{re_f} +{re_f})',
                            dtype=np.dtype(np.float64),
                            unit=ureg.km / ureg.s,
                        ),
                        Quantity(
                            'compressibility',
                            rf'Compressibility \(1/GPa\) += +({re_f})',
                            dtype=np.float64,
                            unit=1 / ureg.GPa,
                        ),
                        Quantity(
                            'x_gulp_youngs_modulus',
                            rf'Youngs Moduli \(GPa\) += +({re_f} +{re_f} +{re_f})',
                            dtype=np.dtype(np.float64),
                            unit=ureg.GPa,
                        ),
                        Quantity(
                            'poissons_ratio',
                            rf'Poissons Ratio \((?:x|y|z)\) += +(.+)',
                            dtype=np.dtype(np.float64),
                            repeats=True,
                        ),
                    ]
                ),
            ),
            Quantity(
                'x_gulp_piezoelectric_strain_matrix',
                rf'Piezoelectric Strain Matrix\: \(Units=C/m\*\*2\)\s+\-+\s+'
                rf'Indices.+\s*\-+\s+((?:\w +{re_f}.+\s+)+)',
                dtype=np.dtype(np.float64),
                str_operation=lambda x: np.array(
                    [v.strip().split()[1:7] for v in x.strip().splitlines()],
                    np.dtype(np.float64),
                )
                * ureg.C
                / ureg.m**2,
            ),
            Quantity(
                'x_gulp_piezoelectric_stress_matrix',
                rf'Piezoelectric Stress Matrix\: \(Units=10\*\*\-11 C/N\)\s+\-+\s+'
                rf'Indices.+\s*\-+\s+((?:\w +{re_f}.+\s+)+)',
                dtype=np.dtype(np.float64),
                str_operation=lambda x: np.array(
                    [v.strip().split()[1:7] for v in x.strip().splitlines()],
                    np.dtype(np.float64),
                )
                * 10**-11
                * ureg.C
                / ureg.N,
            ),
            Quantity(
                'x_gulp_static_dielectric_constant_tensor',
                r'Static dielectric constant tensor \:\s+\-+\s+x +y +z\s+\-+\s+'
                rf'((?:\w +{re_f} +{re_f} +{re_f}\s+)+)',
                str_operation=lambda x: np.array(
                    [v.strip().split()[1:4] for v in x.strip().splitlines()],
                    np.dtype(np.float64),
                ),
            ),
            Quantity(
                'x_gulp_high_frequency_dielectric_constant_tensor',
                r'High frequency dielectric constant tensor \:\s+\-+\s+x +y +z\s+\-+\s+'
                rf'((?:\w +{re_f} +{re_f} +{re_f}\s+)+)',
                str_operation=lambda x: np.array(
                    [v.strip().split()[1:4] for v in x.strip().splitlines()],
                    np.dtype(np.float64),
                ),
            ),
            Quantity(
                'x_gulp_static_refractive_indices',
                r'Static refractive indices \:\s+\-+\s+([\s\S]+?)\-{50}',
                sub_parser=TextParser(
                    quantities=[
                        Quantity(
                            'value',
                            rf'\d+ += +({re_f})',
                            repeats=True,
                            dtype=np.float64,
                        )
                    ]
                ),
            ),
            Quantity(
                'x_gulp_high_frequency_refractive_indices',
                r'High frequency refractive indices \:\s+\-+\s+([\s\S]+?)\-{50}',
                sub_parser=TextParser(
                    quantities=[
                        Quantity(
                            'value',
                            rf'\d+ += +({re_f})',
                            repeats=True,
                            dtype=np.float64,
                        )
                    ]
                ),
            ),
        ]

        interaction_quantities = [
            Quantity('atom_type', r'([A-Z]\S* +(?:core|shell|c|s))\s', repeats=True),
            Quantity(
                'functional_form',
                r'([A-Z].{12})',
                dtype=str,
                flatten=False,
                str_operation=lambda x: x.strip(),
            ),
        ]

        self._quantities = [
            Quantity(
                'header',
                rf'(Version[\s\S]+?){re_n} *{re_n}',
                sub_parser=TextParser(
                    quantities=[
                        Quantity('program_version', r'Version = (\S+)', dtype=str),
                        Quantity('task', r'\* +(\w+) +\- .+', repeats=True, dtype=str),
                        Quantity(
                            'title',
                            r'\*\*\*\s+\* +(.+?) +\*\s+\*\*\*',
                            dtype=str,
                            flatten=False,
                        ),
                    ]
                ),
            ),
            Quantity(
                'date_start',
                r'Job Started +at (\d+\:\d+\.\d+) (\d+)\w+ (\w+) +(\d+)',
                dtype=str,
                flatten=False,
            ),
            Quantity(
                'date_end',
                r'Job Started +at (\d+\:\d+\.\d+) (\d+)\w+ (\w+) +(\d+)',
                dtype=str,
                flatten=False,
            ),
            Quantity('x_gulp_n_cpu', r'Number of CPUs += +(\d+)', dtype=np.int32),
            Quantity(
                'x_gulp_host_name', r'Host name += +(\S+)', dtype=str, flatten=False
            ),
            Quantity(
                'x_gulp_total_n_configurations_input',
                r'Total number of configurations input += +(\d+)',
                dtype=np.int32,
            ),
            Quantity(
                'input_configuration',
                r'(Input for Configuration.+\s*\*+[\s\S]+?)\*{80}',
                sub_parser=TextParser(
                    quantities=[
                        Quantity('x_gulp_formula', r'Formula = (\S+)', dtype=str),
                        Quantity(
                            'x_gulp_pbc', r'Dimensionality = (\d+)', dtype=np.int32
                        ),
                        Quantity(
                            'x_gulp_space_group',
                            rf'Space group \S+ +\: +(.+?) +{re_n}',
                            dtype=str,
                            flatten=False,
                        ),
                        Quantity(
                            'x_gulp_patterson_group',
                            rf'Patterson group +\: +(.+?) +{re_n}',
                            dtype=str,
                            flatten=False,
                        ),
                        Quantity(
                            'lattice_vectors',
                            rf'Cartesian lattice vectors \(Angstroms\) \:\s+'
                            rf'((?:{re_f} +{re_f} +{re_f}\s+)+)',
                            dtype=np.dtype(np.float64),
                            shape=[3, 3],
                        ),
                        Quantity(
                            'cell_parameters',
                            rf'Primitive cell parameters.+\s+a.+?a += +({re_f}) +alpha += +({re_f})\s+'
                            rf'b.+?b += +({re_f}) +beta += +({re_f})\s+'
                            rf'c.+?c += +({re_f}) +gamma += +({re_f})\s+',
                            str_operation=to_cell_parameters,
                        ),
                        Quantity(
                            'cell_parameters',
                            rf'Cell parameters.+\s+a += +({re_f}) +alpha += +({re_f})\s+'
                            rf'b += +({re_f}) +beta += +({re_f})\s+'
                            rf'c += +({re_f}) +gamma += +({re_f})\s+',
                            str_operation=to_cell_parameters,
                        ),
                        Quantity(
                            'coordinates',
                            r'No\. +Atomic +(x +y +z +.+\s+Label .+)\s*\-+\s*([\s\S]+?)\-{50}',
                            sub_parser=TextParser(quantities=coordinates_quantities),
                        ),
                    ]
                ),
            ),
            Quantity(
                'input_information',
                r'(General input information\s+\*\s*\*+[\s\S]+?)\*{80}',
                sub_parser=TextParser(
                    quantities=[
                        Quantity(
                            'species',
                            rf'Species +Type.+\s+Number.+\s*\-+\s+'
                            rf'((?:[A-Z]\w* +\w+ +\d+.+\s+)+)',
                            str_operation=to_species,
                            convert=False,
                        ),
                        Quantity(
                            'pgfnff',
                            r'(pGFNFF forcefield to be used[\s\S]+pGFNFF.+)',
                            sub_parser=TextParser(
                                quantities=[
                                    Quantity(
                                        'key_parameter',
                                        rf'pGFNFF (.+? += +{re_f})',
                                        repeats=True,
                                        str_operation=lambda x: [
                                            v.strip() for v in x.split('=')
                                        ],
                                    )
                                ]
                            ),
                        ),
                        # old format
                        Quantity(
                            'pair_potential',
                            r'(Atom +Types +Potential +A +B +C +D +.*\s+)'
                            r'(.+\s*\-+\s*[\s\S]+?)\-{80}',
                            repeats=True,
                            sub_parser=TextParser(
                                quantities=[
                                    Quantity(
                                        'interaction',
                                        r'([A-Z]\S* +(?:core|shell|c|s) +[A-Z]\S* +(?:core|shell|c|s) +.+)',
                                        repeats=True,
                                        sub_parser=TextParser(
                                            quantities=interaction_quantities
                                            + [
                                                Quantity(
                                                    'key_parameter',
                                                    rf'({re_f}) +({re_f}) +({re_f}) +({re_f}) +({re_f}) +({re_f})',
                                                    str_operation=lambda x: list(
                                                        zip(
                                                            [
                                                                'A',
                                                                'B',
                                                                'C',
                                                                'D',
                                                                'cutoff_min',
                                                                'cutoff_max',
                                                            ],
                                                            np.array(
                                                                x.strip().split(),
                                                                np.dtype(np.float64),
                                                            ),
                                                        )
                                                    ),
                                                )
                                            ]
                                        ),
                                    )
                                ]
                            ),
                        ),
                        # TODO verify this no example
                        Quantity(
                            'three_body_potential',
                            r'(Atom +Atom +Atom +Force Constants +Theta.*\s+)'
                            r'(.+\s*\-+\s*[\s\S]+?)\-{80}',
                            repeats=True,
                            sub_parser=TextParser(
                                quantities=[
                                    Quantity(
                                        'interaction',
                                        r'([A-Z]\S* +(?:core|shell|c|s) +[A-Z]\S* +(?:core|shell|c|s) +.+)',
                                        repeats=True,
                                        sub_parser=TextParser(
                                            quantities=interaction_quantities
                                            + [
                                                Quantity(
                                                    'key_parameter',
                                                    rf'({re_f}) +({re_f}) +({re_f}) +({re_f})',
                                                    str_operation=lambda x: list(
                                                        zip(
                                                            [
                                                                'force_constant_1',
                                                                'force_constant_2',
                                                                'force_constant_3',
                                                                'Theta',
                                                            ],
                                                            np.array(
                                                                x.strip().split(),
                                                                np.dtype(np.float64),
                                                            ),
                                                        )
                                                    ),
                                                )
                                            ]
                                        ),
                                    )
                                ]
                            ),
                        ),
                        # TODO verify this no example
                        Quantity(
                            'four_body_potential',
                            r'(Atom Types +Force cst\s*Sign\s*Phase\s*Phi0.*\s+)'
                            r'(.+\s*\-+\s*[\s\S]+?)\-{80}',
                            repeats=True,
                            sub_parser=TextParser(
                                quantities=[
                                    Quantity(
                                        'interaction',
                                        r'([A-Z]\S* +(?:core|shell|c|s) +[A-Z]\S* +(?:core|shell|c|s) +.+)',
                                        repeats=True,
                                        sub_parser=TextParser(
                                            quantities=interaction_quantities
                                            + [
                                                Quantity(
                                                    'key_parameter',
                                                    rf'({re_f}) +({re_f}) +({re_f}) +({re_f})',
                                                    str_operation=lambda x: list(
                                                        zip(
                                                            [
                                                                'force_constant',
                                                                'sign',
                                                                'phase',
                                                                'phi0',
                                                            ],
                                                            np.array(
                                                                x.strip().split(),
                                                                np.dtype(np.float64),
                                                            ),
                                                        )
                                                    ),
                                                )
                                            ]
                                        ),
                                    )
                                ]
                            ),
                        ),
                        # new format
                        Quantity(
                            'interatomic_potential',
                            rf'.+? potentials +\:\s+\-+\s+'
                            rf'Atom.+?Potential +Parameter([\s\S]+?){re_n} *{re_n}',
                            repeats=True,
                            sub_parser=TextParser(
                                quantities=[
                                    Quantity(
                                        'interaction',
                                        # r'([A-Z]\S* +(?:core|shell|c|s) +[A-Z]\S* +(?:core|shell|c|s) +[\s\S]+?\-{80})',
                                        r'([A-Z]\S* +(?:core|shell|c|s) +[\s\S]+?\-{80})',
                                        repeats=True,
                                        sub_parser=TextParser(
                                            quantities=interaction_quantities
                                            + [
                                                Quantity(
                                                    'key_parameter',
                                                    r'([A-Z].{14}) {1,4}(\-*\d+\S*)',
                                                    repeats=True,
                                                    str_operation=lambda x: [
                                                        v.strip()
                                                        for v in x.rsplit(' ', 1)
                                                    ],
                                                )
                                            ]
                                        ),
                                    )
                                ]
                            ),
                        ),
                    ]
                ),
            ),
            Quantity(
                'single_point',
                r'Output for configuration.+\s+\*+([\s\S]+?)(?:\*{80}|\Z)',
                repeats=True,
                sub_parser=TextParser(
                    quantities=[
                        Quantity(
                            'calculation',
                            r'(Components of.+energy \:[\s\S]+?(?:Time to end of|Optimisation achieved|\Z).*)',
                            repeats=True,
                            sub_parser=TextParser(quantities=calc_quantities),
                        )
                    ]
                ),
            ),
            Quantity(
                'molecular_dynamics',
                r'Molecular Dynamics.+\s+\*+([\s\S]+?)(?:\*{80}|\Z)',
                repeats=True,
                sub_parser=TextParser(
                    quantities=[
                        Quantity(
                            'ensemble_type', r'ensemble \((\S+)\) to be used', dtype=str
                        ),
                        Quantity(
                            'x_gulp_friction_temperature_bath',
                            rf'Friction for temperature bath += +({re_f})',
                            dtype=np.float64,
                        ),
                        Quantity(
                            'x_gulp_n_mobile_ions',
                            r'No\. of mobile ions += +(\d+)',
                            dtype=np.int32,
                        ),
                        Quantity(
                            'x_gulp_n_degrees_of_freedom',
                            r'No\. of degrees of freedom += +(\d+)',
                            dtype=np.int32,
                        ),
                        Quantity(
                            'timestep',
                            rf'Time step += +({re_f})',
                            dtype=np.float64,
                            unit=ureg.ps,
                        ),
                        Quantity(
                            'x_gulp_equilibration_time',
                            rf'Equilibration time += +({re_f})',
                            dtype=np.float64,
                            unit=ureg.ps,
                        ),
                        Quantity(
                            'x_gulp_production_time',
                            rf'Production time += +({re_f})',
                            dtype=np.float64,
                            unit=ureg.ps,
                        ),
                        Quantity(
                            'x_gulp_scaling_time',
                            rf'Scaling time += +({re_f})',
                            dtype=np.float64,
                            unit=ureg.ps,
                        ),
                        Quantity(
                            'x_gulp_scaling_frequency',
                            rf'Scaling frequency += +({re_f})',
                            dtype=np.float64,
                            unit=ureg.ps,
                        ),
                        Quantity(
                            'x_gulp_sampling_frequency',
                            rf'Sampling frequency += +({re_f})',
                            dtype=np.float64,
                            unit=ureg.ps,
                        ),
                        Quantity(
                            'x_gulp_write_frequency',
                            rf'Write frequency += +({re_f})',
                            dtype=np.float64,
                            unit=ureg.ps,
                        ),
                        Quantity(
                            'x_gulp_td_force_start_time',
                            rf'TD\-Force start time += +({re_f})',
                            dtype=np.float64,
                            unit=ureg.ps,
                        ),
                        Quantity(
                            'x_gulp_td_field_start_time',
                            rf'TD\-Field start time += +({re_f})',
                            dtype=np.float64,
                            unit=ureg.ps,
                        ),
                        Quantity(
                            'step',
                            rf'(Time \: +[\s\S]+?)(?:\*\*|{re_n} *{re_n})',
                            repeats=True,
                            sub_parser=TextParser(
                                quantities=[
                                    Quantity(
                                        'time',
                                        rf'Time \: +({re_f})',
                                        dtype=np.float64,
                                        unit=ureg.ps,
                                    ),
                                    Quantity(
                                        'energy_kinetic',
                                        rf'Kinetic energy +\(eV\) += +({re_f}) +({re_f})',
                                        dtype=np.dtype(np.float64),
                                        unit=ureg.eV,
                                    ),
                                    Quantity(
                                        'energy_potential',
                                        rf'Potential energy +\(eV\) += +({re_f}) +({re_f})',
                                        dtype=np.dtype(np.float64),
                                        unit=ureg.eV,
                                    ),
                                    Quantity(
                                        'energy_total',
                                        rf'Total energy +\(eV\) += +({re_f}) +({re_f})',
                                        dtype=np.dtype(np.float64),
                                        unit=ureg.eV,
                                    ),
                                    Quantity(
                                        'temperature',
                                        rf'Temperature +\(K\) += +({re_f}) +({re_f})',
                                        dtype=np.dtype(np.float64),
                                        unit=ureg.kelvin,
                                    ),
                                    Quantity(
                                        'pressure',
                                        rf'Pressure +\(GPa\) += +({re_f}) +({re_f})',
                                        dtype=np.dtype(np.float64),
                                        unit=ureg.GPa,
                                    ),
                                ]
                            ),
                        ),
                    ]
                ),
            ),
            Quantity(
                'defect',
                r'Defect calculation for configuration.+\s+\*+([\s\S]+?)(?:\*{80}|\Z)',
                repeats=True,
                sub_parser=TextParser(
                    quantities=[
                        Quantity(
                            'calculation',
                            r'(Components of.+energy \:[\s\S]+?(?:Time to end of|Optimisation achieved).+)',
                            repeats=True,
                            sub_parser=TextParser(quantities=calc_quantities),
                        )
                    ]
                ),
            ),
        ]


class GulpParser(Parser):
    def __init__(self):
        self.mainfile_parser = MainfileParser()
        self._metainfo_map = {
            'Attachment energy': 'x_gulp_attachment',
            'Attachment energy/unit': 'x_gulp_attachment_unit',
            'Bond-order potentials': 'x_gulp_bond_order_potentials',
            'Brenner potentials': 'x_gulp_brenner_potentials',
            'Bulk energy': 'x_gulp_bulk',
            'Dispersion (real+recip)': 'x_gulp_dispersion',
            'Electric_field*distance': 'x_gulp_electric_field_distance',
            'Energy shift': 'x_gulp_shift',
            'Four-body potentials': 'x_gulp_four_body_potentials',
            'Improper torsions': 'x_gulp_improper_torsions',
            'Interatomic potentials': 'x_gulp_interatomic_potentials',
            'Many-body potentials': 'x_gulp_many_body_potentials',
            'Monopole - monopole (real)': 'x_gulp_monopole_monopole_real',
            'Monopole - monopole (recip)': 'x_gulp_monopole_monopole_recip',
            'Monopole - monopole (total)': 'x_gulp_monopole_monopole_total',
            'Neutralising energy': 'x_gulp_neutralising',
            'Non-primitive unit cell': 'x_gulp_non_primitive_unit_cell',
            'Out of plane potentials': 'x_gulp_out_of_plane_potentials',
            'Primitive unit cell': 'x_gulp_primitive_unit_cell',
            'ReaxFF force field': 'x_gulp_reaxff_force_field',
            'Region 1-2 interaction': 'x_gulp_region_1_2_interaction',
            'Region 2-2 interaction': 'x_gulp_region_2_2_interaction',
            'Self energy (EEM/QEq/SM)': 'x_gulp_self_eem_qeq_sm',
            'SM Coulomb correction': 'x_gulp_sm_coulomb_correction',
            'Solvation energy': 'x_gulp_solvation',
            'Three-body potentials': 'x_gulp_three_body_potentials',
            'Total lattice energy': 'total',
            'Total defect energy': 'total',
        }
        self._sg_map = {
            'P 1': 1,
            'P -1': 2,
            'P 2': 3,
            'P 21': 4,
            'C 2': 5,
            'P M': 6,
            'P C': 7,
            'C M': 8,
            'C C': 9,
            'P 2/M': 10,
            'P 21/M': 11,
            'C 2/M': 12,
            'P 2/C': 13,
            'P 21/C': 14,
            'C 2/C': 15,
            'P 2 2 2': 16,
            'P 2 2 21': 17,
            'P 21 21 2': 18,
            'P 21 21 21': 19,
            'C 2 2 21': 20,
            'C 2 2 2': 21,
            'F 2 2 2': 22,
            'I 2 2 2': 23,
            'I 21 21 21': 24,
            'P M M 2': 25,
            'P M C 21': 26,
            'P C C 2': 27,
            'P M A 2': 28,
            'P C A 21': 29,
            'P N C 2': 30,
            'P M N 21': 31,
            'P B A 2': 32,
            'P N A 21': 33,
            'P N N 2': 34,
            'C M M 2': 35,
            'C M C 21': 36,
            'C C C 2': 37,
            'A M M 2': 38,
            'A B M 2': 39,
            'A M A 2': 40,
            'A B A 2': 41,
            'F M M 2': 42,
            'F D D 2': 43,
            'I M M 2': 44,
            'I B A 2': 45,
            'I M A 2': 46,
            'P M M M': 47,
            'P N N N': 48,
            'P C C M': 49,
            'P B A N': 50,
            'P M M A': 51,
            'P N N A': 52,
            'P M N A': 53,
            'P C C A': 54,
            'P B A M': 55,
            'P C C N': 56,
            'P B C M': 57,
            'P N N M': 58,
            'P M M N': 59,
            'P B C N': 60,
            'P B C A': 61,
            'P N M A': 62,
            'C M C M': 63,
            'C M C A': 64,
            'C M M M': 65,
            'C C C M': 66,
            'C M M A': 67,
            'C C C A': 68,
            'F M M M': 69,
            'F D D D': 70,
            'I M M M': 71,
            'I B A M': 72,
            'I B C A': 73,
            'I M M A': 74,
            'P 4': 75,
            'P 41': 76,
            'P 42': 77,
            'P 43': 78,
            'I 4': 79,
            'I 41': 80,
            'P -4': 81,
            'I -4': 82,
            'P 4/M': 83,
            'P 42/M': 84,
            'P 4/N': 85,
            'P 42/N': 86,
            'I 4/M': 87,
            'I 41/A': 88,
            'P 4 2 2': 89,
            'P 4 21 2': 90,
            'P 41 2 2': 91,
            'P 41 21 2': 92,
            'P 42 2 2': 93,
            'P 42 21 2': 94,
            'P 43 2 2': 95,
            'P 43 21 2': 96,
            'I 4 2 2': 97,
            'I 41 2 2': 98,
            'P 4 M M': 99,
            'P 4 B M': 100,
            'P 42 C M': 101,
            'P 42 N M': 102,
            'P 4 C C': 103,
            'P 4 N C': 104,
            'P 42 M C': 105,
            'P 42 B C': 106,
            'I 4 M M': 107,
            'I 4 C M': 108,
            'I 41 M D': 109,
            'I 41 C D': 110,
            'P -4 2 M': 111,
            'P -4 2 C': 112,
            'P -4 21 M': 113,
            'P -4 21 C': 114,
            'P -4 M 2': 115,
            'P -4 C 2': 116,
            'P -4 B 2': 117,
            'P -4 N 2': 118,
            'I -4 M 2': 119,
            'I -4 C 2': 120,
            'I -4 2 M': 121,
            'I -4 2 D': 122,
            'P 4/M M M': 123,
            'P 4/M C C': 124,
            'P 4/N B M': 125,
            'P 4/N N C': 126,
            'P 4/M B M': 127,
            'P 4/M N C': 128,
            'P 4/N M M': 129,
            'P 4/N C C': 130,
            'P 42/M M C': 131,
            'P 42/M C M': 132,
            'P 42/N B C': 133,
            'P 42/N N M': 134,
            'P 42/M B C': 135,
            'P 42/M N M': 136,
            'P 42/N M C': 137,
            'P 42/N C M': 138,
            'I 4/M M M': 139,
            'I 4/M C M': 140,
            'I 41/A M D': 141,
            'I 41/A C D': 142,
            'P 3': 143,
            'P 31': 144,
            'P 32': 145,
            'R 3': 146,
            'P -3': 147,
            'R -3': 148,
            'P 3 1 2': 149,
            'P 3 2 1': 150,
            'P 31 1 2': 151,
            'P 31 2 1': 152,
            'P 32 1 2': 153,
            'P 32 2 1': 154,
            'R 3 2': 155,
            'P 3 M 1': 156,
            'P 3 1 M': 157,
            'P 3 C 1': 158,
            'P 3 1 C': 159,
            'R 3 M': 160,
            'R 3 C': 161,
            'P -3 1 M': 162,
            'P -3 1 C': 163,
            'P -3 M 1': 164,
            'P -3 C 1': 165,
            'R -3 M': 166,
            'R -3 C': 167,
            'P 6': 168,
            'P 61': 169,
            'P 65': 170,
            'P 62': 171,
            'P 64': 172,
            'P 63': 173,
            'P -6': 174,
            'P 6/M': 175,
            'P 63/M': 176,
            'P 6 2 2': 177,
            'P 61 2 2': 178,
            'P 65 2 2': 179,
            'P 62 2 2': 180,
            'P 64 2 2': 181,
            'P 63 2 2': 182,
            'P 6 M M': 183,
            'P 6 C C': 184,
            'P 63 C M': 185,
            'P 63 M C': 186,
            'P -6 M 2': 187,
            'P -6 C 2': 188,
            'P -6 2 M': 189,
            'P -6 2 C': 190,
            'P 6/M M M': 191,
            'P 6/M C C': 192,
            'P 63/M C M': 193,
            'P 63/M M C': 194,
            'P 2 3': 195,
            'F 2 3': 196,
            'I 2 3': 197,
            'P 21 3': 198,
            'I 21 3': 199,
            'P M 3': 200,
            'P M -3': 200,
            'P N 3': 201,
            'P N -3': 201,
            'F M 3': 202,
            'F M -3': 202,
            'F D 3': 203,
            'F D -3': 203,
            'I M 3': 204,
            'I M -3': 204,
            'P A 3': 205,
            'P A -3': 205,
            'I A 3': 206,
            'I A -3': 206,
            'P 4 3 2': 207,
            'P 42 3 2': 208,
            'F 4 3 2': 209,
            'F 41 3 2': 210,
            'I 4 3 2': 211,
            'P 43 3 2': 212,
            'P 41 3 2': 213,
            'I 41 3 2': 214,
            'P -4 3 M': 215,
            'F -4 3 M': 216,
            'I -4 3 M': 217,
            'P -4 3 N': 218,
            'F -4 3 C': 219,
            'I -4 3 D': 220,
            'P M 3 M': 221,
            'P N 3 N': 222,
            'P M 3 N': 223,
            'P N 3 M': 224,
            'F M 3 M': 225,
            'F M 3 C': 226,
            'F D 3 M': 227,
            'F D 3 C': 228,
            'I M 3 M': 229,
            'I A 3 D': 230,
            'C 1': 231,
            'C -1': 232,
        }

    def write_to_archive(self) -> None:
        self.mainfile_parser.logger = self.logger
        self.mainfile_parser.mainfile = self.mainfile
        self.maindir = os.path.dirname(self.mainfile)

        sec_run = Run()
        self.archive.run.append(sec_run)
        # general run parameters
        header = self.mainfile_parser.get('header', {})
        sec_run.program = Program(version=header.get('program_version'))
        sec_run.x_gulp_title = header.get('title')
        for key, val in self.mainfile_parser.items():
            if key.startswith('x_gulp_'):
                setattr(sec_run, key, val)
        if self.mainfile_parser.date_start is not None:
            sec_run.time_run = TimeRun(
                date_start=datetime.strptime(
                    self.mainfile_parser.date_start, '%H:%M.%S %d %B %Y'
                ).timestamp()
            )
            if self.mainfile_parser.date_end is not None:
                sec_run.time_run.date_end = datetime.strptime(
                    self.mainfile_parser.date_end, '%H:%M.%S %d %B %Y'
                ).timestamp()

        input_info = self.mainfile_parser.get('input_information', {})
        sec_method = Method()
        sec_run.method.append(sec_method)
        # add force field interaction models
        force_field = ForceField()
        sec_method.force_field = force_field
        # for the new format, all potentials are in potentials while in old, needs to add
        # pair, three- and four-body interactions
        for potential_type in [
            'interatomic_potential',
            'pair_potential',
            'three_body_potential',
            'four_body_potential',
        ]:
            for potential in input_info.get(potential_type, []):
                sec_model = Model()
                force_field.model.append(sec_model)
                for interaction in potential.get('interaction', []):
                    sec_model.contributions.append(
                        Interaction(
                            functional_form=interaction.functional_form,
                            atom_labels=[[v[0] for v in interaction.get('atom_type')]],
                            parameters={
                                key: float(val) if isinstance(val, np.float64) else val
                                for key, val in interaction.get('key_parameter', [])
                            },
                        )
                    )

        if (pgfnff := input_info.get('pgfnff')) is not None:
            sec_model = Model()
            force_field.model.append(sec_model)
            sec_model.name = 'pGFNFF'
            sec_model.contributions.append(
                Interaction(
                    parameters={
                        key: float(val) for key, val in pgfnff.get('key_parameter', [])
                    }
                )
            )
        # atom parameters
        for n in range(len(input_info.get('species', {}).get('label', []))):
            sec_atom_parameter = AtomParameters()
            sec_method.atom_parameters.append(sec_atom_parameter)
            for key, val in input_info.species.items():
                setattr(sec_atom_parameter, key, val[n])

        input_config = self.mainfile_parser.get('input_configuration', {})

        def parse_system(source):
            # read from input configuration if section does not have structure information
            source = (
                self.mainfile_parser.input_configuration
                if source.coordinates is None
                else source
            )
            if source.coordinates is None:
                return

            sec_system = System()
            sec_run.system.append(sec_system)
            positions = []
            labels = []
            for atom in source.coordinates.get('atom', []):
                # include only core atoms
                if atom[1].lower().startswith('c'):
                    positions.append(atom[2:5])
                    labels.append(atom[0])
            lattice_vectors = source.get(
                'lattice_vectors', input_config.lattice_vectors
            )
            unit = source.coordinates.get('unit', '')
            if unit.lower().startswith('frac') and lattice_vectors is not None:
                positions = np.dot(positions, lattice_vectors)
            # get periodicity
            periodic = np.array([False] * 3)
            periodic[0 : input_config.get('x_gulp_pbc', 3)] = True
            # build the basis atoms
            atoms = ase_Atoms(symbols=labels, cell=lattice_vectors, positions=positions)
            # build the full crystal
            space_group = self._sg_map.get(input_config.x_gulp_space_group)
            # cellpar from source or primitive cellpar or from input_config
            cellpar = source.get(
                'cell_parameters',
                source.get('cell_parameters_primitive', input_config.cell_parameters),
            )
            if space_group is not None and cellpar is not None:
                atoms = ase_crystal(
                    atoms,
                    spacegroup=space_group,
                    pbc=periodic,
                    primitive_cell=True,
                    cellpar=cellpar,
                    onduplicates='replace',
                )
            # TODO take fractional occupancies into consideration
            sec_system.atoms = Atoms(
                positions=atoms.get_positions() * ureg.angstrom,
                labels=atoms.get_chemical_symbols(),
                lattice_vectors=atoms.get_cell().array * ureg.angstrom,
                periodic=periodic,
            )
            return sec_system

        def parse_mechanical_property(name, source, target):
            values = source.get(name, [None] * 3)
            types = (
                ['x', 'y', 'z']
                if 'youngs_modulus' in name
                else ['reuss', 'voigt', 'hill']
            )
            for n, type_n in enumerate(types):
                setattr(target, f'{name}_{type_n}', values[n])

        def parse_calculation(source):
            sec_calc = Calculation()
            sec_run.calculation.append(sec_calc)

            if source.energy_components is not None:
                sec_energy = Energy()
                sec_calc.energy = sec_energy
                for key, val in source.energy_components.get('key_val', []):
                    name = self._metainfo_map.get(key)
                    if name is None:
                        continue
                    val = (
                        val * ureg.eV
                        if name.startswith('x_gulp_')
                        else EnergyEntry(value=val * ureg.eV)
                    )
                    setattr(sec_energy, name, val)
                # assign primitive unit cell energy to energy total
                sec_energy.total = EnergyEntry(
                    value=sec_energy.x_gulp_primitive_unit_cell
                )

            if source.bulk_optimisation is not None:
                sec_opt = x_gulp_bulk_optimisation()
                sec_calc.x_gulp_bulk_optimisation = sec_opt
                for cycle in source.bulk_optimisation.get('cycle', []):
                    sec_opt.x_gulp_bulk_optimisation_cycle.append(
                        x_gulp_bulk_optimisation_cycle(
                            x_gulp_energy=cycle[0] * ureg.eV,
                            x_gulp_gnorm=cycle[1],
                            x_gulp_cpu_time=cycle[2],
                        )
                    )
                for key, val in source.bulk_optimisation.items():
                    if key.startswith('x_gulp_'):
                        setattr(sec_opt, key, val)

            if source.elastic_constants is not None:
                workflow = Elastic(method=ElasticMethod(), results=ElasticResults())
                workflow.method.energy_stress_calculator = 'gulp'
                workflow.results.elastic_constants_matrix_second_order = (
                    source.elastic_constants
                )
                workflow.results.compliance_matrix_second_order = (
                    source.elastic_compliance
                )
                mechanical_properties = source.get('mechanical_properties', {})
                parse_mechanical_property(
                    'bulk_modulus', mechanical_properties, workflow.results
                )
                parse_mechanical_property(
                    'shear_modulus', mechanical_properties, workflow.results
                )
                parse_mechanical_property(
                    'x_gulp_velocity_s_wave', mechanical_properties, workflow.results
                )
                parse_mechanical_property(
                    'x_gulp_velocity_p_wave', mechanical_properties, workflow.results
                )
                parse_mechanical_property(
                    'x_gulp_youngs_modulus', mechanical_properties, workflow.results
                )
                workflow.results.x_gulp_compressibility = mechanical_properties.get(
                    'compressibility'
                )
                poissons = mechanical_properties.get('poissons_ratio', np.zeros((3, 3)))
                # insert zeros for diagonal elements
                for n in range(3):
                    poissons[n] = np.insert(poissons[n], n, 0.0)
                workflow.results.x_gulp_poissons_ratio = poissons
                self.archive.workflow2 = workflow

            # md properties
            if source.energy_total is not None:
                sec_calc.energy = Energy(
                    total=EnergyEntry(
                        value=source.energy_total[0],
                        kinetic=source.energy_kinetic[0],
                        potential=source.energy_potential[0],
                    )
                )
                sec_calc.energy.x_gulp_total_averaged = EnergyEntry(
                    value=source.energy_total[1],
                    kinetic=source.energy_kinetic[1],
                    potential=source.energy_potential[1],
                )
            sec_calc.time = source.time
            sec_calc.temperature = source.get('temperature', [None, None])[0]
            sec_calc.pressure = source.get('pressure', [None, None])[0]
            sec_calc.x_gulp_temperature_averaged = source.get(
                'temperature', [None, None]
            )[1]
            sec_calc.x_gulp_pressure_averaged = source.get('pressure', [None, None])[1]

            # other properties
            for key, val in source.items():
                if key.startswith('x_gulp_'):
                    if key.endswith('refractive_indices') and val is not None:
                        val = val.value
                    setattr(sec_calc, key, val)

            return sec_calc

        for output in self.mainfile_parser.get('single_point', []):
            for calculation in output.get('calculation', []):
                sec_system = parse_system(calculation)
                sec_calc = parse_calculation(calculation)
                sec_calc.system_ref = sec_system

        for output in self.mainfile_parser.get('defect', []):
            for calculation in output.get('calculation', []):
                sec_system = parse_system(calculation)
                sec_calc = parse_calculation(calculation)
                sec_calc.system_ref = sec_system

        for output in self.mainfile_parser.get('molecular_dynamics', []):
            workflow = MolecularDynamics(method=MolecularDynamicsMethod())
            workflow.method.thermodynamic_ensemble = output.get(
                'ensemble_type', ''
            ).upper()
            workflow.method.integration_timestep = output.timestep
            for key, val in output.items():
                if key.startswith('x_gulp_'):
                    setattr(workflow, key, val)
            # parse md steps
            for step in output.get('step', []):
                parse_calculation(step)
                # TODO where are the trajectory data saved
            self.archive.workflow2 = workflow
