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

from nomad.units import ureg
from nomad.parsing.file_parser import TextParser, Quantity, DataTextParser
from runschema.run import Run, Program
from runschema.method import Method, TB, xTB, ForceField, Model, Interaction
from runschema.system import System, Atoms
from runschema.calculation import (
    Calculation,
    Energy,
    EnergyEntry,
    Forces,
    ForcesEntry,
    Stress,
    StressEntry,
    Charges,
    ChargesValue,
)
from simulationworkflowschema import (
    GeometryOptimization,
    GeometryOptimizationMethod,
    SinglePoint,
)
from atomisticparsers.utils import MDParser
from atomisticparsers.bopfox.metainfo.bopfox import (
    x_bopfox_onsite_levels,
    x_bopfox_onsite_levels_value,
)


re_f = r'[-+]?\d*\.\d*(?:[Ee][-+]\d+)?'
re_n = r'[\n\r]'


class ModelsbxParser(TextParser):
    def init_quantities(self):
        def to_parameters(val_in):
            parameters = dict()
            for val in val_in.strip().splitlines():
                if val.startswith('!'):
                    continue
                val = val.split('=')
                if len(val) == 2:
                    val[1] = val[1].split()
                    parameters[val[0].strip().lower()] = (
                        val[1][0] if len(val[1]) == 1 else val[1]
                    )
            return parameters

        self._quantities = [
            Quantity(
                'model',
                rf'(el *= *\w+ *{re_n}[\s\S]+?)(?:{re_n} *mod|\Z)',
                repeats=True,
                sub_parser=TextParser(
                    quantities=[
                        Quantity('name', r'el *= *(\S+)', dtype=str),
                        Quantity(
                            'parameters',
                            r'((?:[\w!]+ *= *\w+\s+)+)',
                            str_operation=to_parameters,
                        ),
                        Quantity(
                            'atom',
                            r'([aA]tom *= *[A-Z]\S*[\s\S]+?(?:[\w\!]+ *= *[\w\. \-\+]+\s+)+)',
                            repeats=True,
                            str_operation=to_parameters,
                        ),
                        Quantity(
                            'bond',
                            r'([bB]ond *= *[A-Z]\S* +[A-Z]\S*[\s\S]+?(?:[\w\!]+ *= *[\w\. \-\+]+\s+)+)',
                            repeats=True,
                            str_operation=to_parameters,
                        ),
                    ]
                ),
            )
        ]


class StrucbxParser(TextParser):
    def init_quantities(self):
        def to_magnetisation(val_in):
            val = val_in.strip().splitlines()
            magnetisation = val[0].lower().startswith('t')
            values = np.array(
                [v.strip().split() for v in val[1:]], dtype=np.dtype(np.float64)
            )
            return magnetisation, values

        self._quantities = [
            Quantity(
                'lattice_constant', rf'[aA][lL][aA][tT] *= *({re_f})', dtype=np.float64
            ),
            Quantity(
                'lattice_vectors',
                rf'[aA]\d+ *= *({re_f} +{re_f} +{re_f})',
                repeats=True,
                dtype=np.dtype(np.float64),
            ),
            Quantity('coordinate_type', r'[cC][oO][oO][rR][dD] *= *(\S+)', dtype=str),
            Quantity(
                'label_position',
                rf'{re_n} *([A-Z][a-z]* +{re_f} +{re_f} +{re_f})',
                repeats=True,
            ),
            Quantity(
                'magnetisation',
                rf'[mM][aA][gG][nN][eE][tT][iI][sS][aA][tT][iI][oO][nN] *= *(\S+[\s\d\.]+)',
                str_operation=to_magnetisation,
            ),
        ]

    # def get_positions(self):
    #     positions = np.array([v[1:4] for v in self.get('label_position', [])])
    #     if self.get('coordinate_type').lower().startswith('d'):
    #         # positions are scaled by lattice vectors
    #         if self.lattice_vectors is not None:
    #             positions = np.dot(positions, self.lattice_vectors)
    #     return positions


# we do not use MDA for xyz file in order to read stress and forces data which are not
# necessarily printed in mainfile
class XYZParser(TextParser):
    def init_quantities(self):
        def to_frame(val_in):
            val = val_in.strip().splitlines()
            n_atoms = int(val[0])
            md = 'fs' in val[1]
            step = float(val[1].split()[0])
            labels = []
            positions = np.zeros((n_atoms, 3))
            constraints = []
            energies = np.zeros(n_atoms)
            forces = np.zeros((n_atoms, 3))
            stresses = np.zeros((n_atoms, 6))
            for n, line in enumerate(val[2 : 2 + n_atoms]):
                line = line.split()
                labels.append(line[0])
                positions[n] = line[1:4]
                constraints.append(list(line[4]))
                if md:
                    forces[n] = line[5:8]
                else:
                    energies[n] = line[5]
                    forces[n] = line[6:9]
                    stresses[n] = line[9:15]
            return dict(
                n_atoms=n_atoms,
                step=step,
                labels=labels,
                positions=positions,
                constraints=constraints,
                energies_total=energies,
                forces_total=forces,
                stresses_total=stresses,
            )

        self._quantities = [
            Quantity(
                'frame',
                rf'(\d+\s+\d+.*?\s+(?:[A-Z][a-z]* +{re_f} +{re_f} +{re_f}.+\s+)+)',
                repeats=True,
                str_operation=to_frame,
            )
        ]


class InfoxParser(TextParser):
    def init_quantities(self):
        self._quantities = [
            Quantity(
                'parameter',
                r'(\w+ *= *.+)',
                repeats=True,
                str_operation=lambda x: [v.strip() for v in x.split('=')],
            )
        ]

    def get_parameters(self):
        return {v[0].lower(): v[1] for v in self.get('parameter', [])}


class MainfileParser(TextParser):
    def init_quantities(self):
        calc_quantities = [
            Quantity(
                'energy',
                r'(?:Contributions to the Energy|Energies)\s+\=+([\s\S]+?)\={50}',
                sub_parser=TextParser(
                    quantities=[
                        Quantity(
                            'contribution',
                            r'(U_\w+.+?\( *atom = +1 *\)[\s\S]+?U_\w+/atom.+)',
                            repeats=True,
                            sub_parser=TextParser(
                                quantities=[
                                    Quantity('type', r'U_(\w+)', dtype=str),
                                    Quantity(
                                        'atomic',
                                        rf'atom += +\d+ \).+?({re_f})',
                                        repeats=True,
                                        dtype=np.float64,
                                    ),
                                    Quantity(
                                        'total',
                                        rf'U_\w+/atom.+?({re_f})',
                                        dtype=np.float64,
                                    ),
                                ]
                            ),
                        )
                    ]
                ),
            ),
            Quantity(
                'forces',
                r'(?:Contributions to the Forces|Forces \(Fx,Fy,Fz,x,y,z\))\s+\=+([\s\S]+?)\={50}',
                sub_parser=TextParser(
                    quantities=[
                        Quantity(
                            'contribution',
                            r'(FBOP \(\w+ *\) +1 +[\s\S]+?)\-{50}',
                            repeats=True,
                            sub_parser=TextParser(
                                quantities=[
                                    Quantity('type', r'FBOP \((\w+)', dtype=str),
                                    Quantity(
                                        'atomic',
                                        rf'\) +\d+ +({re_f} +{re_f} +{re_f})',
                                        repeats=True,
                                        dtype=np.dtype(np.float64),
                                    ),
                                ]
                            ),
                        )
                    ]
                ),
            ),
            Quantity(
                'stress',
                r'(?:Contributions for the stresses|stresses \(11,22,33,23,13,12\))\s+\=+([\s\S]+?)\={50}',
                sub_parser=TextParser(
                    quantities=[
                        Quantity(
                            'total',
                            rf'sum\(stress\)/volume +({re_f} +{re_f} +{re_f} +{re_f} +{re_f} +{re_f})',
                            dtype=np.dtype(np.float64),
                        ),
                        Quantity(
                            'contribution',
                            r'(stress \(\w+ *\) +1 +[\s\S]+?)\-{50}',
                            repeats=True,
                            sub_parser=TextParser(
                                quantities=[
                                    Quantity('type', r'stress \((\w+)', dtype=str),
                                    Quantity(
                                        'atomic',
                                        rf'\) +\d+ +({re_f} +{re_f} +{re_f} +{re_f} +{re_f} +{re_f})',
                                        repeats=True,
                                        dtype=np.dtype(np.float64),
                                    ),
                                ]
                            ),
                        ),
                    ]
                ),
            ),
            Quantity('energy_fermi', rf'E_Fermi +.+?({re_f})', dtype=np.float64),
            Quantity(
                'charges',
                r'(?:Charge terms|Charges)\s+\=+([\s\S]+?)\={50}',
                sub_parser=TextParser(
                    quantities=[
                        Quantity(
                            'n_electrons',
                            rf'Nelec \( atom = +\d+ \).+?({re_f})',
                            dtype=np.float64,
                            repeats=True,
                        ),
                        Quantity(
                            'charge',
                            rf'Charge \( atom = +\d+ \).+?({re_f})',
                            dtype=np.float64,
                            repeats=True,
                        ),
                    ]
                ),
            ),
            Quantity(
                'magnetic_moments',
                r'Magnetic moments\s+\=+([\s\S]+?)\={50}',
                sub_parser=TextParser(
                    quantities=[
                        Quantity(
                            'mag_mom',
                            rf'Mag_mom \( atom = +(\d+), orbital = +((?:s|p|d)) \) +({re_f}) +({re_f}) +({re_f})',
                            repeats=True,
                        )
                    ]
                ),
            ),
            Quantity(
                'onsite_levels',
                r'Onsite levels\s+\=+([\s\S]+?)\={50}',
                sub_parser=TextParser(
                    quantities=[
                        Quantity(
                            'energy',
                            rf'E((?:s|p|d)) \( atom = +(\d+), spin = \d+ \) +({re_f})',
                            repeats=True,
                        )
                    ]
                ),
            ),
        ]

        self._quantities = [
            Quantity(
                'program_version',
                r'BOPfox \(v (\S+)\) (rev\. \d+)',
                dtype=str,
                flatten=False,
            ),
            Quantity(
                'simulation',
                r'(\w+ +\: +\S+ +\( *\S+ *\)\s+[\s\S]+?)init\: N\(',
                sub_parser=TextParser(
                    quantities=[
                        Quantity(
                            'parameter',
                            r'(\w+) +\: +(\S+) +\( *(\S+) *\)\s+',
                            repeats=True,
                        )
                    ]
                ),
            ),
            Quantity(
                'lattice_vectors',
                rf'cell\(\:,\d+\) \: +({re_f}) +({re_f}) +({re_f})',
                repeats=True,
                dtype=np.dtype(np.float64),
            ),
            Quantity(
                'label_position',
                rf'init\: atom/type/pos/fix\: +\d+ +([A-Z]\S*) +({re_f}) +({re_f}) +({re_f}) +([ FT]+)',
                repeats=True,
            ),
            Quantity(
                'n_atoms',
                r'Atoms in cell/cluster\: +(\d+) +(\d+)',
                dtype=np.dtype(np.int32),
            ),
            Quantity(
                'relaxation',
                r'(relax\: [\s\S]+?(?:relax\: cycle finished|\Z))',
                sub_parser=TextParser(
                    quantities=[
                        Quantity(
                            'cycle',
                            rf'( \d+ +{re_f} +{re_f} +\d+ *{re_n}[\s\S]+?)(?:relax\:  |\Z)',
                            repeats=True,
                            sub_parser=TextParser(quantities=calc_quantities),
                        )
                    ]
                ),
            ),
            Quantity('md_column_names', r'col \d+\: (.+?) +\[', repeats=True),
        ] + calc_quantities

    def get_simulation_parameters(self):
        return {
            v[0]: (v[2] if v[2] != 'none' else None) if v[1] == '--' else v[1]
            for v in self.get('simulation', {}).get('parameter', [])
        }


class BOPfoxParser(MDParser):
    def __init__(self):
        self.mainfile_parser = MainfileParser()
        self.strucbx_parser = StrucbxParser()
        self.xyz_parser = XYZParser()
        self.modelsbx_parser = ModelsbxParser()
        self.dat_parser = DataTextParser()
        self.infox_parser = InfoxParser()
        self._metainfo_map = {
            'binding': 'total',
            'coulomb': 'electrostatic',
            'ionic': 'nuclear_repulsion',
            'total': 'total',
        }
        super().__init__()

    def write_to_archive(self) -> None:
        self.mainfile_parser.logger = self.logger
        self.mainfile_parser.mainfile = self.mainfile
        self.strucbx_parser.logger = self.logger
        self.xyz_parser.logger = self.logger
        self.modelsbx_parser.logger = self.logger
        self.dat_parser.logger = self.logger
        self.infox_parser.logger = self.logger
        self.maindir = os.path.dirname(self.mainfile)

        sec_run = Run()
        self.archive.run.append(sec_run)
        sec_run.program = Program(
            name='BOPfox', version=self.mainfile_parser.get('program_version')
        )

        sec_method = Method()
        sec_run.method.append(sec_method)
        parameters = self.mainfile_parser.get_simulation_parameters()
        sec_method.x_bopfox_simulation_parameters = parameters
        # force field parameters
        self.modelsbx_parser.mainfile = os.path.join(
            self.maindir, parameters.get('modelfile', 'models.bx')
        )
        for model in self.modelsbx_parser.get('model', []):
            # pick out only the model indicated in parameters
            if model.name == parameters.get('model'):
                # bop uses a tight-binding model
                tb = model.parameters.get('version', 'bop').lower() in [
                    'bop',
                    'tight-binding',
                ]
                if tb:
                    sec_model = xTB()
                    sec_method.tb = TB(xtb=sec_model)
                else:
                    sec_model = Model()
                    sec_method.force_field = ForceField(model=[sec_model])

                sec_model.name = model.name
                sec_model.x_bopfox_parameters = model.parameters
                # interaction between each bond pair
                for bond in model.get('bond', []):
                    # functional terms for each contribution
                    for key, val in bond.items():
                        key = key.lower()
                        if tb:
                            if (
                                key.endswith('sigma')
                                or key.endswith('pi')
                                or key.endswith('delta')
                            ):
                                sec_model.hamiltonian.append(
                                    Interaction(
                                        name=key,
                                        functional_form=val[0],
                                        parameters=val[1:],
                                        atom_labels=[bond.get('bond')],
                                        x_bopfox_valence=bond.get('valence'),
                                        x_bopfox_cutoff=bond.get('rcut'),
                                        x_bopfox_dcutoff=bond.get('dcut'),
                                        x_bopfox_chargetransfer=bond.get(
                                            'chargetransfer'
                                        ),
                                    )
                                )
                            if key.endswith('overlap'):
                                sec_model.overlap.append(
                                    Interaction(
                                        name=key,
                                        functional_form=val[0],
                                        parameters=val[1:],
                                        atom_labels=bond.get('bond'),
                                        x_bopfox_valence=bond.get('valence'),
                                        x_bopfox_cutoff=bond.get('rcut'),
                                        x_bopfox_dcutoff=bond.get('dcut'),
                                    )
                                )
                            elif key.startswith('rep'):
                                sec_model.repulsion.append(
                                    Interaction(
                                        name=key,
                                        functional_form=val[0],
                                        parameters=val[1:],
                                        atom_labels=[bond.get('bond')],
                                        x_bopfox_cutoff=bond.get('r2cut'),
                                        x_bopfox_dcutoff=bond.get('d2cut'),
                                    )
                                )
                        else:
                            if key.startswith('rep'):
                                sec_model.contributions.append(
                                    Interaction(
                                        name=key,
                                        functional_form=val[0],
                                        parameters=val[1:],
                                        atom_labels=[bond.get('bond')],
                                        x_bopfox_cutoff=bond.get('r2cut'),
                                        x_bopfox_dcutoff=bond.get('d2cut'),
                                    )
                                )

        def parse_system(source, target=None):
            if source is None:
                return

            label_position = source.get('label_position')
            lattice_vectors = source.get('lattice_vectors')
            if lattice_vectors is not None:
                lattice_vectors = np.array(lattice_vectors) * source.get(
                    'lattice_constant', 1.0
                )
            if label_position is not None:
                labels = [v[0] for v in label_position]
                positions = np.array([v[1:4] for v in label_position])
                if source.get('coordinate_type', '').lower().startswith('d'):
                    # positions are scaled by lattice vectors
                    if lattice_vectors is not None:
                        positions = np.dot(positions, lattice_vectors)
            else:
                labels = source.get('labels')
                positions = source.get('positions')

            if positions is None:
                return

            sec_system = System()
            sec_run.system.append(sec_system) if target is None else target
            sec_system.atoms = Atoms(labels=labels, positions=positions * ureg.angstrom)
            if lattice_vectors is not None:
                sec_system.atoms.lattice_vectors = lattice_vectors * ureg.angstrom

            return sec_system

        def parse_calculation(source, target=None):
            sec_calc = Calculation()
            sec_run.calculation.append(sec_calc) if target is None else target

            # energy
            n_atoms = self.mainfile_parser.get('n_atoms', [1, 1])[0]
            if source.get('energy') is not None:
                sec_energy = Energy()
                sec_calc.energy = sec_energy
                for contribution in source.energy.get('contribution', []):
                    name = self._metainfo_map.get(contribution.type)
                    energy_entry = EnergyEntry(
                        value=contribution.total * ureg.eV * n_atoms,
                        values_per_atom=contribution.atomic * ureg.eV,
                    )
                    if name is None:
                        energy_entry.kind = contribution.type
                        sec_energy.contributions.append(energy_entry)
                    else:
                        setattr(sec_energy, name, energy_entry)
                if source.energy_fermi is not None:
                    sec_energy.fermi = source.energy_fermi * ureg.eV

            # forces
            if source.get('forces') is not None:
                sec_forces = Forces()
                sec_calc.forces = sec_forces
                for contribution in source.forces.get('contribution', []):
                    name = self._metainfo_map.get(contribution.type)
                    forces_entry = ForcesEntry(
                        value=contribution.atomic * ureg.eV / ureg.angstrom
                    )
                    if name is None:
                        forces_entry.kind = contribution.type
                        sec_forces.contributions.append(forces_entry)
                    else:
                        setattr(sec_forces, name, forces_entry)

            def symmetrize(stress):
                symmetrized = np.zeros((3, 3))
                symmetrized[0][0] = stress[0]
                symmetrized[1][1] = stress[1]
                symmetrized[2][2] = stress[2]
                symmetrized[1][2] = symmetrized[2][1] = stress[3]
                symmetrized[0][2] = symmetrized[2][0] = stress[4]
                symmetrized[0][1] = symmetrized[1][0] = stress[5]
                return symmetrized

            # stress
            if source.get('stress') is not None:
                sec_stress = Stress()
                sec_calc.stress = sec_stress
                for contribution in source.stress.get('contribution', []):
                    name = self._metainfo_map.get(contribution.type)
                    stress_entry = StressEntry(
                        values_per_atom=[
                            symmetrize(atomic) for atomic in contribution.atomic
                        ]
                        * ureg.eV
                        / ureg.angstrom**3
                    )
                    if name is None:
                        stress_entry.kind = contribution.type
                        sec_stress.contributions.append(stress_entry)
                    else:
                        if name == 'total':
                            stress_entry.value = (
                                symmetrize(source.stress.total)
                                * ureg.eV
                                / ureg.angstrom**3
                            )
                        setattr(sec_stress, name, stress_entry)

            # charges
            if source.get('charges') is not None:
                sec_charges = Charges()
                sec_calc.charges.append(sec_charges)
                sec_charges.n_electrons = source.charges.n_electrons
                sec_charges.value = source.charges.charge * ureg.elementary_charge
                # magnetic moments
                if source.magnetic_moments is not None:
                    for mag_mom in source.magnetic_moments.get('mag_mom', []):
                        sec_charges.orbital_projected.append(
                            ChargesValue(
                                atom_index=mag_mom[0] - 1,
                                orbital=mag_mom[1],
                                spin_z=mag_mom[2],
                            )
                        )

            # onsite levels
            if source.get('onsite_levels') is not None:
                sec_onsite = x_bopfox_onsite_levels()
                sec_calc.x_bopfox_onsite_levels.append(sec_onsite)
                for onsite in source.onsite_levels.get('energy', []):
                    sec_onsite.orbital_projected.append(
                        x_bopfox_onsite_levels_value(
                            orbital=onsite[0], atom_index=onsite[1] - 1, value=onsite[2]
                        )
                    )

            # energies and forces from trajectory file
            if source.get('energies_total') is not None:
                sec_calc.energy = Energy(
                    total=EnergyEntry(
                        values_per_atom=source.get('energies_total') * ureg.eV,
                        value=sum(source.get('energies_total')) * ureg.eV,
                    )
                )
            if source.get('forces_total') is not None:
                sec_calc.forces = Forces(
                    total=ForcesEntry(
                        value=source.get('forces_total') * ureg.eV / ureg.angstrom
                    )
                )

            # total energy from struc.log.dat
            if source.get('energy_total') is not None:
                sec_calc.energy = Energy(
                    total=EnergyEntry(value=source.get('energy_total') * ureg.eV)
                )

            return sec_calc

        task = parameters.get('task')
        # read the strucfile from infox because string may be truncated in mainfile
        self.infox_parser.mainfile = os.path.join(self.maindir, 'infox.bx')
        struc_basename = (
            self.infox_parser.get_parameters().get('strucfile', '').rstrip('.bx')
        )

        # initial structure
        self.strucbx_parser.mainfile = os.path.join(
            self.maindir, f'{struc_basename}.bx'
        )
        sec_system = parse_system(self.strucbx_parser)

        # initial single point calculation
        sec_calc = parse_calculation(self.mainfile_parser)
        sec_calc.system_ref = sec_system
        workflow = None
        if task in ['energy', 'force']:
            self.archive.workflow2 = SinglePoint()

        elif task == 'relax':
            # relaxation trajectory from struc.RX.xyz
            self.xyz_parser.mainfile = os.path.join(
                self.maindir, f'{struc_basename}.RX.xyz'
            )
            frames = {
                int(frame.get('step')): frame
                for frame in self.xyz_parser.get('frame', [])
            }
            self.dat_parser.mainfile = os.path.join(
                self.maindir, f'{struc_basename}.log.dat'
            )
            for n, cycle in enumerate(self.mainfile_parser.relaxation.get('cycle', [])):
                sec_calc = parse_calculation(cycle)
                frame = frames.get(n + 1, dict(energy_total=self.dat_parser.data[n][1]))
                if sec_calc.energy is None:
                    # if energy is not present in the case of non-verbose output, read energy
                    # from trajectory file or from struc.log.dat
                    sec_calc = parse_calculation(frame, sec_calc)

                # read frame from trajectory
                sec_system = parse_system(frame)
                sec_calc.system_ref = sec_system

            # read final structure from struc.final.bx
            self.strucbx_parser.mainfile = os.path.join(
                self.maindir, f'{struc_basename}.final.bx'
            )
            sec_calc.system_ref = parse_system(self.strucbx_parser, sec_calc.system_ref)

            workflow = GeometryOptimization(method=GeometryOptimizationMethod())
            workflow.method.convergence_tolerance_energy_difference = (
                parameters.get('rxeconv', 0) * ureg.eV
            )
            workflow.method.convergence_tolerance_force_maximum = (
                parameters.get('rxfconv', 0) * ureg.eV / ureg.angstrom
            )
            self.archive.workflow2 = workflow

        elif task == 'md':
            # md trajectory from struc.MD.xyz
            self.xyz_parser.mainfile = os.path.join(
                self.maindir, f'{struc_basename}.MD.xyz'
            )
            # thermodynamic properties from struc.erg.dat
            # TODO determine if the column names are arbitrary
            self.dat_parser.mainfile = os.path.join(
                self.maindir, f'{struc_basename}.erg.dat'
            )

            traj_steps = [
                int(frame.get('step', 0)) for frame in self.xyz_parser.get('frame', [])
            ]
            time_step = parameters.get('mdtimestep', 1.0)
            thermo_steps = [int(d[0] / time_step) for d in self.dat_parser.data]
            n_atoms = self.mainfile_parser.get('n_atoms', [1, 1])[0]

            self.n_atoms = n_atoms
            self.trajectory_steps = traj_steps
            self.thermodynamics_steps = thermo_steps

            if (lattice_vectors := self.strucbx_parser.get('lattice_vectors')) is None:
                lattice_vectors = lattice_vectors * ureg.angstrom
            for step in self.trajectory_steps:
                frame = self.xyz_parser.frame[traj_steps.index(step)]
                labels = frame.get('labels')
                positions = frame.get('positions')
                if positions is None or labels is None:
                    continue
                self.parse_trajectory_step(
                    dict(
                        atoms=dict(
                            labels=labels,
                            positions=positions * ureg.angstrom,
                            lattice_vectors=lattice_vectors,
                        )
                    )
                )

            for n, step in enumerate(self.thermodynamics_steps):
                # md does not do an initial single point calculation so override first calc
                data = self.dat_parser.data[n]
                thermo_data = dict(
                    time_physical=data[0] * ureg.fs,
                    time_step=step,
                    energy=dict(
                        total=dict(
                            value=data[1] * n_atoms * ureg.eV,
                            potential=data[2] * n_atoms * ureg.eV,
                            kinetic=data[3] * n_atoms * ureg.eV,
                        )
                    ),
                    temperature=data[5] * ureg.K,
                    pressure=data[7] * ureg.MPa,
                )
                if n == 0:
                    self.parse_section(thermo_data, self.archive.run[-1].calculation[0])
                else:
                    self.parse_thermodynamics_step(thermo_data)

            integrator_map = {'velocity-verlet': 'velocity_verlet'}
            ensemble = None
            if parameters.get('mdthermostat'):
                ensemble = 'NVT'
            elif parameters.get('mdbarostat'):
                ensemble = 'NVE'
            self.parse_md_workflow(
                dict(
                    method=dict(
                        integration_timestep=time_step * ureg.fs,
                        integrator_type=integrator_map.get(parameters.get('mdkernel')),
                        n_steps=parameters.get('mdsteps'),
                        thermodynamic_ensemble=ensemble,
                    )
                )
            )
