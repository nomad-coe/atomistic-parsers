# Atomistic parser mapping report

This report is the concatenation of the per-parser `MAPPING_REPORT.md` fragments. Do not edit by hand — edit the fragment and rebuild.

## AMBER

**Coverage:** Not available — this parser uses custom archive-writing logic and exposes no reportable file-parser quantities.

It reads data through `BasicParser` from `simulationparsers.utils`, configured with regex patterns for the text log (program version, total energy, atom positions/numbers) and auxiliary `.inpcrd`/`.prmtop` files, rather than declaring any `Quantity` file-parser class.

## ASAP

**Coverage:** Not available — this parser uses custom archive-writing logic and exposes no reportable file-parser quantities.

`AsapParser` subclasses `ASETrajParser` and reads data from ASE trajectory objects (`self.traj_parser.traj`), declaring no `Quantity(...)` file-parser classes of its own.

## ASE

**Coverage:** Not available — this parser uses custom archive-writing logic and exposes no reportable file-parser quantities.

The parser reads structures and trajectories through ASE's `ase.io.trajectory.Trajectory` reader (via the shared `ASETrajParser` base) and populates the archive programmatically rather than declaring `Quantity(...)`-based file-parser classes.

## BOPfox / ModelsbxParser

**Summary:** 3 mapped, 2 unmapped quantities (60.00% coverage).

| File-parser quantity | Status | Archive mapper source |
| --- | --- | --- |
| `model` | Mapped | `runschema.method.Method.tb.xtb.name`<br>`runschema.method.Method.force_field.model.name` |
| `model.name` | Mapped | `runschema.method.Method.tb.xtb.name`<br>`runschema.method.Method.force_field.model.name` |
| `model.parameters` | Unmapped | — |
| `model.atom` | Unmapped | — |
| `model.bond` | Mapped | `runschema.method.Interaction.name`<br>`runschema.method.Interaction.functional_form`<br>`runschema.method.Interaction.parameters`<br>`runschema.method.Interaction.atom_labels` |

## BOPfox / StrucbxParser

**Summary:** 4 mapped, 1 unmapped quantities (80.00% coverage).

| File-parser quantity | Status | Archive mapper source |
| --- | --- | --- |
| `lattice_constant` | Mapped | `runschema.system.Atoms.lattice_vectors` |
| `lattice_vectors` | Mapped | `runschema.system.Atoms.lattice_vectors` |
| `coordinate_type` | Mapped | `runschema.system.Atoms.positions` |
| `label_position` | Mapped | `runschema.system.Atoms.labels`<br>`runschema.system.Atoms.positions` |
| `magnetisation` | Unmapped | — |

## BOPfox / XYZParser

**Summary:** 1 mapped, 0 unmapped quantities (100.00% coverage).

| File-parser quantity | Status | Archive mapper source |
| --- | --- | --- |
| `frame` | Mapped | `runschema.system.Atoms.labels`<br>`runschema.system.Atoms.positions`<br>`runschema.calculation.Energy.total.value`<br>`runschema.calculation.Energy.total.values_per_atom`<br>`runschema.calculation.Forces.total.value` |

## BOPfox / InfoxParser

**Summary:** 0 mapped, 1 unmapped quantities (0.00% coverage).

| File-parser quantity | Status | Archive mapper source |
| --- | --- | --- |
| `parameter` | Unmapped | — |

## BOPfox / MainfileParser

**Summary:** 24 mapped, 7 unmapped quantities (77.42% coverage).

| File-parser quantity | Status | Archive mapper source |
| --- | --- | --- |
| `energy` | Mapped | `runschema.calculation.Calculation.energy` |
| `energy.contribution` | Mapped | `runschema.calculation.Energy.total`<br>`runschema.calculation.Energy.electrostatic`<br>`runschema.calculation.Energy.nuclear_repulsion`<br>`runschema.calculation.Energy.contributions` |
| `energy.contribution.type` | Mapped | `runschema.calculation.Energy.total`<br>`runschema.calculation.Energy.electrostatic`<br>`runschema.calculation.Energy.nuclear_repulsion`<br>`runschema.calculation.EnergyEntry.kind` |
| `energy.contribution.atomic` | Mapped | `runschema.calculation.EnergyEntry.values_per_atom` |
| `energy.contribution.total` | Mapped | `runschema.calculation.EnergyEntry.value` |
| `forces` | Mapped | `runschema.calculation.Calculation.forces` |
| `forces.contribution` | Mapped | `runschema.calculation.Forces.total`<br>`runschema.calculation.Forces.contributions` |
| `forces.contribution.type` | Mapped | `runschema.calculation.Forces.total`<br>`runschema.calculation.ForcesEntry.kind` |
| `forces.contribution.atomic` | Mapped | `runschema.calculation.ForcesEntry.value` |
| `stress` | Mapped | `runschema.calculation.Calculation.stress` |
| `stress.total` | Mapped | `runschema.calculation.StressEntry.value` |
| `stress.contribution` | Mapped | `runschema.calculation.Stress.total`<br>`runschema.calculation.Stress.contributions` |
| `stress.contribution.type` | Mapped | `runschema.calculation.Stress.total`<br>`runschema.calculation.StressEntry.kind` |
| `stress.contribution.atomic` | Mapped | `runschema.calculation.StressEntry.values_per_atom` |
| `energy_fermi` | Mapped | `runschema.calculation.Energy.fermi` |
| `charges` | Mapped | `runschema.calculation.Calculation.charges` |
| `charges.n_electrons` | Mapped | `runschema.calculation.Charges.n_electrons` |
| `charges.charge` | Mapped | `runschema.calculation.Charges.value` |
| `magnetic_moments` | Mapped | `runschema.calculation.Charges.orbital_projected` |
| `magnetic_moments.mag_mom` | Mapped | `runschema.calculation.ChargesValue.atom_index`<br>`runschema.calculation.ChargesValue.orbital`<br>`runschema.calculation.ChargesValue.spin_z` |
| `onsite_levels` | Unmapped | — |
| `onsite_levels.energy` | Unmapped | — |
| `program_version` | Mapped | `runschema.run.Program.version` |
| `simulation` | Unmapped | — |
| `simulation.parameter` | Unmapped | — |
| `lattice_vectors` | Unmapped | — |
| `label_position` | Unmapped | — |
| `n_atoms` | Mapped | `runschema.calculation.EnergyEntry.value` |
| `relaxation` | Mapped | `runschema.calculation.Run.calculation` |
| `relaxation.cycle` | Mapped | `runschema.calculation.Run.calculation` |
| `md_column_names` | Unmapped | — |

## DFTB+ / DetailedParser

**Summary:** 10 mapped, 11 unmapped quantities (47.62% coverage).

| File-parser quantity | Status | Archive mapper source |
| --- | --- | --- |
| `coordinates` | Unmapped | — |
| `charges` | Unmapped | — |
| `eigenvalues` | Unmapped | — |
| `occupations` | Unmapped | — |
| `eigenvalues_occupations` | Mapped | `runschema.calculation.Calculation.eigenvalues.energies`<br>`runschema.calculation.Calculation.eigenvalues.occupations` |
| `eigenvalues_occupations.kpoint` | Mapped | `runschema.calculation.Calculation.eigenvalues.energies`<br>`runschema.calculation.Calculation.eigenvalues.occupations` |
| `fermi_level` | Mapped | `runschema.calculation.Calculation.energy.fermi` |
| `energy_x_dftbp_band` | Unmapped | — |
| `energy_x_dftbp_ts` | Unmapped | — |
| `energy_x_dftbp_band_free` | Unmapped | — |
| `energy_x_dftbp_band_t0` | Unmapped | — |
| `energy_sum_eigenvalues` | Mapped | `runschema.calculation.Calculation.energy.sum_eigenvalues.value` |
| `energy_x_dftbp_scc` | Unmapped | — |
| `energy_electronic` | Mapped | `runschema.calculation.Calculation.energy.electronic.value` |
| `energy_nuclear_repulsion` | Mapped | `runschema.calculation.Calculation.energy.nuclear_repulsion.value` |
| `energy_x_dftbp_dispersion` | Unmapped | — |
| `energy_total` | Mapped | `runschema.calculation.Calculation.energy.total.value` |
| `energy_x_dftbp_total_mermin` | Unmapped | — |
| `pressure` | Mapped | `runschema.calculation.Calculation.pressure` |
| `forces` | Mapped | `runschema.calculation.Calculation.forces.total.value` |
| `dipole` | Mapped | `runschema.calculation.Calculation.multipoles.dipole.total` |

## DFTB+ / OutParser

**Summary:** 5 mapped, 10 unmapped quantities (33.33% coverage).

| File-parser quantity | Status | Archive mapper source |
| --- | --- | --- |
| `program_version` | Mapped | `runschema.run.Program.version` |
| `input_file` | Unmapped | — |
| `processed_input_file` | Unmapped | — |
| `parser_version` | Unmapped | — |
| `sk_files` | Unmapped | — |
| `input_parameters` | Unmapped | — |
| `input_parameters.key_val` | Unmapped | — |
| `input_parameters.kpoints_weights` | Unmapped | — |
| `step` | Unmapped | — |
| `step.scf` | Mapped | `runschema.calculation.Calculation.scf_iteration.energy.total.value`<br>`runschema.calculation.Calculation.scf_iteration.energy.change` |
| `step.energy_total` | Mapped | `runschema.calculation.Calculation.energy.total.value` |
| `step.energy_total_t0` | Mapped | `runschema.calculation.Calculation.energy.total_t0.value` |
| `step.energy_x_dftbp_total_mermin` | Unmapped | — |
| `step.pressure` | Mapped | `runschema.calculation.Calculation.pressure` |
| `step.maximum_force` | Unmapped | — |

## DL_POLY / TrajParser

**Summary:** 5 mapped, 1 unmapped quantities (83.33% coverage).

| File-parser quantity | Status | Archive mapper source |
| --- | --- | --- |
| `frame` | Mapped | `runschema.system.System` |
| `frame.info` | Unmapped | — |
| `frame.lattice_vectors` | Mapped | `runschema.system.System.atoms.lattice_vectors` |
| `frame.atoms` | Mapped | `runschema.system.System.atoms` |
| `frame.atoms.label` | Mapped | `runschema.system.System.atoms.labels` |
| `frame.atoms.array` | Mapped | `runschema.system.System.atoms.positions`<br>`runschema.system.System.atoms.velocities`<br>`runschema.calculation.Calculation.forces.total.value` |

## DL_POLY / FieldParser

**Summary:** 14 mapped, 4 unmapped quantities (77.78% coverage).

| File-parser quantity | Status | Archive mapper source |
| --- | --- | --- |
| `units` | Unmapped | — |
| `neutral_groups` | Unmapped | — |
| `molecule_types` | Unmapped | — |
| `molecule` | Mapped | `runschema.method.Method.molecule_parameters` |
| `molecule.label_nummols` | Mapped | `runschema.method.MoleculeParameters.label` |
| `molecule.atoms` | Mapped | `runschema.method.AtomParameters.label`<br>`runschema.method.AtomParameters.mass`<br>`runschema.method.AtomParameters.charge` |
| `molecule.shell` | Unmapped | — |
| `molecule.bonds` | Mapped | `runschema.method.Interaction.functional_form`<br>`runschema.method.Interaction.atom_indices`<br>`runschema.method.Interaction.parameters` |
| `molecule.angles` | Mapped | `runschema.method.Interaction.functional_form`<br>`runschema.method.Interaction.atom_indices`<br>`runschema.method.Interaction.parameters` |
| `molecule.constraints` | Mapped | `runschema.system.System.constraint.kind`<br>`runschema.system.System.constraint.atom_indices`<br>`runschema.system.System.constraint.parameters` |
| `molecule.dihedrals` | Mapped | `runschema.method.Interaction.functional_form`<br>`runschema.method.Interaction.atom_indices`<br>`runschema.method.Interaction.parameters` |
| `molecule.inversions` | Mapped | `runschema.method.Interaction.functional_form`<br>`runschema.method.Interaction.atom_indices`<br>`runschema.method.Interaction.parameters` |
| `molecule.rigid` | Mapped | `runschema.system.System.constraint.kind`<br>`runschema.system.System.constraint.atom_indices` |
| `molecule.teth` | Mapped | `runschema.method.Interaction.functional_form`<br>`runschema.method.Interaction.atom_indices`<br>`runschema.method.Interaction.parameters` |
| `vdw` | Mapped | `runschema.method.Interaction.functional_form`<br>`runschema.method.Interaction.atom_labels`<br>`runschema.method.Interaction.parameters` |
| `tbp` | Mapped | `runschema.method.Interaction.functional_form`<br>`runschema.method.Interaction.atom_labels`<br>`runschema.method.Interaction.parameters` |
| `fbp` | Mapped | `runschema.method.Interaction.functional_form`<br>`runschema.method.Interaction.atom_labels`<br>`runschema.method.Interaction.parameters` |
| `metal` | Mapped | `runschema.method.Interaction.functional_form`<br>`runschema.method.Interaction.atom_labels`<br>`runschema.method.Interaction.parameters` |

## DL_POLY / MainfileParser

**Summary:** 7 mapped, 3 unmapped quantities (70.00% coverage).

| File-parser quantity | Status | Archive mapper source |
| --- | --- | --- |
| `program_version_date` | Mapped | `runschema.run.Program.version` |
| `program_name` | Mapped | `runschema.run.Program.name` |
| `control_parameters` | Mapped | `simulationworkflowschema.molecular_dynamics.MolecularDynamics.method.thermodynamic_ensemble`<br>`simulationworkflowschema.molecular_dynamics.MolecularDynamics.method.integration_timestep` |
| `control_parameters.parameter` | Mapped | `simulationworkflowschema.molecular_dynamics.MolecularDynamics.method.thermodynamic_ensemble`<br>`simulationworkflowschema.molecular_dynamics.MolecularDynamics.method.integration_timestep` |
| `system_specification` | Unmapped | — |
| `system_specification.energy_unit` | Unmapped | — |
| `properties` | Mapped | `runschema.calculation.Calculation` |
| `properties.names` | Mapped | `runschema.calculation.Calculation.energy.total.value`<br>`runschema.calculation.Calculation.temperature`<br>`runschema.calculation.Calculation.pressure` |
| `properties.instantaneous` | Mapped | `runschema.calculation.Calculation.energy.total.value`<br>`runschema.calculation.Calculation.temperature`<br>`runschema.calculation.Calculation.pressure`<br>`runschema.calculation.Calculation.step`<br>`runschema.calculation.Calculation.time` |
| `properties.average` | Unmapped | — |

## GROMACS / md.log

**Summary:** 6 mapped, 8 unmapped quantities (42.86% coverage).

| File-parser quantity | Status | Archive mapper source |
| --- | --- | --- |
| `energies` | Unmapped | — |
| `step_info` | Unmapped | — |
| `time_start` | Mapped | `runschema.run.TimeRun.date_start` |
| `host_info` | Unmapped | — |
| `module_version` | Unmapped | — |
| `execution_path` | Unmapped | — |
| `working_path` | Unmapped | — |
| `header` | Mapped | `runschema.run.Program.version` |
| `header` | Mapped | `runschema.run.Program.version` |
| `input_parameters` | Mapped | `runschema.method.Method.force_field`<br>`simulationworkflowschema.MolecularDynamics.method`<br>`simulationworkflowschema.GeometryOptimization.method` |
| `maximum_force` | Mapped | `simulationworkflowschema.GeometryOptimizationResults.final_force_maximum` |
| `step` | Unmapped | — |
| `averages` | Unmapped | — |
| `time_end` | Mapped | `runschema.run.TimeRun.date_end` |

## GROMACS / mdout.mdp

**Summary:** 1 mapped, 0 unmapped quantities (100.00% coverage).

| File-parser quantity | Status | Archive mapper source |
| --- | --- | --- |
| `input_parameters` | Mapped | `runschema.method.Method.force_field`<br>`simulationworkflowschema.MolecularDynamics.method`<br>`simulationworkflowschema.GeometryOptimization.method` |

## GROMACS / dhdl.xvg

**Summary:** 0 mapped, 4 unmapped quantities (0.00% coverage).

| File-parser quantity | Status | Archive mapper source |
| --- | --- | --- |
| `title` | Unmapped | — |
| `xaxis` | Unmapped | — |
| `yaxis` | Unmapped | — |
| `column_headers` | Unmapped | — |

## GROMACS / ener.edr

**Coverage:** Not available — this parser uses custom archive-writing logic and exposes no reportable file-parser quantities.

## GROMOS

**Coverage:** Not available — this parser uses custom archive-writing logic and exposes no reportable file-parser quantities.

The parser configures a shared `BasicParser` (`simulationparsers.utils.BasicParser`) with regex patterns for program version, positions, total energy, pressure, and timestep rather than declaring a `TextParser`/`XMLParser`/`FileParser` subclass with `Quantity(...)` definitions.

## GULP / parser.py::MainfileParser

**Summary:** 56 mapped, 42 unmapped quantities (57.14% coverage).

| File-parser quantity | Status | Archive mapper source |
| --- | --- | --- |
| `coordinates_quantities.unit` | Mapped | `runschema.system.System.atoms.positions` |
| `coordinates_quantities.auxilliary_keys` | Unmapped | — |
| `coordinates_quantities.atom` | Mapped | `runschema.system.System.atoms.positions`<br>`runschema.system.System.atoms.labels` |
| `calc_quantities.energy_components` | Mapped | `runschema.calculation.Calculation.energy.total` |
| `calc_quantities.energy_components.key_val` | Mapped | `runschema.calculation.Calculation.energy.total` |
| `calc_quantities.bulk_optimisation` | Unmapped | — |
| `calc_quantities.bulk_optimisation.x_gulp_n_variables` | Unmapped | — |
| `calc_quantities.bulk_optimisation.x_gulp_max_n_calculations` | Unmapped | — |
| `calc_quantities.bulk_optimisation.x_gulp_max_hessian_update_interval` | Unmapped | — |
| `calc_quantities.bulk_optimisation.x_gulp_max_step_size` | Unmapped | — |
| `calc_quantities.bulk_optimisation.x_gulp_max_parameter_tolerance` | Unmapped | — |
| `calc_quantities.bulk_optimisation.x_gulp_max_function_tolerance` | Unmapped | — |
| `calc_quantities.bulk_optimisation.x_gulp_max_gradient_tolerance` | Unmapped | — |
| `calc_quantities.bulk_optimisation.x_gulp_max_gradient_component` | Unmapped | — |
| `calc_quantities.bulk_optimisation.cycle` | Unmapped | — |
| `calc_quantities.coordinates` | Mapped | `runschema.system.System.atoms.positions`<br>`runschema.system.System.atoms.labels` |
| `calc_quantities.lattice_vectors` | Mapped | `runschema.system.System.atoms.lattice_vectors` |
| `calc_quantities.cell_parameters_primitive` | Mapped | `runschema.system.System.atoms.lattice_vectors`<br>`runschema.system.System.atoms.positions` |
| `calc_quantities.cell_parameters` | Mapped | `runschema.system.System.atoms.lattice_vectors`<br>`runschema.system.System.atoms.positions` |
| `calc_quantities.elastic_constants` | Mapped | `simulationworkflowschema.ElasticResults.elastic_constants_matrix_second_order` |
| `calc_quantities.elastic_compliance` | Mapped | `simulationworkflowschema.ElasticResults.compliance_matrix_second_order` |
| `calc_quantities.mechanical_properties` | Mapped | `simulationworkflowschema.ElasticResults.bulk_modulus_reuss`<br>`simulationworkflowschema.ElasticResults.shear_modulus_reuss` |
| `calc_quantities.mechanical_properties.bulk_modulus` | Mapped | `simulationworkflowschema.ElasticResults.bulk_modulus_reuss`<br>`simulationworkflowschema.ElasticResults.bulk_modulus_voigt`<br>`simulationworkflowschema.ElasticResults.bulk_modulus_hill` |
| `calc_quantities.mechanical_properties.shear_modulus` | Mapped | `simulationworkflowschema.ElasticResults.shear_modulus_reuss`<br>`simulationworkflowschema.ElasticResults.shear_modulus_voigt`<br>`simulationworkflowschema.ElasticResults.shear_modulus_hill` |
| `calc_quantities.mechanical_properties.x_gulp_velocity_s_wave` | Unmapped | — |
| `calc_quantities.mechanical_properties.x_gulp_velocity_p_wave` | Unmapped | — |
| `calc_quantities.mechanical_properties.compressibility` | Unmapped | — |
| `calc_quantities.mechanical_properties.x_gulp_youngs_modulus` | Unmapped | — |
| `calc_quantities.mechanical_properties.poissons_ratio` | Unmapped | — |
| `calc_quantities.x_gulp_piezoelectric_strain_matrix` | Unmapped | — |
| `calc_quantities.x_gulp_piezoelectric_stress_matrix` | Unmapped | — |
| `calc_quantities.x_gulp_static_dielectric_constant_tensor` | Unmapped | — |
| `calc_quantities.x_gulp_high_frequency_dielectric_constant_tensor` | Unmapped | — |
| `calc_quantities.x_gulp_static_refractive_indices` | Unmapped | — |
| `calc_quantities.x_gulp_static_refractive_indices.value` | Unmapped | — |
| `calc_quantities.x_gulp_high_frequency_refractive_indices` | Unmapped | — |
| `calc_quantities.x_gulp_high_frequency_refractive_indices.value` | Unmapped | — |
| `interaction_quantities.atom_type` | Mapped | `runschema.method.Interaction.atom_labels` |
| `interaction_quantities.functional_form` | Mapped | `runschema.method.Interaction.functional_form` |
| `header` | Mapped | `runschema.run.Run.program.version`<br>`runschema.run.Run.x_gulp_title` |
| `header.program_version` | Mapped | `runschema.run.Run.program.version` |
| `header.task` | Unmapped | — |
| `header.title` | Unmapped | — |
| `date_start` | Mapped | `runschema.run.Run.time_run.date_start` |
| `date_end` | Mapped | `runschema.run.Run.time_run.date_end` |
| `x_gulp_n_cpu` | Unmapped | — |
| `x_gulp_host_name` | Unmapped | — |
| `x_gulp_total_n_configurations_input` | Unmapped | — |
| `input_configuration` | Mapped | `runschema.system.System.atoms` |
| `input_configuration.x_gulp_formula` | Unmapped | — |
| `input_configuration.x_gulp_pbc` | Mapped | `runschema.system.System.atoms.periodic` |
| `input_configuration.x_gulp_space_group` | Mapped | `runschema.system.System.atoms.positions`<br>`runschema.system.System.atoms.labels`<br>`runschema.system.System.atoms.lattice_vectors` |
| `input_configuration.x_gulp_patterson_group` | Unmapped | — |
| `input_configuration.lattice_vectors` | Mapped | `runschema.system.System.atoms.lattice_vectors` |
| `input_configuration.cell_parameters` | Mapped | `runschema.system.System.atoms.lattice_vectors`<br>`runschema.system.System.atoms.positions` |
| `input_configuration.cell_parameters` | Mapped | `runschema.system.System.atoms.lattice_vectors`<br>`runschema.system.System.atoms.positions` |
| `input_configuration.coordinates` | Mapped | `runschema.system.System.atoms.positions`<br>`runschema.system.System.atoms.labels` |
| `input_information` | Mapped | `runschema.method.Method.force_field`<br>`runschema.method.Method.atom_parameters` |
| `input_information.species` | Mapped | `runschema.method.AtomParameters.label`<br>`runschema.method.AtomParameters.atom_number`<br>`runschema.method.AtomParameters.mass`<br>`runschema.method.AtomParameters.charge` |
| `input_information.pgfnff` | Mapped | `runschema.method.ForceField.model.name`<br>`runschema.method.ForceField.model.contributions.parameters` |
| `input_information.pgfnff.key_parameter` | Mapped | `runschema.method.Interaction.parameters` |
| `input_information.pair_potential` | Mapped | `runschema.method.ForceField.model.contributions` |
| `input_information.pair_potential.interaction` | Mapped | `runschema.method.Interaction.functional_form`<br>`runschema.method.Interaction.atom_labels`<br>`runschema.method.Interaction.parameters` |
| `input_information.pair_potential.interaction.key_parameter` | Mapped | `runschema.method.Interaction.parameters` |
| `input_information.three_body_potential` | Mapped | `runschema.method.ForceField.model.contributions` |
| `input_information.three_body_potential.interaction` | Mapped | `runschema.method.Interaction.functional_form`<br>`runschema.method.Interaction.atom_labels`<br>`runschema.method.Interaction.parameters` |
| `input_information.three_body_potential.interaction.key_parameter` | Mapped | `runschema.method.Interaction.parameters` |
| `input_information.four_body_potential` | Mapped | `runschema.method.ForceField.model.contributions` |
| `input_information.four_body_potential.interaction` | Mapped | `runschema.method.Interaction.functional_form`<br>`runschema.method.Interaction.atom_labels`<br>`runschema.method.Interaction.parameters` |
| `input_information.four_body_potential.interaction.key_parameter` | Mapped | `runschema.method.Interaction.parameters` |
| `input_information.interatomic_potential` | Mapped | `runschema.method.ForceField.model.contributions` |
| `input_information.interatomic_potential.interaction` | Mapped | `runschema.method.Interaction.functional_form`<br>`runschema.method.Interaction.atom_labels`<br>`runschema.method.Interaction.parameters` |
| `input_information.interatomic_potential.interaction.key_parameter` | Mapped | `runschema.method.Interaction.parameters` |
| `single_point` | Mapped | `runschema.calculation.Calculation` |
| `single_point.calculation` | Mapped | `runschema.calculation.Calculation` |
| `molecular_dynamics` | Mapped | `simulationworkflowschema.MolecularDynamics` |
| `molecular_dynamics.ensemble_type` | Mapped | `simulationworkflowschema.MolecularDynamicsMethod.thermodynamic_ensemble` |
| `molecular_dynamics.x_gulp_friction_temperature_bath` | Unmapped | — |
| `molecular_dynamics.x_gulp_n_mobile_ions` | Unmapped | — |
| `molecular_dynamics.x_gulp_n_degrees_of_freedom` | Unmapped | — |
| `molecular_dynamics.timestep` | Mapped | `simulationworkflowschema.MolecularDynamicsMethod.integration_timestep` |
| `molecular_dynamics.x_gulp_equilibration_time` | Unmapped | — |
| `molecular_dynamics.x_gulp_production_time` | Unmapped | — |
| `molecular_dynamics.x_gulp_scaling_time` | Unmapped | — |
| `molecular_dynamics.x_gulp_scaling_frequency` | Unmapped | — |
| `molecular_dynamics.x_gulp_sampling_frequency` | Unmapped | — |
| `molecular_dynamics.x_gulp_write_frequency` | Unmapped | — |
| `molecular_dynamics.x_gulp_td_force_start_time` | Unmapped | — |
| `molecular_dynamics.x_gulp_td_field_start_time` | Unmapped | — |
| `molecular_dynamics.step` | Mapped | `runschema.calculation.Calculation` |
| `molecular_dynamics.step.time` | Mapped | `runschema.calculation.Calculation.time` |
| `molecular_dynamics.step.energy_kinetic` | Mapped | `runschema.calculation.Calculation.energy.total.kinetic` |
| `molecular_dynamics.step.energy_potential` | Mapped | `runschema.calculation.Calculation.energy.total.potential` |
| `molecular_dynamics.step.energy_total` | Mapped | `runschema.calculation.Calculation.energy.total.value` |
| `molecular_dynamics.step.temperature` | Mapped | `runschema.calculation.Calculation.temperature` |
| `molecular_dynamics.step.pressure` | Mapped | `runschema.calculation.Calculation.pressure` |
| `defect` | Mapped | `runschema.calculation.Calculation` |
| `defect.calculation` | Mapped | `runschema.calculation.Calculation` |

## H5MD

**Coverage:** Not available — this parser uses custom archive-writing logic and exposes no reportable file-parser quantities.

The `HDF5Parser` subclass of `FileParser` reads the mainfile as an HDF5 file via `h5py` and extracts values by runtime path lookup (`get_value`/`get_attribute`), so it declares no static `Quantity(...)` definitions to report.

## LAMMPS / data

**Summary:** 1 mapped, 48 unmapped quantities (2.04% coverage).

| File-parser quantity | Status | Archive mapper source |
| --- | --- | --- |
| `atoms` | Unmapped | — |
| `bonds` | Unmapped | — |
| `angles` | Unmapped | — |
| `dihedrals` | Unmapped | — |
| `impropers` | Unmapped | — |
| `atom types` | Unmapped | — |
| `bond types` | Unmapped | — |
| `angle types` | Unmapped | — |
| `dihedral types` | Unmapped | — |
| `improper types` | Unmapped | — |
| `extra bond per atom` | Unmapped | — |
| `extra/bond/per/atom` | Unmapped | — |
| `extra angle per atom` | Unmapped | — |
| `extra/angle/per/atom` | Unmapped | — |
| `extra dihedral per atom` | Unmapped | — |
| `extra/dihedral/per/atom` | Unmapped | — |
| `extra improper per atom` | Unmapped | — |
| `extra/improper/per/atom` | Unmapped | — |
| `extra special per atom` | Unmapped | — |
| `extra/special/per/atom` | Unmapped | — |
| `ellipsoids` | Unmapped | — |
| `lines` | Unmapped | — |
| `triangles` | Unmapped | — |
| `bodies` | Unmapped | — |
| `Atoms` | Unmapped | — |
| `Velocities` | Unmapped | — |
| `Masses` | Mapped | `system.atoms.labels` |
| `Ellipsoids` | Unmapped | — |
| `Lines` | Unmapped | — |
| `Triangles` | Unmapped | — |
| `Bodies` | Unmapped | — |
| `Bonds` | Unmapped | — |
| `Angles` | Unmapped | — |
| `Dihedrals` | Unmapped | — |
| `Impropers` | Unmapped | — |
| `Pair Coeffs` | Unmapped | — |
| `PairIJ Coeffs` | Unmapped | — |
| `Bond Coeffs` | Unmapped | — |
| `Angle Coeffs` | Unmapped | — |
| `Dihedral Coeffs` | Unmapped | — |
| `Improper Coeffs` | Unmapped | — |
| `BondBond Coeffs` | Unmapped | — |
| `BondAngle Coeffs` | Unmapped | — |
| `MiddleBondTorsion Coeffs` | Unmapped | — |
| `EndBondTorsion Coeffs` | Unmapped | — |
| `AngleTorsion Coeffs` | Unmapped | — |
| `AngleAngleTorsion Coeffs` | Unmapped | — |
| `BondBond13 Coeffs` | Unmapped | — |
| `AngleAngle Coeffs` | Unmapped | — |

## LAMMPS / dump

**Summary:** 4 mapped, 0 unmapped quantities (100.00% coverage).

| File-parser quantity | Status | Archive mapper source |
| --- | --- | --- |
| `time_step` | Mapped | `calculation.step`<br>`calculation.time` |
| `n_atoms` | Mapped | `system.atoms.n_atoms` |
| `pbc_cell` | Mapped | `system.atoms.lattice_vectors`<br>`system.atoms.periodic` |
| `atoms_info` | Mapped | `system.atoms.positions`<br>`system.atoms.velocities`<br>`system.atoms.labels`<br>`calculation.forces.total.value`<br>`system.atoms_group` |

## LAMMPS / dump.xyz

**Summary:** 1 mapped, 0 unmapped quantities (100.00% coverage).

| File-parser quantity | Status | Archive mapper source |
| --- | --- | --- |
| `atoms_info` | Mapped | `system.atoms.positions`<br>`system.atoms.labels` |

## LAMMPS / log.lammps

**Summary:** 13 mapped, 100 unmapped quantities (11.50% coverage).

| File-parser quantity | Status | Archive mapper source |
| --- | --- | --- |
| `angle_coeff` | Unmapped | — |
| `angle_style` | Unmapped | — |
| `atom_modify` | Unmapped | — |
| `atom_style` | Unmapped | — |
| `balance` | Unmapped | — |
| `bond_coeff` | Unmapped | — |
| `bond_style` | Unmapped | — |
| `bond_write` | Unmapped | — |
| `boundary` | Unmapped | — |
| `change_box` | Unmapped | — |
| `clear` | Unmapped | — |
| `comm_modify` | Unmapped | — |
| `comm_style` | Unmapped | — |
| `compute` | Unmapped | — |
| `compute_modify` | Unmapped | — |
| `create_atoms` | Unmapped | — |
| `create_bonds` | Unmapped | — |
| `create_box` | Unmapped | — |
| `delete_bonds` | Unmapped | — |
| `dielectric` | Unmapped | — |
| `dihedral_coeff` | Unmapped | — |
| `dihedral_style` | Unmapped | — |
| `dimension` | Unmapped | — |
| `displace_atoms` | Unmapped | — |
| `dump` | Unmapped | — |
| `dump_modify` | Unmapped | — |
| `dynamical_matrix` | Unmapped | — |
| `echo` | Unmapped | — |
| `fix` | Mapped | `simulationworkflowschema.MolecularDynamics.method.thermodynamic_ensemble`<br>`simulationworkflowschema.MolecularDynamics.method.thermostat_parameters`<br>`simulationworkflowschema.MolecularDynamics.method.barostat_parameters` |
| `fix_modify` | Unmapped | — |
| `group` | Unmapped | — |
| `group2ndx` | Unmapped | — |
| `ndx2group` | Unmapped | — |
| `hyper` | Unmapped | — |
| `if` | Unmapped | — |
| `improper_coeff` | Unmapped | — |
| `improper_style` | Unmapped | — |
| `include` | Unmapped | — |
| `info` | Unmapped | — |
| `jump` | Unmapped | — |
| `kim_init` | Unmapped | — |
| `kim_interactions` | Unmapped | — |
| `kim_query` | Unmapped | — |
| `kim_param` | Unmapped | — |
| `kim_property` | Unmapped | — |
| `kspace_modify` | Unmapped | — |
| `kspace_style` | Mapped | `method.force_field.force_calculations.coulomb_type` |
| `label` | Unmapped | — |
| `lattice` | Unmapped | — |
| `log` | Unmapped | — |
| `mass` | Unmapped | — |
| `message` | Unmapped | — |
| `min_modify` | Unmapped | — |
| `min_style` | Mapped | `simulationworkflowschema.GeometryOptimization.method.method` |
| `minimize` | Mapped | `simulationworkflowschema.GeometryOptimization.method.optimization_steps_maximum`<br>`simulationworkflowschema.GeometryOptimization.method.convergence_tolerance_force_maximum`<br>`simulationworkflowschema.GeometryOptimization.method.convergence_tolerance_energy_difference` |
| `minimize/kk` | Mapped | `simulationworkflowschema.GeometryOptimization.method.optimization_steps_maximum`<br>`simulationworkflowschema.GeometryOptimization.method.convergence_tolerance_force_maximum`<br>`simulationworkflowschema.GeometryOptimization.method.convergence_tolerance_energy_difference` |
| `molecule` | Unmapped | — |
| `neb` | Unmapped | — |
| `neb/spin` | Unmapped | — |
| `neigh_modify` | Mapped | `method.force_field.force_calculations.neighbor_searching.neighbor_update_frequency` |
| `neighbor` | Mapped | `method.force_field.force_calculations.neighbor_searching.neighbor_update_cutoff` |
| `newton` | Unmapped | — |
| `next` | Unmapped | — |
| `package` | Unmapped | — |
| `pair_coeff` | Unmapped | — |
| `pair_modify` | Unmapped | — |
| `pair_style` | Mapped | `method.force_field.force_calculations.vdw_cutoff`<br>`method.force_field.force_calculations.coulomb_cutoff` |
| `pair_write` | Unmapped | — |
| `partition` | Unmapped | — |
| `prd` | Unmapped | — |
| `print` | Unmapped | — |
| `processors` | Unmapped | — |
| `quit` | Unmapped | — |
| `read_data` | Unmapped | — |
| `read_dump` | Unmapped | — |
| `read_restart` | Unmapped | — |
| `region` | Unmapped | — |
| `replicate` | Unmapped | — |
| `rerun` | Unmapped | — |
| `reset_atom_ids` | Unmapped | — |
| `reset_mol_ids` | Unmapped | — |
| `reset_timestep` | Unmapped | — |
| `restart` | Unmapped | — |
| `run` | Unmapped | — |
| `run_style` | Unmapped | — |
| `server` | Unmapped | — |
| `set` | Unmapped | — |
| `shell` | Unmapped | — |
| `special_bonds` | Unmapped | — |
| `suffix` | Unmapped | — |
| `tad` | Unmapped | — |
| `temper/grem` | Unmapped | — |
| `temper/npt` | Unmapped | — |
| `thermo` | Unmapped | — |
| `thermo_modify` | Unmapped | — |
| `thermo_style` | Unmapped | — |
| `third_order` | Unmapped | — |
| `timer` | Unmapped | — |
| `timestep` | Mapped | `simulationworkflowschema.MolecularDynamics.method.integration_timestep`<br>`calculation.time` |
| `uncompute` | Unmapped | — |
| `undump` | Unmapped | — |
| `unfix` | Unmapped | — |
| `units` | Unmapped | — |
| `variable` | Unmapped | — |
| `velocity` | Unmapped | — |
| `write_coeff` | Unmapped | — |
| `write_data` | Unmapped | — |
| `write_dump` | Unmapped | — |
| `write_restart` | Unmapped | — |
| `program_version` | Mapped | `run.program.version` |
| `finished` | Mapped | `simulationworkflowschema.MolecularDynamics.results.finished_normally` |
| `minimization_stats` | Mapped | `simulationworkflowschema.GeometryOptimization.results.final_energy_difference`<br>`simulationworkflowschema.GeometryOptimization.results.final_force_maximum` |
| `thermo_data` | Mapped | `calculation.energy.total`<br>`calculation.energy.current`<br>`calculation.energy.contributions`<br>`calculation.pressure`<br>`calculation.temperature`<br>`calculation.time_calculation` |

## libAtoms

**Coverage:** Not available — this parser uses custom archive-writing logic and exposes no reportable file-parser quantities.

The parser delegates to `simulationparsers.utils.BasicParser`, configuring it with regex string patterns (passed as keyword arguments) rather than declaring a `TextParser`/`XMLParser`/`FileParser` subclass with `Quantity(...)` objects.

## NAMD / ConfigParser

**Summary:** 0 mapped, 1 unmapped quantities (0.00% coverage).

| File-parser quantity | Status | Archive mapper source |
| --- | --- | --- |
| `parameter` | Unmapped | — |

## NAMD / MainfileParser

**Summary:** 6 mapped, 8 unmapped quantities (42.86% coverage).

| File-parser quantity | Status | Archive mapper source |
| --- | --- | --- |
| `version_arch` | Mapped | `runschema.run.program.version` |
| `config_file` | Unmapped | — |
| `simulation_parameters` | Mapped | `runschema.run.system.atoms.lattice_vectors` |
| `simulation_parameters.parameter` | Unmapped | — |
| `simulation_parameters.cell` | Mapped | `runschema.run.system.atoms.lattice_vectors` |
| `simulation_parameters.output_file` | Unmapped | — |
| `simulation_parameters.coordinate_file` | Unmapped | — |
| `simulation_parameters.structure_file` | Unmapped | — |
| `simulation_parameters.parameter_file` | Unmapped | — |
| `step` | Mapped | `runschema.run.calculation.energy.total.value`<br>`runschema.run.calculation.energy.electronic.value`<br>`runschema.run.calculation.energy.van_der_waals.value`<br>`runschema.run.calculation.energy.contributions.value`<br>`runschema.run.calculation.temperature`<br>`runschema.run.calculation.pressure` |
| `timing` | Mapped | `runschema.run.calculation.time_calculation`<br>`runschema.run.calculation.time_physical` |
| `total_time` | Mapped | `runschema.run.calculation.time_calculation`<br>`runschema.run.calculation.time_physical` |
| `property_names` | Unmapped | — |
| `coordinates_write_step` | Unmapped | — |

## Tinker / KeyParser

**Summary:** 0 mapped, 1 unmapped quantities (0.00% coverage).

| File-parser quantity | Status | Archive mapper source |
| --- | --- | --- |
| `key_val` | Unmapped | — |

## Tinker / RunParser

**Summary:** 1 mapped, 2 unmapped quantities (33.33% coverage).

| File-parser quantity | Status | Archive mapper source |
| --- | --- | --- |
| `molecular_dynamics` | Mapped | `simulationworkflowschema.MolecularDynamics.method.integration_timestep`<br>`simulationworkflowschema.MolecularDynamics.method.thermodynamic_ensemble` |
| `geometry_optimization` | Unmapped | — |
| `single_point` | Unmapped | — |

## Tinker / OutParser

**Summary:** 9 mapped, 22 unmapped quantities (29.03% coverage).

| File-parser quantity | Status | Archive mapper source |
| --- | --- | --- |
| `iteration` | Unmapped | — |
| `iteration.step` | Mapped | `runschema.calculation.Calculation.energy.total.value` |
| `program_version` | Mapped | `runschema.run.Program.version` |
| `vibrate` | Unmapped | — |
| `vibrate.eigenvalues` | Unmapped | — |
| `vibrate.frequencies` | Mapped | `runschema.calculation.VibrationalFrequencies.value` |
| `minimize` | Unmapped | — |
| `minimize.method` | Mapped | `simulationworkflowschema.GeometryOptimizationMethod.method` |
| `minimize.x_tiner_final_function_value` | Unmapped | — |
| `minimize.x_tinker_final_rms_gradient` | Unmapped | — |
| `minimize.x_tinker_final_gradient_norm` | Unmapped | — |
| `dynamic` | Unmapped | — |
| `dynamic.instantaneous_values` | Unmapped | — |
| `dynamic.instantaneous_values.step` | Mapped | `runschema.calculation.Calculation.step` |
| `dynamic.instantaneous_values.time` | Unmapped | — |
| `dynamic.instantaneous_values.potential` | Mapped | `runschema.calculation.Calculation.energy.total.value`<br>`runschema.calculation.Calculation.energy.potential.value` |
| `dynamic.instantaneous_values.kinetic` | Mapped | `runschema.calculation.Calculation.energy.total.value`<br>`runschema.calculation.Calculation.energy.kinetic.value` |
| `dynamic.instantaneous_values.lattice_lengths` | Unmapped | — |
| `dynamic.instantaneous_values.lattice_angles` | Unmapped | — |
| `dynamic.instantaneous_values.frame` | Unmapped | — |
| `dynamic.instantaneous_values.coordinate_file` | Unmapped | — |
| `dynamic.average_values` | Unmapped | — |
| `dynamic.average_values.step` | Unmapped | — |
| `dynamic.average_values.time` | Unmapped | — |
| `dynamic.average_values.energy_total` | Unmapped | — |
| `dynamic.average_values.potential` | Unmapped | — |
| `dynamic.average_values.kinetic` | Unmapped | — |
| `dynamic.average_values.temperature` | Mapped | `runschema.calculation.Calculation.temperature` |
| `dynamic.average_values.pressure` | Mapped | `runschema.calculation.Calculation.pressure` |
| `dynamic.average_values.density` | Unmapped | — |
| `run` | Unmapped | — |

## xTB / OutParser

**Summary:** 33 mapped, 47 unmapped quantities (41.25% coverage).

| File-parser quantity | Status | Archive mapper source |
| --- | --- | --- |
| `setup` | Unmapped | — |
| `setup.parameter` | Unmapped | — |
| `summary` | Mapped | `runschema.calculation.Calculation.energy` |
| `summary.energy_total` | Mapped | `runschema.calculation.Energy.total.value` |
| `summary.x_xtb_gradient_norm` | Unmapped | — |
| `summary.x_xtb_hl_gap` | Unmapped | — |
| `summary.energy_x_xtb_scc` | Unmapped | — |
| `summary.energy_x_xtb_isotropic_es` | Unmapped | — |
| `summary.energy_x_xtb_anisotropic_es` | Unmapped | — |
| `summary.energy_x_xtb_anisotropic_xc` | Unmapped | — |
| `summary.energy_x_xtb_dispersion` | Unmapped | — |
| `summary.energy_electrostatic` | Mapped | `runschema.calculation.Energy.electrostatic.value` |
| `summary.energy_x_xtb_repulsion` | Unmapped | — |
| `summary.energy_x_xtb_halogen_bond_corr` | Unmapped | — |
| `summary.energy_x_xtb_add_restraining` | Unmapped | — |
| `summary.charge_total` | Unmapped | — |
| `eigenvalues` | Mapped | `runschema.calculation.BandEnergies.occupations`<br>`runschema.calculation.BandEnergies.energies`<br>`runschema.calculation.BandEnergies.kpoints` |
| `hl_gap` | Unmapped | — |
| `energy_fermi` | Unmapped | — |
| `dipole` | Mapped | `runschema.calculation.Multipoles.dipole` |
| `dipole.q` | Unmapped | — |
| `dipole.full` | Mapped | `runschema.calculation.MultipolesEntry.total` |
| `quadrupole` | Mapped | `runschema.calculation.Multipoles.quadrupole` |
| `quadrupole.q` | Unmapped | — |
| `quadrupole.full` | Mapped | `runschema.calculation.MultipolesEntry.total` |
| `quadrupole.q_dip` | Unmapped | — |
| `file` | Unmapped | — |
| `model` | Mapped | `runschema.method.Method.tb.xtb` |
| `model.reference` | Mapped | `runschema.method.xTB.reference` |
| `model.contribution` | Mapped | `runschema.method.xTB.hamiltonian`<br>`runschema.method.xTB.coulomb`<br>`runschema.method.xTB.repulsion`<br>`runschema.method.xTB.contributions` |
| `model.contribution.name` | Mapped | `runschema.method.Interaction.type` |
| `model.contribution.parameters` | Mapped | `runschema.method.Interaction.parameters` |
| `scf_iteration` | Mapped | `runschema.calculation.Calculation.scf_iteration` |
| `scf_iteration.step` | Mapped | `runschema.calculation.ScfIteration.energy.total.value`<br>`runschema.calculation.ScfIteration.energy.change` |
| `scf_iteration.converged` | Unmapped | — |
| `cycle` | Mapped | `runschema.calculation.Calculation` |
| `cycle.energy_total` | Mapped | `runschema.calculation.Energy.total.value` |
| `cycle.energy_change` | Mapped | `runschema.calculation.Energy.change` |
| `cycle.scf_iteration` | Mapped | `runschema.calculation.Calculation.scf_iteration` |
| `cycle.scf_iteration.step` | Mapped | `runschema.calculation.ScfIteration.energy.total.value`<br>`runschema.calculation.ScfIteration.energy.change` |
| `cycle.scf_iteration.time` | Unmapped | — |
| `converged` | Unmapped | — |
| `final_structure` | Unmapped | — |
| `final_structure.atom_labels` | Unmapped | — |
| `final_structure.atom_positions` | Unmapped | — |
| `final_single_point` | Mapped | `runschema.calculation.Calculation` |
| `traj_file` | Unmapped | — |
| `x_xtb_md_time` | Unmapped | — |
| `timestep` | Unmapped | — |
| `x_xtb_scc_accuracy` | Unmapped | — |
| `x_xtb_temperature` | Unmapped | — |
| `x_xtb_max_steps` | Unmapped | — |
| `x_xtb_block_length` | Unmapped | — |
| `x_xtb_dumpstep_trj` | Unmapped | — |
| `x_xtb_dumpstep_coords` | Unmapped | — |
| `x_xtb_h_atoms_mass` | Unmapped | — |
| `x_xtb_n_degrees_freedom` | Unmapped | — |
| `x_xtb_shake_bonds` | Unmapped | — |
| `x_xtb_berendsen` | Unmapped | — |
| `cycle` | Mapped | `simulationworkflowschema.molecular_dynamics` thermodynamics step: `runschema.calculation.Calculation.step`<br>`runschema.calculation.Calculation.temperature`<br>`runschema.calculation.Energy.total.potential`<br>`runschema.calculation.Energy.total.kinetic`<br>`runschema.calculation.Energy.total.value` |
| `program_version` | Mapped | `runschema.run.Program.version` |
| `date_start` | Mapped | `runschema.run.TimeRun.date_start` |
| `date_end` | Mapped | `runschema.run.TimeRun.date_end` |
| `calculation_setup` | Unmapped | — |
| `calculation_setup.parameter` | Unmapped | — |
| `gfnff` | Mapped | `runschema.calculation.Calculation` |
| `gfn1` | Mapped | `runschema.calculation.Calculation` |
| `gfn2` | Mapped | `runschema.calculation.Calculation` |
| `ancopt` | Mapped | `simulationworkflowschema.GeometryOptimization` |
| `md` | Mapped | `simulationworkflowschema.MolecularDynamics` |
| `property` | Mapped | `runschema.calculation.Calculation.multipoles` |
| `geometry` | Unmapped | — |
| `energy_total` | Unmapped | — |
| `gradient_norm` | Unmapped | — |
| `hl_gap` | Unmapped | — |
| `topo_file` | Unmapped | — |
| `footer` | Mapped | `runschema.calculation.Calculation.time_physical`<br>`runschema.calculation.Calculation.time_calculation` |
| `footer.end_time` | Unmapped | — |
| `footer.wall_time` | Mapped | `runschema.calculation.Calculation.time_physical`<br>`runschema.calculation.Calculation.time_calculation` |
| `footer.cpu_time` | Unmapped | — |

## xTB / CoordParser

**Summary:** 7 mapped, 0 unmapped quantities (100.00% coverage).

| File-parser quantity | Status | Archive mapper source |
| --- | --- | --- |
| `coord_unit` | Mapped | `runschema.system.Atoms.positions` |
| `positions_labels` | Mapped | `runschema.system.Atoms.positions`<br>`runschema.system.Atoms.labels` |
| `periodic` | Mapped | `runschema.system.Atoms.periodic` |
| `lattice_unit` | Mapped | `runschema.system.Atoms.lattice_vectors` |
| `lattice` | Mapped | `runschema.system.Atoms.lattice_vectors` |
| `cell_unit` | Mapped | `runschema.system.Atoms.lattice_vectors` |
| `cell` | Mapped | `runschema.system.Atoms.lattice_vectors` |

## xTB / TrajParser

**Summary:** 3 mapped, 0 unmapped quantities (100.00% coverage).

| File-parser quantity | Status | Archive mapper source |
| --- | --- | --- |
| `frame` | Mapped | `runschema.system.Atoms` |
| `frame.positions` | Mapped | `runschema.system.Atoms.positions` |
| `frame.labels` | Mapped | `runschema.system.Atoms.labels` |

