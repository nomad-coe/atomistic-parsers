<!-- generated-by: Claude Opus 4.8 | last_updated: 2026-09-09 -->
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
