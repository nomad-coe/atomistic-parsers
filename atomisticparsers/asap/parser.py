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
from ase.io.trajectory import Trajectory

from nomad.units import ureg
from nomad.parsing.file_parser import FileParser
from runschema.run import Run, Program
from runschema.method import Method, ForceField, Model
from simulationworkflowschema import GeometryOptimization, GeometryOptimizationMethod
from atomisticparsers.utils import MDParser
from .metainfo.asap import MolecularDynamics  # pylint: disable=unused-import


class TrajParser(FileParser):
    def __init__(self):
        super().__init__()

    @property
    def traj(self):
        if self._file_handler is None:
            try:
                self._file_handler = Trajectory(self.mainfile, "r")
                # check if traj file is really asap
                if "calculator" in self._file_handler.backend.keys():
                    if self._file_handler.backend.calculator.name != "emt":  # pylint: disable=E1101
                        self.logger.error("Trajectory is not ASAP.")
                        self._file_handler = None
            except Exception:
                self.logger.error("Error reading trajectory file.")
        return self._file_handler

    def get_version(self):
        if hasattr(self.traj, "ase_version") and self.traj.ase_version:
            return self.traj.ase_version
        else:
            return "3.x.x"

    def parse(self):
        pass


class AsapParser(MDParser):
    def __init__(self):
        self.traj_parser = TrajParser()
        super().__init__()

    def parse_method(self):
        traj = self.traj_parser.traj
        sec_method = Method()
        self.archive.run[0].method.append(sec_method)

        if traj[0].calc is not None:
            sec_method.force_field = ForceField(model=[Model(name=traj[0].calc.name)])

        description = traj.description if hasattr(traj, "description") else dict()
        if not description:
            return

        calc_type = description.get("type")
        if calc_type == "optimization":
            workflow = GeometryOptimization(method=GeometryOptimizationMethod())
            workflow.x_asap_maxstep = description.get("maxstep", 0)
            workflow.method.method = description.get("optimizer", "").lower()
            self.archive.workflow2 = workflow
        elif calc_type == "molecular-dynamics":
            data = {}
            data["x_asap_timestep"] = description.get("timestep", 0)
            data["x_asap_temperature"] = description.get("temperature", 0)
            md_type = description.get("md-type", "")
            thermodynamic_ensemble = None
            if "Langevin" in md_type:
                data["x_asap_langevin_friction"] = description.get("friction", 0)
                thermodynamic_ensemble = "NVT"
            elif "NVT" in md_type:
                thermodynamic_ensemble = "NVT"
            elif "Verlet" in md_type:
                thermodynamic_ensemble = "NVE"
            elif "NPT" in md_type:
                thermodynamic_ensemble = "NPT"
            data["method"] = {"thermodynamic_ensemble": thermodynamic_ensemble}
            self.parse_md_workflow(data)

    def write_to_archive(self) -> None:
        self.traj_parser.mainfile = self.mainfile
        if self.traj_parser.traj is None:
            return

        sec_run = Run()
        self.archive.run.append(sec_run)
        sec_run.program = Program(name="ASAP", version=self.traj_parser.get_version())

        # TODO do we build the topology and method for each frame
        self.parse_method()

        # set up md parser
        self.n_atoms = max(
            [traj.get_global_number_of_atoms() for traj in self.traj_parser.traj]
        )
        steps = [
            (traj.description if hasattr(traj, "description") else dict()).get(
                "interval", 1
            )
            * n
            for n, traj in enumerate(self.traj_parser.traj)
        ]
        self.trajectory_steps = steps
        self.thermodynamics_steps = steps

        def get_constraint_name(constraint):
            def index():
                d = constraint["kwargs"].get("direction")
                return ((d / np.linalg.norm(d)) ** 2).argsort()[2]

            name = constraint.get("name")
            if name == "FixedPlane":
                return ["fix_yz", "fix_xz", "fix_xy"][index()]
            elif name == "FixedLine":
                return ["fix_x", "fix_y", "fix_z"][index()]
            elif name == "FixAtoms":
                return "fix_xyz"
            else:
                return name

        for step in self.trajectory_steps:
            traj = self.traj_parser.traj[steps.index(step)]
            lattice_vectors = traj.get_cell() * ureg.angstrom
            labels = traj.get_chemical_symbols()
            positions = traj.get_positions() * ureg.angstrom
            periodic = traj.get_pbc()
            if (velocities := traj.get_velocities()) is not None:
                velocities = velocities * (ureg.angstrom / ureg.fs)

            constraints = []
            for constraint in traj.constraints:
                as_dict = constraint.todict()
                indices = as_dict["kwargs"].get("a", as_dict["kwargs"].get("indices"))
                constraints.append(
                    dict(
                        atom_indices=np.asarray(indices),
                        kind=get_constraint_name(as_dict),
                    )
                )
            self.parse_trajectory_step(
                dict(
                    atoms=dict(
                        lattice_vectors=lattice_vectors,
                        labels=labels,
                        positions=positions,
                        periodic=periodic,
                        velocities=velocities,
                    ),
                    constraint=constraints,
                )
            )

        for step in self.thermodynamics_steps:
            traj = self.traj_parser.traj[steps.index(step)]
            if (total_energy := traj.get_total_energy()) is not None:
                total_energy = total_energy * ureg.eV
            if (forces := traj.get_forces()) is not None:
                forces = forces * ureg.eV / ureg.angstrom
            if (forces_raw := traj.get_forces(apply_constraint=False)) is not None:
                forces_raw * ureg.eV / ureg.angstrom
            self.parse_thermodynamics_step(
                dict(
                    energy=dict(total=dict(value=total_energy)),
                    forces=dict(total=dict(value=forces, value_raw=forces_raw)),
                )
            )
