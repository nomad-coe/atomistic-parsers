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
from simulationworkflowschema.geometry_optimization import GeometryOptimization
from simulationworkflowschema.molecular_dynamics import MolecularDynamics
from atomisticparsers.utils import ASETrajParser
from .metainfo import asap  # pylint: disable=unused-import


class AsapParser(ASETrajParser):
    def parse_method(self):
        super().parse_method()
        traj = self.traj_parser.traj

        description = traj.description if hasattr(traj, 'description') else dict()
        if not description:
            return

        workflow = self.archive.workflow2
        if isinstance(workflow, GeometryOptimization):
            workflow.x_asap_maxstep = description.get('maxstep', 0)
        elif isinstance(workflow, MolecularDynamics):
            workflow.x_asap_timestep = description.get('timestep', 0)
            workflow.x_asap_temperature = description.get('temperature', 0)
            md_type = description.get('md-type', '')
            if 'Langevin' in md_type:
                workflow.x_asap_langevin_friction = description.get('friction', 0)

    def write_to_archive(self) -> None:
        self.traj_parser.mainfile = self.mainfile

        # check if traj file is really asap
        if 'calculator' in self.traj_parser.traj.backend.keys():
            if self.traj_parser.traj.backend.calculator.name != 'emt':  # pylint: disable=E1101
                self.logger.error('Trajectory is not ASAP.')
                return

        super().write_to_archive()

        if self.archive.run:
            self.archive.run[0].program.name = 'ASAP'
