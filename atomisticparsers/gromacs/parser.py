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
import numpy as np
import logging
import re
import datetime

import panedr

try:
    import MDAnalysis
    from MDAnalysis.topology.tpr import utils as tpr_utils, setting as tpr_setting
except Exception:
    logging.warning("Required module MDAnalysis not found.")
    MDAnalysis = False
from ase.symbols import symbols2numbers
from nomad.units import ureg
from nomad.parsing.file_parser import TextParser, Quantity, FileParser
from runschema.run import Run, Program, TimeRun
from runschema.method import (
    NeighborSearching,
    ForceCalculations,
    Method,
    ForceField,
    Model,
    AtomParameters,
)
from runschema.system import AtomsGroup
from simulationworkflowschema import (
    GeometryOptimization,
    GeometryOptimizationMethod,
    GeometryOptimizationResults,
)
from .metainfo.gromacs import (
    x_gromacs_section_control_parameters,
    x_gromacs_section_input_output_files,
)
from atomisticparsers.utils import MDAnalysisParser, MDParser
from nomad.atomutils import get_bond_list_from_model_contributions
from nomad.parsing.parser import to_hdf5

re_float = r"[-+]?\d+\.*\d*(?:[Ee][-+]\d+)?"
re_n = r"[\n\r]"

MOL = 6.022140857e23


def to_float(string):
    try:
        value = float(string)
    except ValueError:
        value = None
    return value


class GromacsLogParser(TextParser):
    def __init__(self):
        super().__init__(None)

    def init_quantities(self):
        def str_to_header(val_in):
            val = [v.split(":", 1) for v in val_in.strip().splitlines()]
            return {v[0].strip(): v[1].strip() for v in val if len(v) == 2}

        def str_to_input_parameters(val_in):
            re_array = re.compile(r"\s*([\w\-]+)\[[\d ]+\]\s*=\s*\{*(.+)")
            re_scalar = re.compile(r"\s*([\w\-]+)\s*[=:]\s*(.+)")
            parameters = dict()
            val = val_in.strip().splitlines()
            for val_n in val:
                val_scalar = re_scalar.match(val_n)
                if val_scalar:
                    parameters[val_scalar.group(1)] = val_scalar.group(2)
                    continue
                val_array = re_array.match(val_n)
                if val_array:
                    parameters.setdefault(val_array.group(1), [])
                    value = [
                        float(v) for v in val_array.group(2).rstrip("}").split(",")
                    ]
                    parameters[val_array.group(1)].append(
                        value[0] if len(value) == 1 else value
                    )
            return parameters

        def str_to_energies(val_in):
            thermo_common = [
                r"Total Energy",
                r"Potential",
                r"Kinetic En.",
                r"Temperature",
                r"Pressure \(bar\)",
                r"LJ \(SR\)",
                r"Coulomb \(SR\)",
                r"Proper Dih.",
            ]
            n_chars_val = re.search(rf'( +{"| +".join(thermo_common)})', val_in)
            n_chars_val = len(n_chars_val.group(1)) if n_chars_val is not None else None
            if n_chars_val is None:
                n_chars_val = 15
            energies = {}
            rows = [v for v in val_in.splitlines() if v]
            for n in range(0, len(rows), 2):
                pointer = 0
                while pointer < len(rows[n]):
                    key = rows[n][pointer : pointer + n_chars_val].strip()
                    value = rows[n + 1][pointer : pointer + n_chars_val]
                    energies[key] = to_float(value)
                    pointer += n_chars_val
            return energies

        def str_to_step_info(val_in):
            val = val_in.strip().splitlines()
            keys = val[0].split()
            values = [to_float(v) for v in val[1].split()]
            return {key: values[n] for n, key in enumerate(keys)}

        thermo_quantities = [
            Quantity(
                "energies",
                r"Energies \(kJ/mol\).*\n(\s*[\s\S]+?)(?:\n.*step.* load imb.*|\n\n)",
                str_operation=str_to_energies,
                convert=False,
            ),
            Quantity(
                "step_info",
                rf"{re_n}\s*(Step.+\n[\d\.\- ]+)",
                str_operation=str_to_step_info,
                convert=False,
            ),
        ]

        self._quantities = [
            Quantity("time_start", r"Log file opened on (.+)", flatten=False),
            Quantity(
                "host_info",
                r"Host:\s*(\S+)\s*pid:\s*(\d+)\s*rank ID:\s*(\d+)\s*number of ranks:\s*(\d*)",
            ),
            Quantity(
                "module_version", r"GROMACS:\s*(.+?),\s*VERSION\s*(\S+)", flatten=False
            ),
            Quantity("execution_path", r"Executable:\s*(.+)"),
            Quantity("working_path", r"Data prefix:\s*(.+)"),
            # TODO cannot understand treatment of the command line in the old parser
            Quantity(
                "header",
                r"(?:GROMACS|Gromacs) (20[\s\S]+?)\n\n",
                str_operation=str_to_header,
            ),
            Quantity(
                "header",
                r"(?:GROMACS|Gromacs) (version:[\s\S]+?)\n\n",
                str_operation=str_to_header,
            ),
            Quantity(
                "input_parameters",
                r"Input Parameters:\s*([\s\S]+?)\n\n",
                str_operation=str_to_input_parameters,
            ),
            Quantity("maximum_force", r"Norm of force\s*([\s\S]+?)\n\n", flatten=False),
            Quantity(
                "step",
                r"(Step\s*Time[\s\S]+?Energies[\s\S]+?\n\n)",
                repeats=True,
                sub_parser=TextParser(quantities=thermo_quantities),
            ),
            Quantity(
                "averages",
                r"A V E R A G E S  ====>([\s\S]+?\n\n\n)",
                sub_parser=TextParser(quantities=thermo_quantities),
            ),
            Quantity("time_end", r"Finished \S+ on rank \d+ (.+)", flatten=False),
        ]


class GromacsMdpParser(TextParser):
    def __init__(self):
        super().__init__(None)

    def init_quantities(self):
        def str_to_input_parameters(val_in):
            re_array = re.compile(r"\s*([\w\-]+)\[[\d ]+\]\s*=\s*\{*(.+)")
            re_scalar = re.compile(r"\s*([\w\-]+)\s*[=:]\s*(.+)")
            parameters = dict()
            val = [line.strip() for line in val_in.splitlines()]
            for val_n in val:
                val_scalar = re_scalar.match(val_n)
                if val_scalar:
                    parameters[val_scalar.group(1)] = val_scalar.group(2)
                    continue
                val_array = re_array.match(val_n)
                if val_array:
                    parameters.setdefault(val_array.group(1), [])
                    value = [
                        to_float(v) for v in val_array.group(2).rstrip("}").split(",")
                    ]
                    parameters[val_array.group(1)].append(
                        value[0] if len(value) == 1 else value
                    )
            return parameters

        self._quantities = [
            Quantity(
                "input_parameters",
                r"([\s\S]+)",
                str_operation=str_to_input_parameters,
            ),
        ]


class GromacsXvgParser(TextParser):
    def __init__(self):
        super().__init__(None)
        self.re_columns = re.compile(r"@\s*s\d{1,2}\s*legend\s*\".*\"")
        self.re_comment = re.compile(r"^[@#]")
        self.re_quotes = re.compile(r"\"(.*)\"")
        self.re_label = re.compile(r'@\s*(title|xaxis|yaxis)\s*(?: label)?\s*"(.*)"')

        def str_to_results(val_in):
            results = {
                "column_vals": None,
                "title": "",
                "xaxis": "",
                "yaxis": "",
                "column_headers": [],
            }

            val = val_in.strip().splitlines()
            val = [line.strip() for line in val]
            for val_n in val:
                val_label = self.re_label.match(val_n)
                val_legend = self.re_columns.match(val_n)
                val_comment = self.re_comment.match(val_n)
                if val_label:
                    key, label = val_label.groups()
                    results[key] = label
                elif val_legend:  # TODO convert out of xmgrace notation
                    column = val_legend.group()
                    column = self.re_quotes.findall(column)
                    column = column[0] if column else None
                    results["column_headers"].append(column)
                elif not val_comment:
                    results["column_vals"] = (
                        np.vstack((results["column_vals"], [val_n.split()]))
                        if results["column_vals"] is not None
                        else [val_n.split()]
                    )
            return results

        self._quantities = [
            Quantity(
                "results",
                r"([\s\S]+)",
                str_operation=str_to_results,
            ),
        ]


class GromacsEDRParser(FileParser):
    def __init__(self):
        super().__init__(None)

    @property
    def fileedr(self):
        if self._file_handler is None:
            try:
                self._file_handler = panedr.edr_to_df(self.mainfile)
            except Exception:
                self.logger.error("Error reading edr file.")

        return self._file_handler

    def parse(self, key):
        if self.fileedr is None:
            return

        val = self.fileedr.get(key, None)
        if self._results is None:
            self._results = dict()

        if val is not None:
            val = np.asarray(val)

        self._results[key] = val

    def keys(self):
        return list(self.fileedr.keys())

    @property
    def length(self):
        return self.fileedr.shape[0]


class GromacsMDAnalysisParser(MDAnalysisParser):
    def __init__(self):
        super().__init__(None)

    def get_interactions(self):
        interactions = super().get_interactions()

        # add force field parameters
        try:
            interactions.extend(self.get_force_field_parameters())
        except Exception:
            self.logger.error("Error parsing force field parameters.")

        self._results["interactions"] = interactions

        return interactions

    def get_force_field_parameters(self):
        # read force field parameters not saved by MDAnalysis
        # copied from MDAnalysis.topology.tpr.utils
        # TODO maybe a better implementation exists
        if MDAnalysis.__version__ != "2.0.0":
            self.logger.warning("Incompatible version of MDAnalysis.")

        with open(self.mainfile, "rb") as f:
            data = tpr_utils.TPXUnpacker(f.read())

        interactions = []

        # read header
        header = tpr_utils.read_tpxheader(data)
        # address compatibility issue
        if header.fver >= tpr_setting.tpxv_AddSizeField and header.fgen >= 27:
            actual_body_size = len(data.get_buffer()) - data.get_position()
            if actual_body_size == 4 * header.sizeOfTprBody:
                self.logger.error("Unsupported tpr format.")
                return interactions
            data = tpr_utils.TPXUnpacker2020.from_unpacker(data)

        # read other unimportant parts
        if header.bBox:
            tpr_utils.extract_box_info(data, header.fver)
        if header.ngtc > 0:
            if header.fver < 69:
                tpr_utils.ndo_real(data, header.ngtc)
            tpr_utils.ndo_real(data, header.ngtc)
        if not header.bTop:
            return interactions

        tpr_utils.do_symstr(data, tpr_utils.do_symtab(data))
        data.unpack_int()
        ntypes = data.unpack_int()
        # functional types
        functypes = tpr_utils.ndo_int(data, ntypes)
        data.unpack_double() if header.fver >= 66 else 12.0
        data.unpack_real()
        # read the ffparams
        for i in functypes:
            parameters = []
            if i in [
                tpr_setting.F_ANGLES,
                tpr_setting.F_G96ANGLES,
                tpr_setting.F_BONDS,
                tpr_setting.F_G96BONDS,
                tpr_setting.F_HARMONIC,
                tpr_setting.F_IDIHS,
            ]:
                parameters.append(data.unpack_real())  # rA
                parameters.append(data.unpack_real())  # krA
                parameters.append(data.unpack_real())  # rB
                parameters.append(data.unpack_real())  # krB

            elif i in [tpr_setting.F_RESTRANGLES]:
                parameters.append(data.unpack_real())  # harmonic.rA
                parameters.append(data.unpack_real())  # harmonic.krA
            elif i in [tpr_setting.F_LINEAR_ANGLES]:
                parameters.append(data.unpack_real())  # linangle.klinA
                parameters.append(data.unpack_real())  # linangle.aA
                parameters.append(data.unpack_real())  # linangle.klinB
                parameters.append(data.unpack_real())  # linangle.aB);
            elif i in [tpr_setting.F_FENEBONDS]:
                parameters.append(data.unpack_real())  # fene.bm
                parameters.append(data.unpack_real())  # fene.kb
            elif i in [tpr_setting.F_RESTRBONDS]:
                parameters.append(data.unpack_real())  # restraint.lowA
                parameters.append(data.unpack_real())  # restraint.up1A
                parameters.append(data.unpack_real())  # restraint.up2A
                parameters.append(data.unpack_real())  # restraint.kA
                parameters.append(data.unpack_real())  # restraint.lowB
                parameters.append(data.unpack_real())  # restraint.up1B
                parameters.append(data.unpack_real())  # restraint.up2B
                parameters.append(data.unpack_real())  # restraint.kB
            elif i in [
                tpr_setting.F_TABBONDS,
                tpr_setting.F_TABBONDSNC,
                tpr_setting.F_TABANGLES,
                tpr_setting.F_TABDIHS,
            ]:
                parameters.append(data.unpack_real())  # tab.kA
                parameters.append(data.unpack_int())  # tab.table
                parameters.append(data.unpack_real())  # tab.kB
            elif i in [tpr_setting.F_CROSS_BOND_BONDS]:
                parameters.append(data.unpack_real())  # cross_bb.r1e
                parameters.append(data.unpack_real())  # cross_bb.r2e
                parameters.append(data.unpack_real())  # cross_bb.krr
            elif i in [tpr_setting.F_CROSS_BOND_ANGLES]:
                parameters.append(data.unpack_real())  # cross_ba.r1e
                parameters.append(data.unpack_real())  # cross_ba.r2e
                parameters.append(data.unpack_real())  # cross_ba.r3e
                parameters.append(data.unpack_real())  # cross_ba.krt
            elif i in [tpr_setting.F_UREY_BRADLEY]:
                parameters.append(data.unpack_real())  # u_b.theta
                parameters.append(data.unpack_real())  # u_b.ktheta
                parameters.append(data.unpack_real())  # u_b.r13
                parameters.append(data.unpack_real())  # u_b.kUB
                if header.fver >= 79:
                    parameters.append(data.unpack_real())  # u_b.thetaB
                    parameters.append(data.unpack_real())  # u_b.kthetaB
                    parameters.append(data.unpack_real())  # u_b.r13B
                    parameters.append(data.unpack_real())  # u_b.kUBB
            elif i in [tpr_setting.F_QUARTIC_ANGLES]:
                parameters.append(data.unpack_real())  # qangle.theta
                parameters.append(tpr_utils.ndo_real(data, 5))  # qangle.c
            elif i in [tpr_setting.F_BHAM]:
                parameters.append(data.unpack_real())  # bham.a
                parameters.append(data.unpack_real())  # bham.b
                parameters.append(data.unpack_real())  # bham.c
            elif i in [tpr_setting.F_MORSE]:
                parameters.append(data.unpack_real())  # morse.b0
                parameters.append(data.unpack_real())  # morse.cb
                parameters.append(data.unpack_real())  # morse.beta
                if header.fver >= 79:
                    parameters.append(data.unpack_real())  # morse.b0B
                    parameters.append(data.unpack_real())  # morse.cbB
                    parameters.append(data.unpack_real())  # morse.betaB
            elif i in [tpr_setting.F_CUBICBONDS]:
                parameters.append(data.unpack_real())  # cubic.b0g
                parameters.append(data.unpack_real())  # cubic.kb
                parameters.append(data.unpack_real())  # cubic.kcub
            elif i in [tpr_setting.F_CONNBONDS]:
                pass
            elif i in [tpr_setting.F_POLARIZATION]:
                parameters.append(data.unpack_real())  # polarize.alpha
            elif i in [tpr_setting.F_ANHARM_POL]:
                parameters.append(data.unpack_real())  # anharm_polarize.alpha
                parameters.append(data.unpack_real())  # anharm_polarize.drcut
                parameters.append(data.unpack_real())  # anharm_polarize.khyp
            elif i in [tpr_setting.F_WATER_POL]:
                parameters.append(data.unpack_real())  # wpol.al_x
                parameters.append(data.unpack_real())  # wpol.al_y
                parameters.append(data.unpack_real())  # wpol.al_z
                parameters.append(data.unpack_real())  # wpol.rOH
                parameters.append(data.unpack_real())  # wpol.rHH
                parameters.append(data.unpack_real())  # wpol.rOD
            elif i in [tpr_setting.F_THOLE_POL]:
                parameters.append(data.unpack_real())  # thole.a
                parameters.append(data.unpack_real())  # thole.alpha1
                parameters.append(data.unpack_real())  # thole.alpha2
                parameters.append(data.unpack_real())  # thole.rfac

            elif i in [tpr_setting.F_LJ]:
                parameters.append(data.unpack_real())  # lj_c6
                parameters.append(data.unpack_real())  # lj_c9
            elif i in [tpr_setting.F_LJ14]:
                parameters.append(data.unpack_real())  # lj14_c6A
                parameters.append(data.unpack_real())  # lj14_c12A
                parameters.append(data.unpack_real())  # lj14_c6B
                parameters.append(data.unpack_real())  # lj14_c12B
            elif i in [tpr_setting.F_LJC14_Q]:
                parameters.append(data.unpack_real())  # ljc14.fqq
                parameters.append(data.unpack_real())  # ljc14.qi
                parameters.append(data.unpack_real())  # ljc14.qj
                parameters.append(data.unpack_real())  # ljc14.c6
                parameters.append(data.unpack_real())  # ljc14.c12
            elif i in [tpr_setting.F_LJC_PAIRS_NB]:
                parameters.append(data.unpack_real())  # ljcnb.qi
                parameters.append(data.unpack_real())  # ljcnb.qj
                parameters.append(data.unpack_real())  # ljcnb.c6
                parameters.append(data.unpack_real())  # ljcnb.c12

            elif i in [
                tpr_setting.F_PIDIHS,
                tpr_setting.F_ANGRES,
                tpr_setting.F_ANGRESZ,
                tpr_setting.F_PDIHS,
            ]:
                parameters.append(data.unpack_real())  # pdihs_phiA
                parameters.append(data.unpack_real())  # pdihs_cpA
                parameters.append(data.unpack_real())  # pdihs_phiB
                parameters.append(data.unpack_real())  # pdihs_cpB
                parameters.append(data.unpack_int())  # pdihs_mult

            elif i in [tpr_setting.F_RESTRDIHS]:
                parameters.append(data.unpack_real())  # pdihs.phiA
                parameters.append(data.unpack_real())  # pdihs.cpA
            elif i in [tpr_setting.F_DISRES]:
                parameters.append(data.unpack_int())  # disres.label
                parameters.append(data.unpack_int())  # disres.type
                parameters.append(data.unpack_real())  # disres.low
                parameters.append(data.unpack_real())  # disres.up1
                parameters.append(data.unpack_real())  # disres.up2
                parameters.append(data.unpack_real())  # disres.kfac

            elif i in [tpr_setting.F_ORIRES]:
                parameters.append(data.unpack_int())  # orires.ex
                parameters.append(data.unpack_int())  # orires.label
                parameters.append(data.unpack_int())  # orires.power
                parameters.append(data.unpack_real())  # orires.c
                parameters.append(data.unpack_real())  # orires.obs
                parameters.append(data.unpack_real())  # orires.kfac

            elif i in [tpr_setting.F_DIHRES]:
                if header.fver < 72:
                    parameters.append(data.unpack_int())  # idum
                    parameters.append(data.unpack_int())  # idum
                parameters.append(data.unpack_real())  # dihres.phiA
                parameters.append(data.unpack_real())  # dihres.dphiA
                parameters.append(data.unpack_real())  # dihres.kfacA
                if header.fver >= 72:
                    parameters.append(data.unpack_real())  # dihres.phiB
                    parameters.append(data.unpack_real())  # dihres.dphiB
                    parameters.append(data.unpack_real())  # dihres.kfacB

            elif i in [tpr_setting.F_POSRES]:
                parameters.append(tpr_utils.do_rvec(data))  # posres.pos0A
                parameters.append(tpr_utils.do_rvec(data))  # posres.fcA
                parameters.append(tpr_utils.do_rvec(data))  # posres.pos0B
                parameters.append(tpr_utils.do_rvec(data))  # posres.fcB

            elif i in [tpr_setting.F_FBPOSRES]:
                parameters.append(data.unpack_int())  # fbposres.geom
                parameters.append(tpr_utils.do_rvec(data))  # fbposres.pos0
                parameters.append(data.unpack_real())  # fbposres.r
                parameters.append(data.unpack_real())  # fbposres.k

            elif i in [tpr_setting.F_CBTDIHS]:
                parameters.append(
                    tpr_utils.ndo_real(data, tpr_setting.NR_CBTDIHS)
                )  # cbtdihs.cbtcA

            elif i in [tpr_setting.F_RBDIHS]:
                parameters.append(
                    tpr_utils.ndo_real(data, tpr_setting.NR_RBDIHS)
                )  # iparams_rbdihs_rbcA
                parameters.append(
                    tpr_utils.ndo_real(data, tpr_setting.NR_RBDIHS)
                )  # iparams_rbdihs_rbcB

            elif i in [tpr_setting.F_FOURDIHS]:
                # Fourier dihedrals
                parameters.append(
                    tpr_utils.ndo_real(data, tpr_setting.NR_RBDIHS)
                )  # rbdihs.rbcA
                parameters.append(
                    tpr_utils.ndo_real(data, tpr_setting.NR_RBDIHS)
                )  # rbdihs.rbcB

            elif i in [tpr_setting.F_CONSTR, tpr_setting.F_CONSTRNC]:
                parameters.append(data.unpack_real())  # dA
                parameters.append(data.unpack_real())  # dB

            elif i in [tpr_setting.F_SETTLE]:
                parameters.append(data.unpack_real())  # settle.doh
                parameters.append(data.unpack_real())  # settle.dhh

            elif i in [tpr_setting.F_VSITE1]:
                pass

            elif i in [tpr_setting.F_VSITE2, tpr_setting.F_VSITE2FD]:
                parameters.append(data.unpack_real())  # vsite.a

            elif i in [
                tpr_setting.F_VSITE3,
                tpr_setting.F_VSITE3FD,
                tpr_setting.F_VSITE3FAD,
            ]:
                parameters.append(data.unpack_real())  # vsite.a

            elif i in [
                tpr_setting.F_VSITE3OUT,
                tpr_setting.F_VSITE4FD,
                tpr_setting.F_VSITE4FDN,
            ]:
                parameters.append(data.unpack_real())  # vsite.a
                parameters.append(data.unpack_real())  # vsite.b
                parameters.append(data.unpack_real())  # vsite.c

            elif i in [tpr_setting.F_VSITEN]:
                parameters.append(data.unpack_int())  # vsiten.n
                parameters.append(data.unpack_real())  # vsiten.a

            elif i in [tpr_setting.F_GB12, tpr_setting.F_GB13, tpr_setting.F_GB14]:
                # /* We got rid of some parameters in version 68 */
                if header.fver < 68:
                    parameters.append(data.unpack_real())  # rdum
                    parameters.append(data.unpack_real())  # rdum
                    parameters.append(data.unpack_real())  # rdum
                    parameters.append(data.unpack_real())  # rdum
                parameters.append(data.unpack_real())  # gb.sar
                parameters.append(data.unpack_real())  # gb.st
                parameters.append(data.unpack_real())  # gb.pi
                parameters.append(data.unpack_real())  # gb.gbr
                parameters.append(data.unpack_real())  # gb.bmlt

            elif i in [tpr_setting.F_CMAP]:
                parameters.append(data.unpack_int())  # cmap.cmapA
                parameters.append(data.unpack_int())  # cmap.cmapB
            else:
                raise NotImplementedError(f"unknown functype: {i}")
            interactions.append(
                dict(type=tpr_setting.interaction_types[i][1], parameters=parameters)
            )

        return interactions


class GromacsParser(MDParser):
    def __init__(self):
        self.log_parser = GromacsLogParser()
        self.traj_parser = GromacsMDAnalysisParser()
        self.energy_parser = GromacsEDRParser()
        self.mdp_parser = GromacsMdpParser()
        self.mdp_ext = "mdp"
        self.mdp_std_filename = "mdout"
        self.xvg_parser = GromacsXvgParser()
        self.input_parameters = {}
        self._gro_energy_units = ureg.kilojoule * MOL
        self._thermo_ignore_list = ["Time", "Box-X", "Box-Y", "Box-Z"]
        self._base_calc_map = {
            "Temperature": ("temperature", ureg.kelvin),
            "Volume": ("volume", ureg.nm**3),
            "Density": ("density", ureg.kilogram / ureg.m**3),
            "Pressure (bar)": ("pressure", ureg.bar),
            "Pressure": ("pressure", ureg.bar),
            "Enthalpy": ("enthalpy", self._gro_energy_units),
        }
        self._energy_map = {
            "Potential": "potential",
            "Kinetic En.": "kinetic",
            "Total Energy": "total",
            "pV": "pressure_volume_work",
        }
        self._vdw_map = {
            "LJ (SR)": "short_range",
            "LJ (LR)": "long_range",
            "Disper. corr.": "correction",
        }
        self._electrostatic_map = {
            "Coulomb (SR)": "short_range",
            "Coul. recip.": "long_range",
        }
        self._energy_keys_contain = [
            "bond",
            "angle",
            "dih.",
            "coul-",
            "coulomb-",
            "lj-",
            "en.",
        ]
        super().__init__()

    def get_pbc(self):
        pbc = self.input_parameters.get("pbc", "xyz")
        return ["x" in pbc, "y" in pbc, "z" in pbc]

    def get_mdp_file(self):
        """
        Tries to find the mdp input parameters (ext = mdp) that match the mainfile calculation.
        Priority is as follows:
            1. output mdp file containing both the matching mainfile name and the standard
            gromacs name `mdout`
            2. file containing the standard gromacs name `mdout`
            3. input mdp file matching the mainfile name (as usual)
            4. any `.mdp` file within the directory (as usual)
        """
        files = [d for d in self._gromacs_files if d.endswith(self.mdp_ext)]

        if len(files) == 0:
            return ""

        if len(files) == 1:
            return os.path.join(self._maindir, files[0])

        for f in files:
            filename = f.rsplit(".", 1)[0]
            if self._basename in filename and self.mdp_std_filename in filename:
                return os.path.join(self._maindir, f)

        for f in files:
            filename = f.rsplit(".", 1)[0]
            if self.mdp_std_filename in filename:
                return os.path.join(self._maindir, f)

        return self.get_gromacs_file(self.mdp_ext)

    def get_gromacs_file(self, ext):
        files = [d for d in self._gromacs_files if d.endswith(ext)]

        if len(files) == 0:
            return ""

        if len(files) == 1:
            return os.path.join(self._maindir, files[0])

        # we assume that the file has the same basename as the log file e.g.
        # out.log would correspond to out.tpr and out.trr and out.edr
        for f in files:
            if f.rsplit(".", 1)[0] == self._basename:
                return os.path.join(self._maindir, f)

        for f in files:
            if f.rsplit(".", 1)[0].startswith(self._basename):
                return os.path.join(self._maindir, f)

        # if the files are all named differently, we guess that the one that does not
        # share the same basename would be file we are interested in
        # e.g. in a list of files out.log someout.log out.tpr out.trr another.tpr file.trr
        # we guess that the out.* files belong together and the rest that does not share
        # a basename would be grouped together
        counts = []
        for f in files:
            count = 0
            for reff in self._gromacs_files:
                if f.rsplit(".", 1)[0] == reff.rsplit(".", 1)[0]:
                    count += 1
            if count == 1:
                return os.path.join(self._maindir, f)
            counts.append(count)

        return os.path.join(self._maindir, files[counts.index(min(counts))])

    def parse_thermodynamic_data(self):
        sec_run = self.archive.run[-1]

        n_frames = self.traj_parser.get("n_frames")

        # # TODO read also from ene
        edr_file = self.get_gromacs_file("edr")
        self.energy_parser.mainfile = edr_file

        # get it from edr file
        if self.energy_parser.keys():
            thermo_data = self.energy_parser
        else:
            # try to get it from log file
            steps = self.input_parameters.get("step", [])
            thermo_data = dict()
            for n, step in enumerate(steps):
                n = int(step.get("step_info", {}).get("Step", n))
                if step.energies is None:
                    continue
                keys = step.energies.keys()
                for key in keys:
                    thermo_data.setdefault(key, [None] * n_frames)
                    thermo_data[key][n] = step.energies.get(key)
                info = step.get("step_info", {})
                thermo_data.setdefault("Time", [None] * n_frames)
                thermo_data["Time"][n] = info.get("Time", None)

        if not thermo_data:
            # get it from edr file
            thermo_data = self.energy_parser

        calculation_times = thermo_data.get("Time", [])
        time_step = self.input_parameters.get("dt")
        if time_step is None and len(calculation_times) > 1:
            time_step = calculation_times[1] - calculation_times[0]
        self.thermodynamics_steps = [
            int(time / time_step if time_step else 1) for time in calculation_times
        ]

        for n, step in enumerate(self.thermodynamics_steps):
            data = {
                "step": step,
                "time": calculation_times[n] * ureg.picosecond,
                "method_ref": sec_run.method[-1] if sec_run.method else None,
                "energy": {},
            }
            if step in self._trajectory_steps:
                data["forces"] = dict(
                    total=dict(
                        value=self.traj_parser.get_forces(
                            self._trajectory_steps.index(step)
                        )
                    )
                )

            pressure_tensor, virial_tensor = None, None
            for key in thermo_data.keys():
                if (
                    key in self._thermo_ignore_list
                    or (val := thermo_data.get(key)[n]) is None
                ):
                    continue

                # Attributes of BaseCalculation
                if key in self._base_calc_map:
                    data[self._base_calc_map[key][0]] = (
                        val * self._base_calc_map[key][1]
                    )

                # pressure tensor
                elif match := re.match(r"Pres-([XYZ]{2})", key):
                    if pressure_tensor is None:
                        pressure_tensor = np.zeros(shape=(3, 3))
                    pressure_tensor[tuple("XYZ".index(n) for n in match.group(1))] = val

                # virial tensor
                elif match := re.match(r"Vir-([XYZ]{2})", key):
                    if virial_tensor is None:
                        virial_tensor = np.zeros(shape=(3, 3))
                    virial_tensor[tuple("XYZ".index(n) for n in match.group(1))] = val

                # well-defined, single Energy quantities
                elif (nomad_key := self._energy_map.get(key)) is not None:
                    data["energy"][nomad_key] = dict(value=val * self._gro_energy_units)
                # well-defined, piecewise energy quantities
                elif (nomad_key := self._vdw_map.get(key)) is not None:
                    data["energy"].setdefault(
                        "van_der_waals", {"value": 0.0 * self._gro_energy_units}
                    )
                    data["energy"]["van_der_waals"][nomad_key] = (
                        val * self._gro_energy_units
                    )
                    data["energy"]["van_der_waals"]["value"] += (
                        val * self._gro_energy_units
                    )
                elif (nomad_key := self._electrostatic_map.get(key)) is not None:
                    data["energy"].setdefault(
                        "electrostatic", {"value": 0.0 * self._gro_energy_units}
                    )
                    data["energy"]["electrostatic"][nomad_key] = (
                        val * self._gro_energy_units
                    )
                    data["energy"]["electrostatic"]["value"] += (
                        val * self._gro_energy_units
                    )
                # try to identify other known energy keys to be stored as gromacs-specific
                elif any(
                    keyword in key.lower() for keyword in self._energy_keys_contain
                ):
                    data["energy"].setdefault("x_gromacs_energy_contributions", [])
                    data["energy"]["x_gromacs_energy_contributions"].append(
                        dict(kind=key, value=val * self._gro_energy_units)
                    )
                else:  # store all other quantities as gromacs-specific under BaseCalculation
                    data.setdefault("x_gromacs_thermodynamics_contributions", [])
                    data["x_gromacs_thermodynamics_contributions"].append(
                        dict(kind=key, value=val)
                    )

            if pressure_tensor is not None:
                data["pressure_tensor"] = pressure_tensor * ureg.bar

            if virial_tensor is not None:
                data["virial_tensor"] = virial_tensor * (ureg.bar * ureg.nm**3)

            self.parse_thermodynamics_step(data)

    def parse_system(self):
        sec_run = self.archive.run[-1]

        def get_composition(children_names):
            children_count_tup = np.unique(children_names, return_counts=True)
            formula = "".join(
                [f"{name}({count})" for name, count in zip(*children_count_tup)]
            )
            return formula

        n_frames = self.traj_parser.get("n_frames", 0)
        traj_sampling_rate = self.input_parameters.get("nstxout", 1)
        self.n_atoms = [self.traj_parser.get_n_atoms(n) for n in range(n_frames)]
        traj_steps = [n * traj_sampling_rate for n in range(n_frames)]
        self.trajectory_steps = traj_steps

        pbc = self.get_pbc()
        self._system_time_map = {}
        for step in self.trajectory_steps:
            n = traj_steps.index(step)
            positions = self.traj_parser.get_positions(n)
            if positions is None:
                continue

            bond_list = []
            if n == 0:  # TODO add references to the bond list for other steps
                bond_list = get_bond_list_from_model_contributions(
                    sec_run, method_index=-1, model_index=-1
                )

            atom_labels = self.traj_parser.get_atom_labels(n)
            if atom_labels is not None:
                try:
                    symbols2numbers(atom_labels)
                except KeyError:
                    atom_labels = ["X"] * len(atom_labels)

            self.parse_trajectory_step(
                {
                    "atoms": {
                        "n_atoms": self.traj_parser.get_n_atoms(n),
                        "periodic": pbc,
                        "lattice_vectors": self.traj_parser.get_lattice_vectors(n),
                        "labels": atom_labels,
                        "positions": positions,
                        "velocities": self.traj_parser.get_velocities(n),
                        "bond_list": bond_list,
                    }
                }
            )

        if not sec_run.system:
            return

        # parse atomsgroup (segments --> molecules --> residues)
        atoms_info = self.traj_parser._results["atoms_info"]
        atoms_moltypes = np.array(atoms_info["moltypes"])
        atoms_molnums = np.array(atoms_info["molnums"])
        atoms_resids = np.array(atoms_info["resids"])
        atoms_elements = np.array(atoms_info["elements"])
        atoms_resnames = np.array(atoms_info["resnames"])
        for segment in self.traj_parser.universe.segments:
            # we only create atomsgroup in the initial system
            sec_segment = AtomsGroup()
            sec_run.system[0].atoms_group.append(sec_segment)
            sec_segment.type = "molecule_group"
            sec_segment.index = int(segment.segindex)
            sec_segment.atom_indices = segment.atoms.ix
            sec_segment.n_atoms = len(sec_segment.atom_indices)
            sec_segment.is_molecule = False

            moltypes = np.unique(atoms_moltypes[sec_segment.atom_indices])
            moltypes_count = {}
            for moltype in moltypes:
                atom_indices = np.where(atoms_moltypes == moltype)[0]
                # mol_nums is the molecule identifier for each atom
                mol_nums = atoms_molnums[atom_indices]
                moltypes_count[moltype] = np.unique(mol_nums).shape[0]
            formula = "".join(
                [f"{moltype}({moltypes_count[moltype]})" for moltype in moltypes_count]
            )
            sec_segment.composition_formula = formula
            sec_segment.label = f"group_{moltypes[0]}"

            for i_molecule, molecule in enumerate(
                np.unique(atoms_molnums[sec_segment.atom_indices])
            ):
                sec_molecule = AtomsGroup()
                sec_segment.atoms_group.append(sec_molecule)
                sec_molecule.index = i_molecule
                sec_molecule.atom_indices = np.where(atoms_molnums == molecule)[0]
                sec_molecule.n_atoms = len(sec_molecule.atom_indices)
                # use first particle to get the moltype
                # not sure why but this value is being cast to int, cast back to str
                sec_molecule.label = str(atoms_moltypes[sec_molecule.atom_indices[0]])
                sec_molecule.type = "molecule"
                sec_molecule.is_molecule = True

                mol_resids = np.unique(atoms_resids[sec_molecule.atom_indices])
                n_res = mol_resids.shape[0]
                if n_res == 1:
                    elements = atoms_elements[sec_molecule.atom_indices]
                    sec_molecule.composition_formula = get_composition(elements)
                else:
                    mol_resnames = atoms_resnames[sec_molecule.atom_indices]
                    restypes = np.unique(mol_resnames)
                    for i_restype, restype in enumerate(restypes):
                        sec_monomer_group = AtomsGroup()
                        sec_molecule.atoms_group.append(sec_monomer_group)
                        restype_indices = np.where(atoms_resnames == restype)[0]
                        sec_monomer_group.label = f"group_{restype}"
                        sec_monomer_group.type = "monomer_group"
                        sec_monomer_group.index = i_restype
                        sec_monomer_group.atom_indices = np.intersect1d(
                            restype_indices, sec_molecule.atom_indices
                        )
                        sec_monomer_group.n_atoms = len(sec_monomer_group.atom_indices)
                        sec_monomer_group.is_molecule = False

                        restype_resids = np.unique(
                            atoms_resids[sec_monomer_group.atom_indices]
                        )
                        restype_count = restype_resids.shape[0]
                        sec_monomer_group.composition_formula = (
                            f"{restype}({restype_count})"
                        )
                        for i_res, res_id in enumerate(restype_resids):
                            sec_residue = AtomsGroup()
                            sec_monomer_group.atoms_group.append(sec_residue)
                            sec_residue.index = i_res
                            atom_indices = np.where(atoms_resids == res_id)[0]
                            sec_residue.atom_indices = np.intersect1d(
                                atom_indices, sec_monomer_group.atom_indices
                            )
                            sec_residue.n_atoms = len(sec_residue.atom_indices)
                            sec_residue.label = str(restype)
                            sec_residue.type = "monomer"
                            sec_residue.is_molecule = False
                            elements = atoms_elements[sec_residue.atom_indices]
                            sec_residue.composition_formula = get_composition(elements)

                    names = atoms_resnames[sec_molecule.atom_indices]
                    ids = atoms_resids[sec_molecule.atom_indices]
                    # filter for the first instance of each residue, as to not overcount
                    __, ids_count = np.unique(ids, return_counts=True)
                    # get the index of the first atom of each residue
                    ids_firstatom = np.cumsum(ids_count)[:-1]
                    # add the 0th index manually
                    ids_firstatom = np.insert(ids_firstatom, 0, 0)
                    names_firstatom = names[ids_firstatom]
                    sec_molecule.composition_formula = get_composition(names_firstatom)

    def parse_method(self):
        sec_method = Method()
        self.archive.run[-1].method.append(sec_method)
        sec_force_field = ForceField()
        sec_method.force_field = sec_force_field
        sec_model = Model()
        sec_force_field.model.append(sec_model)
        try:
            n_atoms = self.traj_parser.get("n_atoms", 0)
        except Exception:
            gro_file = self.get_gromacs_file("gro")
            self.traj_parser.mainfile = gro_file
            n_atoms = self.traj_parser.get("n_atoms", 0)

        atoms_info = self.traj_parser.get("atoms_info", {})
        for n in range(n_atoms):
            sec_atom = AtomParameters()
            sec_method.atom_parameters.append(sec_atom)
            sec_atom.charge = atoms_info.get("charges", [None] * (n + 1))[n]
            sec_atom.mass = atoms_info.get("masses", [None] * (n + 1))[n]
            sec_atom.label = atoms_info.get("names", [None] * (n + 1))[n]
            sec_atom.x_gromacs_atom_name = atoms_info.get(
                "atom_names", [None] * (n + 1)
            )[n]
            sec_atom.x_gromacs_atom_resid = atoms_info.get("resids", [None] * (n + 1))[
                n
            ]
            sec_atom.x_gromacs_atom_resname = atoms_info.get(
                "resnames", [None] * (n + 1)
            )[n]
            sec_atom.x_gromacs_atom_molnum = atoms_info.get(
                "molnums", [None] * (n + 1)
            )[n]
            sec_atom.x_gromacs_atom_moltype = atoms_info.get(
                "moltypes", [None] * (n + 1)
            )[n]

        if n_atoms == 0:
            self.logger.error("Error parsing interactions.")

        interactions = self.traj_parser.get_interactions()
        self.parse_interactions(interactions, sec_model)

        input_parameters = self.input_parameters
        sec_force_calculations = ForceCalculations()
        sec_force_field.force_calculations = sec_force_calculations
        sec_neighbor_searching = NeighborSearching()
        sec_force_calculations.neighbor_searching = sec_neighbor_searching

        nstlist = input_parameters.get("nstlist", None)
        sec_neighbor_searching.neighbor_update_frequency = (
            int(nstlist) if nstlist else None
        )
        rlist = input_parameters.get("rlist", None)
        sec_neighbor_searching.neighbor_update_cutoff = (
            to_float(rlist) * ureg.nanometer if to_float(rlist) else None
        )
        rvdw = input_parameters.get("rvdw", None)
        sec_force_calculations.vdw_cutoff = (
            to_float(rvdw) * ureg.nanometer if to_float(rvdw) else None
        )
        coulombtype = input_parameters.get("coulombtype", "no").lower()
        coulombtype_map = {
            "cut-off": "cutoff",
            "ewald": "ewald",
            "pme": "particle_mesh_ewald",
            "p3m-ad": "particle_particle_particle_mesh",
            "reaction-field": "reaction_field",
            "shift": "cutoff",
            "switch": "cutoff",
            "user": "cutoff",
        }
        value = coulombtype_map.get(
            coulombtype,
            [val for key, val in coulombtype_map.items() if key in coulombtype],
        )
        value = (
            value
            if not isinstance(value, list)
            else value[0]
            if len(value) != 0
            else None
        )
        sec_force_calculations.coulomb_type = value
        rcoulomb = input_parameters.get("rcoulomb", None)
        sec_force_calculations.coulomb_cutoff = (
            to_float(rcoulomb) if to_float(rcoulomb) else None
        )

    def get_thermostat_parameters(self, integrator: str = ""):
        thermostat = self.input_parameters.get("tcoupl", "no").lower()
        thermostat_map = {
            "berendsen": "berendsen",
            "v-rescale": "velocity_rescaling",
            "nose-hoover": "nose_hoover",
            "andersen": "andersen",
        }
        value = thermostat_map.get(
            thermostat,
            [val for key, val in thermostat_map.items() if key in thermostat],
        )
        value = (
            value
            if not isinstance(value, list)
            else value[0]
            if len(value) != 0
            else None
        )
        thermostat_parameters = {}
        thermostat_parameters["thermostat_type"] = value
        if "sd" in integrator:
            thermostat_parameters["thermostat_type"] = "langevin_goga"
        if thermostat_parameters["thermostat_type"]:
            reference_temperature = self.input_parameters.get("ref-t", None)
            if isinstance(reference_temperature, str):
                reference_temperature = to_float(
                    reference_temperature.split()[0]
                )  # ! simulated annealing protocols not supported
            reference_temperature *= ureg.kelvin if reference_temperature else None
            thermostat_parameters["reference_temperature"] = reference_temperature
            coupling_constant = self.input_parameters.get("tau-t", None)
            if isinstance(coupling_constant, str):
                coupling_constant = to_float(
                    coupling_constant.split()[0]
                )  # ! simulated annealing protocols not supported
            coupling_constant *= ureg.picosecond if coupling_constant else None
            thermostat_parameters["coupling_constant"] = coupling_constant

        return thermostat_parameters

    def get_barostat_parameters(self):
        barostat_parameters = {}
        barostat_map = {
            "berendsen": "berendsen",
            "parrinello-rahman": "parrinello_rahman",
            "mttk": "martyna_tuckerman_tobias_klein",
            "c-rescale": "stochastic_cell_rescaling",
        }
        barostat = self.input_parameters.get("pcoupl", "no").lower()
        value = barostat_map.get(
            barostat, [val for key, val in barostat_map.items() if key in barostat]
        )
        value = (
            value
            if not isinstance(value, list)
            else value[0]
            if len(value) != 0
            else None
        )
        barostat_parameters["barostat_type"] = value
        if barostat_parameters["barostat_type"]:
            couplingtype = self.input_parameters.get("pcoupltype", None).lower()
            couplingtype_map = {
                "isotropic": "isotropic",
                "semiisotropic": "semi_isotropic",
                "anisotropic": "anisotropic",
            }
            value = couplingtype_map.get(
                couplingtype,
                [val for key, val in couplingtype_map.items() if key in couplingtype],
            )
            barostat_parameters["coupling_type"] = (
                value[0] if isinstance(value, list) else value
            )
            taup = self.input_parameters.get("tau-p", None)
            barostat_parameters["coupling_constant"] = (
                np.ones(shape=(3, 3)) * to_float(taup) * ureg.picosecond
                if to_float(taup)
                else None
            )
            refp = self.input_parameters.get("ref-p", None)
            barostat_parameters["reference_pressure"] = (
                refp * ureg.bar if refp is not None else None
            )
            compressibility = self.input_parameters.get("compressibility", None)
            barostat_parameters["compressibility"] = (
                compressibility * (1.0 / ureg.bar)
                if compressibility is not None
                else None
            )
        return barostat_parameters

    def get_free_energy_calculation_parameters(self):
        free_energy_parameters = {}
        free_energy = self.input_parameters.get("free-energy", "")
        free_energy = free_energy.lower() if free_energy else ""
        expanded = self.input_parameters.get("expanded", "")
        expanded = expanded.lower() if expanded else ""
        delta_lambda = int(self.input_parameters.get("delta-lamda", -1))
        if free_energy == "yes" and expanded == "yes":
            self.logger.warning(
                "storage of expanded ensemble simulation data not supported, skipping storage of free energy calculation parameters"
            )
        elif free_energy == "yes" and delta_lambda == "no":
            self.logger.warning(
                "Only fixed state free energy calculation calculations are explicitly supported, skipping storage of free energy calculation parameters."
            )
        elif free_energy == "yes":
            free_energy_parameters["type"] = "alchemical"
            lambda_key_map = {
                "fep": "output",
                "coul": "coulomb",
                "vdw": "vdw",
                "bonded": "bonded",
                "restraint": "restraint",
                "mass": "mass",
                "temperature": "temperature",
            }
            lambdas = {
                key: self.input_parameters.get(f"{key}-lambdas", "")
                for key in lambda_key_map.keys()
            }
            lambdas = {
                key: [to_float(i) for i in val.split()] for key, val in lambdas.items()
            }
            free_energy_parameters["lambdas"] = [
                {"kind": nomad_key, "value": lambdas[gromacs_key]}
                for gromacs_key, nomad_key in lambda_key_map.items()
                if lambdas[gromacs_key]
            ]
            free_energy_parameters["lambda_index"] = self.input_parameters.get(
                "init-lambda-state", ""
            )

            atoms_info = self.traj_parser._results["atoms_info"]
            atoms_moltypes = np.array(atoms_info["moltypes"])
            couple_moltype = self.input_parameters.get("couple-moltype", "").split()
            n_atoms = len(atoms_moltypes)
            indices = []
            if len(couple_moltype) == 1 and couple_moltype[0].lower() == "system":
                indices.extend(range(n_atoms))
            else:
                for moltype in couple_moltype:
                    indices.extend(
                        [
                            index
                            for index in range(n_atoms)
                            if atoms_moltypes[index].lower() == moltype
                        ]
                    )
            free_energy_parameters["atom_indices"] = indices

            couple_vdw_map = {"vdw-q": "on", "vdw": "on", "q": "off", "none": "off"}
            couple_coloumb_map = {
                "vdw-q": "on",
                "vdw": "off",
                "q": "on",
                "none": "off",
            }
            couple_initial = self.input_parameters.get("couple-lambda0", "none").lower()
            couple_final = self.input_parameters.get("couple-lambda1", "vdw-q").lower()

            free_energy_parameters["initial_state_vdw"] = couple_vdw_map[couple_initial]
            free_energy_parameters["final_state_vdw"] = couple_vdw_map[couple_final]
            free_energy_parameters["initial_state_coloumb"] = couple_coloumb_map[
                couple_initial
            ]
            free_energy_parameters["final_state_coloumb"] = couple_coloumb_map[
                couple_final
            ]

            couple_intramolecular = self.input_parameters.get(
                "couple-intramol", "on"
            ).lower()
            free_energy_parameters["final_state_bonded"] = "on"
            free_energy_parameters["initial_state_bonded"] = (
                "off" if couple_intramolecular == "yes" else "on"
            )
        return free_energy_parameters

    def parse_workflow(self):
        sec_run = self.archive.run[-1]
        sec_calc = sec_run.get("calculation")
        input_parameters = self.input_parameters

        workflow = None
        integrator = input_parameters.get("integrator", "md").lower()
        if integrator in ["l-bfgs", "cg", "steep"]:
            workflow = GeometryOptimization(
                method=GeometryOptimizationMethod(),
                results=GeometryOptimizationResults(),
            )
            workflow.method.type = "atomic"
            integrator_map = {
                "steep": "steepest_descent",
                "cg": "conjugant_gradient",
                "l-bfgs": "low_memory_broyden_fletcher_goldfarb_shanno",
            }
            value = integrator_map.get(
                integrator,
                [val for key, val in integrator_map.items() if key in integrator],
            )
            value = (
                value
                if not isinstance(value, list)
                else value[0]
                if len(value) != 0
                else None
            )
            workflow.method.method = value
            nsteps = input_parameters.get("nsteps", None)
            workflow.method.optimization_steps_maximum = int(nsteps) if nsteps else None
            nstenergy = input_parameters.get("nstenergy", None)
            workflow.method.save_frequency = int(nstenergy) if nstenergy else None

            force_maximum = input_parameters.get("emtol", None)
            force_conversion = ureg.convert(
                1.0, ureg.kilojoule * ureg.avogadro_number / ureg.nanometer, ureg.newton
            )
            workflow.method.convergence_tolerance_force_maximum = (
                to_float(force_maximum) * force_conversion
                if to_float(force_maximum)
                else None
            )

            energies = []
            steps = []
            for calc in sec_calc:
                val = calc.get("energy")
                energy = val.get("potential") if val else None
                if energy:
                    energies.append(energy.value.magnitude)
                    step = calc.get("step")
                    steps.append(step)
            workflow.results.energies = energies
            workflow.results.steps = steps
            workflow.results.optimization_steps = len(energies) + 1

            final_force_maximum = self.log_parser.get("maximum_force")
            final_force_maximum = (
                re.split("=|\n", final_force_maximum)[1]
                if final_force_maximum
                else None
            )
            final_force_maximum = to_float(final_force_maximum)
            workflow.results.final_force_maximum = (
                to_float(final_force_maximum) * force_conversion
                if to_float(final_force_maximum)
                else None
            )
            self.archive.workflow2 = workflow
        else:
            method, results = {}, {}
            nsteps = input_parameters.get("nsteps", None)
            method["n_steps"] = int(nsteps) if nsteps else None
            nstxout = input_parameters.get("nstxout", None)
            method["coordinate_save_frequency"] = int(nstxout) if nstxout else None
            nstvout = input_parameters.get("nstvout", None)
            method["velocity_save_frequency"] = int(nstvout) if nstvout else None
            nstfout = input_parameters.get("nstfout", None)
            method["force_save_frequency"] = int(nstfout) if nstfout else None
            nstenergy = input_parameters.get("nstenergy", None)
            method["thermodynamics_save_frequency"] = (
                int(nstenergy) if nstenergy else None
            )

            integrator_map = {
                "md": "leap_frog",
                "md-vv": "velocity_verlet",
                "sd": "langevin_goga",
                "bd": "brownian",
            }
            value = integrator_map.get(
                integrator,
                [val for key, val in integrator_map.items() if key in integrator],
            )
            value = (
                value
                if not isinstance(value, list)
                else value[0]
                if len(value) != 0
                else None
            )
            method["integrator_type"] = value
            timestep = input_parameters.get("dt", None)
            method["integration_timestep"] = (
                to_float(timestep) * ureg.picosecond if to_float(timestep) else None
            )

            thermostat_parameters = self.get_thermostat_parameters(integrator)
            method["thermostat_parameters"] = thermostat_parameters
            barostat_parameters = self.get_barostat_parameters()
            method["barostat_parameters"] = barostat_parameters

            if thermostat_parameters.get("thermostat_type"):
                method["thermodynamic_ensemble"] = (
                    "NPT" if barostat_parameters.get("barostat_type") else "NVT"
                )
            elif barostat_parameters.get("barostat_type"):
                method["thermodynamic_ensemble"] = "NPH"
            else:
                method["thermodynamic_ensemble"] = "NVE"

            params_key = "free_energy_calculation_parameters"
            method[params_key] = self.get_free_energy_calculation_parameters()

            self.xvg_parser.mainfile = self.get_gromacs_file("xvg")
            free_energies = self.xvg_parser.get("results")

            title = free_energies.get("title", "") if free_energies is not None else ""
            flag_fe = False
            if (
                r"dH/d\xl\f{}" in title and r"\xD\f{}H" in title
            ):  # TODO incorporate x and y axis labels into the checks
                flag_fe = True
                results_key = "free_energy_calculations"
                results[results_key] = {}
                columns = free_energies.get("column_vals")
                results[results_key]["n_frames"] = len(columns)
                lambdas = method[params_key].get("lambdas", None)
                results[results_key]["n_states"] = (
                    len(lambdas[0].get("value", [])) if lambdas is not None else None
                )
                results[results_key]["lambda_index"] = method[params_key].get(
                    "lambda_index", None
                )
                results[results_key]["value_unit"] = str(self._gro_energy_units.units)
                xaxis = free_energies.get("xaxis", "").lower()
                # The expected columns of the xvg file are:
                # Total Energy
                # dH/dlambda current lambda
                # Delta H between each lambda and current lambda (n_lambda columns)
                # PV Energy
                if (
                    "time" in xaxis
                    and columns[:, 3:-1].shape[1] == results[results_key]["n_states"]
                ):
                    results[results_key]["times"] = columns[:, 0] * ureg.ps
                    columns = columns[:, 1:] * self._gro_energy_units.magnitude
                else:
                    self.logger.warning(
                        "Unexpected format of xvg file. Not storing free energy calculation results."
                    )
                    flag_fe = False

            self.parse_md_workflow(dict(method=method, results=results))

            if flag_fe:
                filename = os.path.join(
                    os.path.dirname(self.filepath.split("/raw/")[-1]),
                    f"{os.path.basename(self.filepath)}.archive.hdf5",
                )
                if not os.path.isfile(
                    os.path.join(os.path.dirname(self.filepath), filename)
                ):
                    if self.archive.m_context:
                        with self.archive.m_context.raw_file(filename, "wb") as f:
                            pass
                sec_fe_parameters = (
                    self.archive.workflow2.method.free_energy_calculation_parameters[0]
                )
                sec_fe = self.archive.workflow2.results.free_energy_calculations[0]
                sec_fe.method_ref = sec_fe_parameters
                if self.archive.m_context:
                    with self.archive.m_context.raw_file(filename, "r+b") as f:
                        sec_fe.value_total_energy_magnitude = to_hdf5(
                            columns[:, 0],
                            f,
                            f"{sec_fe.m_path()}/value_total_energy_magnitude",
                        )
                        sec_fe.value_total_energy_derivative_magnitude = to_hdf5(
                            columns[:, 1],
                            f,
                            f"{sec_fe.m_path()}/value_total_energy_derivative_magnitude",
                        )
                        sec_fe.value_total_energy_differences_magnitude = to_hdf5(
                            columns[:, 2:-1],
                            f,
                            f"{sec_fe.m_path()}/value_total_energy_differences_magnitude",
                        )
                        sec_fe.value_PV_energy_magnitude = to_hdf5(
                            columns[:, -1],
                            f,
                            f"{sec_fe.m_path()}/value_PV_energy_magnitude",
                        )

    def parse_input(self):
        sec_run = self.archive.run[-1]
        sec_input_output_files = x_gromacs_section_input_output_files()
        sec_run.x_gromacs_section_input_output_files = sec_input_output_files

        topology_file = os.path.basename(self.traj_parser.mainfile)
        if topology_file.endswith("tpr"):
            sec_input_output_files.x_gromacs_inout_file_topoltpr = topology_file
        elif topology_file.endswith("gro"):
            sec_input_output_files.x_gromacs_inout_file_confoutgro = topology_file

        trajectory_file = os.path.basename(self.traj_parser.auxilliary_files[0])
        sec_input_output_files.x_gromacs_inout_file_trajtrr = trajectory_file

        edr_file = os.path.basename(self.energy_parser.mainfile)
        sec_input_output_files.x_gromacs_inout_file_eneredr = edr_file

        sec_control_parameters = x_gromacs_section_control_parameters()
        sec_run.x_gromacs_section_control_parameters = sec_control_parameters
        input_parameters = self.input_parameters
        input_parameters.update(self.get("header", {}))
        for key, val in input_parameters.items():
            key = (
                "x_gromacs_inout_control_%s"
                % key.replace("-", "").replace(" ", "_").lower()
            )
            if hasattr(sec_control_parameters, key):
                val = str(val) if not isinstance(val, np.ndarray) else val
                setattr(sec_control_parameters, key, val)

    def init_parser(self):
        self.log_parser.mainfile = self.filepath
        self.log_parser.logger = self.logger
        self.traj_parser.logger = self.logger
        self.energy_parser.logger = self.logger
        self._frame_rate = None

    def reuse_parser(self, parser):
        self.log_parser.quantities = parser.log_parser.quantities

    def parse(self, filepath, archive, logger):
        self.filepath = os.path.abspath(filepath)
        self.archive = archive
        self.logger = logging.getLogger(__name__) if logger is None else logger
        self._maindir = os.path.dirname(self.filepath)
        self._gromacs_files = os.listdir(self._maindir)
        self._basename = os.path.basename(filepath).rsplit(".", 1)[0]

        self.init_parser()

        sec_run = Run()
        self.archive.run.append(sec_run)

        header = self.log_parser.get("header", {})

        sec_run.program = Program(
            name="GROMACS",
            version=str(header.get("version", "unknown")).lstrip("VERSION "),
        )

        sec_time_run = TimeRun()
        sec_run.time_run = sec_time_run
        for key in ["start", "end"]:
            time = self.log_parser.get("time_%s" % key)
            if time is None:
                continue
            setattr(
                sec_time_run,
                "date_%s" % key,
                datetime.datetime.strptime(time, "%a %b %d %H:%M:%S %Y").timestamp(),
            )

        host_info = self.log_parser.get("host_info")
        if host_info is not None:
            sec_run.x_gromacs_program_execution_host = host_info[0]
            sec_run.x_gromacs_parallel_task_nr = host_info[1]
            sec_run.x_gromacs_number_of_tasks = host_info[2]

        # parse the input parameters using log file as default and mdp output or input as supplementary
        self.input_parameters = {
            key.replace("_", "-"): val.lower() if isinstance(val, str) else val
            for key, val in self.log_parser.get("input_parameters", {}).items()
        }
        self.mdp_parser.mainfile = self.get_mdp_file()
        for key, param in self.mdp_parser.get("input_parameters", {}).items():
            new_key = key.replace("_", "-")
            if new_key not in self.input_parameters:
                self.input_parameters[new_key] = (
                    param.lower() if isinstance(param, str) else param
                )

        topology_file = self.get_gromacs_file("tpr")
        # I have no idea if output trajectory file can be specified in input
        trr_file = self.get_gromacs_file("trr")
        trr_file_nopath = trr_file.rsplit(".", 1)[0]
        trr_file_nopath = trr_file_nopath.rsplit("/")[-1]
        xtc_file = self.get_gromacs_file("xtc")
        xtc_file_nopath = xtc_file.rsplit(".", 1)[0]
        xtc_file_nopath = xtc_file_nopath.rsplit("/")[-1]
        if not trr_file_nopath.startswith(self._basename):
            trajectory_file = (
                xtc_file if xtc_file_nopath.startswith(self._basename) else trr_file
            )
        else:
            trajectory_file = trr_file

        self.traj_parser.mainfile = topology_file
        self.traj_parser.auxilliary_files = [trajectory_file]
        # check to see if the trr file can be read properly (and has positions), otherwise try xtc file instead
        positions = None
        if (universe := self.traj_parser.universe) is not None:
            atoms = getattr(universe, "atoms", None)
            positions = getattr(atoms, "positions", None)
        if positions is None:
            self.traj_parser.auxilliary_files = [xtc_file] if xtc_file else [trr_file]

        self.parse_method()

        self.parse_system()

        self.parse_thermodynamic_data()

        self.parse_input()

        self.parse_workflow()

        self.traj_parser.clean()
