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


m_package = Package()


class x_lib_atoms_section_gap(MSection):
    """
    Description of Gaussian Approximation Potentials (GAPs).
    """

    m_def = Section(validate=False)

    x_lib_atoms_training_config_refs = Quantity(
        type=runschema.calculation.Calculation,
        shape=['n_sparseX'],
        description="""
        References to frames in training configuration.
        """,
    )

    x_lib_atoms_GAP_params_label = Quantity(
        type=str,
        shape=[],
        description="""
        GAP classifier.
        """,
    )

    x_lib_atoms_GAP_params_svn_version = Quantity(
        type=str,
        shape=[],
        description="""
        GAP classifier.
        """,
    )

    x_lib_atoms_GAP_data_do_core = Quantity(
        type=str,
        shape=[],
        description="""
        GAP classifier.
        """,
    )

    x_lib_atoms_GAP_data_e0 = Quantity(
        type=np.float64,
        shape=[],
        description="""
        GAP classifier.
        """,
    )

    x_lib_atoms_command_line_command_line = Quantity(
        type=str,
        shape=[],
        description="""
        GAP classifier.
        """,
    )

    x_lib_atoms_gpSparse_n_coordinate = Quantity(
        type=np.int32,
        shape=[],
        description="""
        GAP classifier.
        """,
    )

    x_lib_atoms_gpCoordinates_n_permutations = Quantity(
        type=np.int32,
        shape=[],
        description="""
        GAP classifier.
        """,
    )

    x_lib_atoms_gpCoordinates_sparsified = Quantity(
        type=str,
        shape=[],
        description="""
        GAP classifier.
        """,
    )

    x_lib_atoms_gpCoordinates_signal_variance = Quantity(
        type=np.float64,
        shape=[],
        description="""
        GAP classifier.
        """,
    )

    x_lib_atoms_gpCoordinates_label = Quantity(
        type=str,
        shape=[],
        description="""
        GAP classifier.
        """,
    )

    x_lib_atoms_gpCoordinates_n_sparseX = Quantity(
        type=np.int32,
        shape=[],
        description="""
        GAP classifier.
        """,
    )

    x_lib_atoms_gpCoordinates_covariance_type = Quantity(
        type=np.int32,
        shape=[],
        description="""
        GAP classifier.
        """,
    )

    x_lib_atoms_gpCoordinates_signal_mean = Quantity(
        type=np.float64,
        shape=[],
        description="""
        GAP classifier.
        """,
    )

    x_lib_atoms_gpCoordinates_sparseX_filename = Quantity(
        type=str,
        shape=[],
        description="""
        GAP classifier.
        """,
    )

    x_lib_atoms_gpCoordinates_dimensions = Quantity(
        type=np.int32,
        shape=[],
        description="""
        GAP classifier.
        """,
    )

    x_lib_atoms_gpCoordinates_theta = Quantity(
        type=np.float64,
        shape=[],
        description="""
        GAP classifier.
        """,
    )

    x_lib_atoms_gpCoordinates_descriptor = Quantity(
        type=str,
        shape=[],
        description="""
        GAP classifier.
        """,
    )

    x_lib_atoms_gpCoordinates_perm_permutation = Quantity(
        type=np.int32,
        shape=[],
        description="""
        GAP classifier.
        """,
    )

    x_lib_atoms_gpCoordinates_perm_i = Quantity(
        type=np.int32,
        shape=[],
        description="""
        GAP classifier.
        """,
    )

    x_lib_atoms_gpCoordinates_alpha = Quantity(
        type=np.float64,
        shape=['n_sparseX', 2],
        description="""
        GAP classifier.
        """,
    )

    x_lib_atoms_gpCoordinates_sparseX = Quantity(
        type=np.float64,
        shape=['n_sparseX', 'dimensions'],
        description="""
        GAP classifier.
        """,
    )


class Run(runschema.run.Run):
    m_def = Section(validate=False, extends_base_section=True)

    x_lib_atoms_section_gap = SubSection(
        sub_section=SectionProxy('x_lib_atoms_section_gap'), repeats=False
    )


class Calculation(runschema.calculation.Calculation):
    m_def = Section(validate=False, extends_base_section=True)

    x_lib_atoms_virial_tensor = Quantity(
        type=np.float64,
        shape=[3, 3],
        unit='pascal',
        description="""
        Virial tensor for this frame.
        """,
    )

    x_lib_atoms_config_type = Quantity(
        type=str,
        shape=[],
        description="""
        Configuration type, e.g. = dislocation_quadrupole.
        """,
    )
