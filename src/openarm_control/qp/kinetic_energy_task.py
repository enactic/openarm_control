# Copyright 2026 Enactic, Inc.
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

"""MuJoCo-compatible Mink kinetic-energy regularization task."""

from __future__ import annotations

import mink
import mujoco
import numpy as np


class KineticEnergyRegularizationTask(mink.KineticEnergyRegularizationTask):
    """Use Mink's energy objective with MuJoCo's current inertia API.

    Mink 1.1.0 calls the pre-MuJoCo-3.10 ``mj_fullM`` signature. This override
    keeps Mink's public task API and exact objective while obtaining an
    up-to-date inertia matrix through MuJoCo's current sparse-matrix API.
    """

    def compute_qp_objective(
        self,
        configuration: mink.Configuration,
    ) -> mink.Objective:
        """Return ``cost * M(q) / dt^2`` as a QP Hessian."""
        if self.inv_dt_sq is None:
            raise mink.exceptions.IntegrationTimestepNotSet(self.__class__.__name__)

        model = configuration.model
        data = configuration.data
        mujoco.mj_makeM(model, data)
        inertia = np.empty((model.nv, model.nv), dtype=np.float64)
        mujoco.mju_sym2dense(
            inertia,
            data.M,
            model.M_rownnz,
            model.M_rowadr,
            model.M_colind,
        )
        return mink.Objective(
            H=self.cost * self.inv_dt_sq * inertia,
            c=np.zeros(model.nv, dtype=np.float64),
        )
