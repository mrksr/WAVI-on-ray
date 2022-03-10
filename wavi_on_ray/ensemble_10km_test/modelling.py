from typing import List

import GPy
import numpy as np


def _build_kernel_list(kernels):
    kernel_map = {
        "bias": GPy.kern.Bias,
        "linear": GPy.kern.Linear,
        "matern32": GPy.kern.Matern32,
        "matern52": GPy.kern.Matern52,
        "rbf": GPy.kern.RBF,
    }
    return [kernel_map[kernel](1) for kernel in kernels]


def build_model(
    inputs: List[np.array], outputs: List[np.array], kernels: List[str]
) -> GPy.models.GPRegression:
    """Builds a coregionalization GP model.

    Args:
        inputs (List[np.array]): List of input datasets, must match outputs.
        outputs (List[np.array]): List of output dataset, must match inputs.
        kernels (List[str]): List of kernel names for LCM model.

    Returns:
        GPy.models.GPRegression: A trained coregionalization GP model.
    """
    lcm_kernel = GPy.util.multioutput.LCM(
        input_dim=1,
        num_outputs=2,
        kernels_list=_build_kernel_list(kernels),
    )

    coreg_model = GPy.models.gp_coregionalized_regression.GPCoregionalizedRegression(
        X_list=inputs, Y_list=outputs, kernel=lcm_kernel
    )
    coreg_model[".*noise.*variance"].constrain_fixed(1e-8)

    try:
        coreg_model[".*Mat.*variance.*"].constrain_bounded(30.0, 100.0)
        coreg_model[".*Mat.*lengthscale.*"].constrain_bounded(50.0, 500.0)
    except AttributeError:
        # Kernel does not exist
        pass

    try:
        coreg_model[".*RBF.*variance.*"].constrain_bounded(30.0, 100.0)
        coreg_model[".*RBF.*lengthscale.*"].constrain_bounded(50.0, 500.0)
    except AttributeError:
        # Kernel does not exist
        pass

    coreg_model.optimize_restarts(num_restarts=5)
    return coreg_model
