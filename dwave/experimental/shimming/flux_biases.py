# Copyright 2025 D-Wave
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from copy import deepcopy
import math
from typing import Any, Optional, Iterable, Callable, Literal

import numpy as np

import dimod
from dimod.typing import Variable, Bias

__all__ = ["shim_flux_biases", "qubit_freezeout_alpha_phi"]


def qubit_freezeout_alpha_phi(
    eff_temp_phi: float = 0.112,
    flux_associated_variance: float = 1 / 1024,
    estimator_variance: float = 1 / 256,
    unit_conversion: float = 1.148e-3,
):
    r"""Determine the learning rate for independent qubits.

    For a qubit offset by :math:`\Phi_{0i}` that is to be corrected by a choice
    of :ref:`flux bias <parameter_qpu_flux_biases>` :math:`\Phi`, a model of
    single-qubit freezeout dictates that magnetization
    :math:`<s_i> = tanh(\frac{\Phi_{0i} + \Phi_i}{T})`, where :math:`T` is the
    effective temperature.

    Assume the unshimmed magnetization to be zero-mean distributed with small
    variance, :math:`\Delta_1`, and a standard sampling-based estimator with
    variance :math:`\Delta_2 = \frac{1}{\text{num_reads}}`. You can then
    determine an update to the flux, :math:`\Phi = l <s_i>_{\text{data}}`, where
    the learning rate :math:`l = T \frac{\Delta_1}{Delta_1 + Delta_2}`. This
    update is optimal in the sense that it minimizes the expected square
    magnetization.

    For correlated spin systems and/or experiments that are not well described
    by thermal freezeout, a data-driven approach is recommended to determining
    the schedule and related parameters. The freezeout (Boltzmann) distribution
    can be extended to correlated models, wherein the covariance matrix plays a
    role in determining the optimal learning rate. A single qubit rate can
    remain a good approximation given weakly correlated spins.

    Args:
        eff_temp_phi: Effective (unitless) inverse temperature at freezeout.
            Can be determined from current device parameters.
        flux_associated_variance: The expected variance of the magnetization
            (:math:`m`) due to flux offset.
        estimator_variance: The expected variance in the magnetization estimate,
            :math:`\frac{1-m^2}{\text{num_reads}}`.
        unit_conversion: Conversion factor from units of
            :ref:`h <parameter_qpu_h>` to units of :math:`\Phi`. Can be
            determined from published device parameters. See
            :func:`~dwave.system.temperatures.h_to_fluxbias`.
    Returns:
        An appropriate scale for the learning rate, minimizing the expected
        square magnetization.

    Example:
        Determining an :math:`\alpha_\Phi` appropriate for forward anneal of a
        weakly coupled system, ``Advantage_system4.1`` based on published
        parameters. Note that defaults (by contrast) are determined based on
        published values for ``Advantage2_system1.3``.

        >>> from dwave.experimental.shimming import qubit_freezeout_alpha_phi
        ...
        >>> alpha_phi = qubit_freezeout_alpha_phi(
        ...     eff_temp_phi=0.198,
        ...     flux_associated_variance=1/1024,
        ...     estimator_variance=1/256,
        ...     unit_conversion=1.647e-3)

    """
    return (
        unit_conversion
        * eff_temp_phi
        * flux_associated_variance
        / (flux_associated_variance + estimator_variance)
    )


def shim_flux_biases(
    bqm: dimod.BinaryQuadraticModel,
    sampler: dimod.Sampler,
    *,
    sampling_params: Optional[dict[str, Any]] = None,
    shimmed_variables: Optional[Iterable[Variable]] = None,
    learning_schedule: Optional[Iterable[float]] = None,
    convergence_test: Optional[Callable] = None,
    symmetrize_experiments: bool = True,
    sampling_params_updates: Optional[list] = None,
    beta_hypergradient: float = 0.4,
    num_steps: int = 10,
    alpha: Optional[float] = None,
    inclusion_by_update: dict[Variable, list] = None,
) -> tuple[list[Bias], dict, dict]:
    r"""Return flux biases that minimize magnetization for symmetry-preserving
    experiments.

    You can refine calibration for specific QPU experiments by modifying your
    QPU programming. The :ref:`flux bias <parameter_qpu_flux_biases>` parameter
    compensates for low-frequency environmental spins that couple into qubits,
    distorting the target distribution. Although you can modify either
    :ref:`flux bias <parameter_qpu_flux_biases>` or :ref:`h <parameter_qpu_h>`
    to restore symmetry of the sampled distribution, flux biases more accurately
    eliminate common forms of low-frequency noise.

    Assuming the magnetization (expectation for measured spins, or sign of the
    persistent current) is a smooth, monotonic function of the qubit body's
    magnetic fluxes (:ref:`flux bias <parameter_qpu_flux_biases>`), you can
    determine parameters that achieve a target magnetization, :math:`m`, by
    iterating :math:`\Phi(t+1) \leftarrow \Phi(t) - L(t) (<s> - m)`, where
    :math:`L(t)` is an iteration-dependent map, :math:`<s>` is the expected
    magnetization, :math:`m` is the target magnetization, and :math:`\Phi` are
    the programmed flux biases.

    By default :math:`L(t)` is uniform with respect to programmed qubits, and
    determined by a
    `hypergradient descent method <https://doi.org/10.48550/arXiv.1703.04782>`_.
    Alternatively you can specify the learning rate as a list, in which
    case the hypergradient method is not used.

    Symmetry can be broken by the choice of initial condition in reverse
    annealing, non-zero :math:`h`, :ref:`parameter_polarizing_schedules`, or
    non-zero flux biases over unshimmed fluxes. You can collect data for two
    experiments, with inverted symmetry breaking between them, anticipating zero
    magnetization in the experimental average rather than per experiment.
    Shimming based on this symmetrized data set is expected to determine a good
    shim for both experiments, assuming a weak dependence of noise on the
    symmetry-breaking field.

    Where strong correlations, or strong symmetry-breaking effects, are present
    in an experiment, the sampled distribution may contain insufficient
    information to independently shim all degrees of freedom. Shims are expected
    to be a smooth function of annealing parameters such as
    :ref:`annealing time <parameter_qpu_annealing_time>`,
    :ref:`anneal schedule <parameter_qpu_anneal_schedule>`, and
    Hamiltonian parameters. You can use shims inferred in smoothly related
    models as approximations (or initial conditions) for searches in target
    models.

    If the provided learning rate or learning schedule is too large, it is
    possible to exceed the bounds of allowed values for the flux-bias offsets.

    Args:
        bqm: A :class:`~dimod.binary.BinaryQuadraticModel`.
        sampler: A :class:`~dwave.system.samplers.DWaveSampler`.
        sampling_params: Parameters of the
            :class:`~dwave.system.samplers.DWaveSampler`. Note that if
            ``sampling_params`` contains
            :ref:`flux biases <parameter_qpu_flux_biases>`, these are treated as
            an initial condition and edited in place. Chose a value for the
            :ref:`parameter_qpu_num_reads` parameter in conjunction with your
            chosen schedule. Note that, the :ref:`parameter_qpu_initial_state`
            parameter, if provided, is assumed to be specified according the
            Ising model convention (:math:`\pm 1`, and :math:`-3` for inactive).
        shimmed_variables: A list of variables to shim; by default all elements
            in :attr:`~dimod.binary.BinaryQuadraticModel.variables`.
        learning_schedule: An iterable of gradient-descent prefactors. When not
            provided, prefactors are determined by a hypergradient-descent
            method parameterized by the ``alpha``, ``beta_hypergradient``, and
            ``num_steps`` arguments.
        convergence_test: A callable that takes the history of magnetizations
            and flux biases as input, returning ``True`` to exit the search, and
            ``False`` otherwise. By default, all stages specified in the
            ``learning_schedule`` argument are completed.
        symmetrize_experiments: If ``True``, performs a test to determine
            symmetry breaking in the experiment: a non-zero
            :ref:`parameter_qpu_initial_state` for reverse anneal, non-zero
            :math:`h`, or non-zero :ref:`parameter_qpu_flux_biases` (on some
            unshimmed variables). If any of these are present, magnetization is
            inferred by averaging over two experiments with symmetry-breaking
            elements inverted. The shim averages the symmetrically related
            experiments to achieve zero magnetization.
        sampling_params_updates: Where you require averaging across many
            experiments, you can specify a list of updates. Each element in your
            list is a dictionary that updates the ``sampling_params`` argument.
            Experiments are averaged over these sampling-parameter updates to
            determine the magnetization used in shimming. The original value of
            the updated sampling parameter is ignored. The
            :ref:`parameter_qpu_flux_biases` parameter must not be an
            updated parameter. See the
            `examples directory <https://github.com/dwavesystems/dwave-experimental/tree/main/examples>`_
            for use cases.
        beta_hypergradient: Controls the learning rate evolution for the
            hypergradient-descent method, enabling improved performance through
            customizing for the annealing protocol and QPU. Supported values are
            in range :math:`(0,1)`. Ignored if you specify the
            ``learning_schedule`` argument.
        num_steps: Number of steps taken by the hypergradient-descent method if
            you do not specify a ``learning_schedule`` from which to infer.
            Default is 10 if neither is specified.
        alpha: Initial learning rate for the hypergradient-descent method,
            enabling improved performance through customizing for the annealing
            protocol and QPU. Supported values are positive, real floats with a
            typical scale you can determine using the
            :func:`.qubit_freezeout_alpha_phi` function. By default, initialized
            using the :func:`.qubit_freezeout_alpha_phi` function. Ignored if
            you specify the ``learning_schedule`` argument.
        inclusion_by_update: Maps each shimmed variable to a list of experiment
            indices (within the combined ``sampling_params_updates`` ×
            signed-experiment product) whose magnetization values are averaged
            for that variable's gradient update. By default all experiments in
            the current step are averaged uniformly.
            THIS ARGUMENT REQUIRES TESTS.

    Returns:
        A tuple consisting of 3 parts:
            1.  Flux biases in a list using the :ref:`parameter_qpu_flux_biases`
                format (for use by a
                :class:`~dwave.system.samplers.DWaveSampler` sampler).
            2.  History of flux-bias assignments per shimmed component.
            3.  History of magnetizations per shimmed component.

    Example:
        See the
        `examples <https://github.com/dwavesystems/dwave-experimental/tree/main/examples>`_
        `test <https://github.com/dwavesystems/dwave-experimental/tree/main/tests>`_
        directories for additional use cases.

        Shim degenerate qubits at constant learning rate and solver defaults.
        The learning schedule and number of reads is for demonstration only, and
        has not been optimized.

        >>> import numpy as np
        >>> import dimod
        >>> from dwave.system import DWaveSampler
        >>> from dwave.experimental.shimming import shim_flux_biases, qubit_freezeout_alpha_phi
        ...
        >>> qpu = DWaveSampler()        # doctest: +SKIP
        >>> bqm = dimod.BQM.from_ising({q: 0 for q in qpu.nodelist}, {})    # doctest: +SKIP
        >>> alpha_phi = qubit_freezeout_alpha_phi()  # Unoptimized to the experiment, for demonstration purposes.
        >>> ls = [alpha_phi]*5
        >>> sp = {'num_reads': 2048, 'auto_scale': False}
        >>> fb, fb_history, mag_history = shim_flux_biases(bqm,     # doctest: +SKIP
        ...     qpu,
        ...     sampling_params=sp,
        ...     learning_schedule=ls)
        ...
        >>> print(f"RMS magnetization by iteration: {np.sqrt(np.mean([np.array(v)**2 for v in mag_history.values()], axis=0))}") # doctest: +SKIP
    """

    # Natural candidates for future feature enhancements:
    # - Use standard stochastic gradient descent methods, such as ADAM, perhaps with
    # hypergradient descent to eliminate learning rate choice inefficiencies.
    # - Allow shimming of linear combinations of fluxes, e.g. to control for a
    # known (or desired) correlation structure.
    # Note: the purpose of shimming should not be to learn parameters of general
    # graph-restricted Boltzmann machines, we should modify our plugin for this
    # purpose.

    if sampling_params is None:
        sampling_params = {}

    if "flux_biases" in sampling_params:
        flux_biases = sampling_params.pop("flux_biases")
        if len(flux_biases) != sampler.properties["num_qubits"]:
            raise ValueError("flux_biases length incompatible with the sampler")
        pop_fb = True
    else:
        flux_biases = [0] * sampler.properties["num_qubits"]
        pop_fb = False

    if shimmed_variables is None:
        # All variables of the model
        shimmed_variables = bqm.variables
    else:
        if len(shimmed_variables) == 0:
            raise ValueError("shimmed_variables should not be empty")
        elif not set(shimmed_variables).issubset(bqm.variables):
            raise ValueError("Invalid shimmed variables")

    if symmetrize_experiments:
        unshimmed_variables = set(bqm.variables).difference(shimmed_variables)
        fbnonzero = any(flux_biases[v] != 0 for v in shimmed_variables)
        if bqm.vartype is dimod.BINARY:
            bqm = bqm.change_vartype(dimod.SPIN, inplace=False)
        hnonzero = any(bqm.linear.values())
        if hnonzero:
            bqm = bqm.copy()
        reverseanneal = "initial_state" in sampling_params
        polarizedmca = (
            sampling_params is not None
            and "x_polarizing_schedules" in sampling_params
            and any(
                v != 0
                for wfm in sampling_params["x_polarizing_schedules"]
                for _, v in wfm
            )
        )
    else:
        fbnonzero = hnonzero = reverseanneal = polarizedmca = False
    num_signed_experiments = 1 + int(
        reverseanneal or hnonzero or fbnonzero or polarizedmca
    )

    if sampling_params_updates is None:
        # By default, a single experimental setting:
        sampling_params_updates = [{}]
    else:
        # Although there are scenarios where some flux_biases are
        # set whilst others are shimmed, support for this is beyond
        # the scope of this function.
        if any("flux_biases" in sp for sp in sampling_params_updates):
            raise ValueError(
                "flux_biases should not be explicitely set"
                "within sampling_params_updates."
            )
    num_experiments = num_signed_experiments * len(sampling_params_updates)

    use_hypergradient = learning_schedule is None
    if not use_hypergradient:
        num_steps = len(learning_schedule)
    else:
        if alpha is None:
            alpha = qubit_freezeout_alpha_phi()
        if not (0 < beta_hypergradient < 1):
            raise ValueError("beta_hypergradient should be in the (0,1) interval")
        if not (alpha > 0):
            raise ValueError("alpha should be positively valued")

    if convergence_test is None:
        convergence_test = lambda x, y: False

    flux_bias_history = {v: [flux_biases[v]] for v in shimmed_variables}
    mag_history = {v: [] for v in bqm.variables}
    for step in range(num_steps):
        # Possible feature enhancement for intermediate num_experiments:
        # following loops are parallelizable, call sample() asyncrhonously.
        for spu in sampling_params_updates:
            sampling_params.update(spu)
            for _ in range(num_signed_experiments):
                if reverseanneal:
                    for i in bqm.variables:
                        sampling_params["initial_state"][i] *= -1
                if hnonzero:
                    for i in bqm.variables:
                        bqm.linear[i] *= -1
                if fbnonzero:
                    for i in unshimmed_variables:
                        flux_biases[i] *= -1
                if polarizedmca:
                    sampling_params["x_polarizing_schedules"] = [
                        [(t, -v) for t, v in wfm]
                        for wfm in sampling_params["x_polarizing_schedules"]
                    ]
                ss = sampler.sample(bqm, flux_biases=flux_biases, **sampling_params)
                all_mags = np.sum(
                    ss.record.sample * ss.record.num_occurrences[:, np.newaxis], axis=0
                ) / np.sum(ss.record.num_occurrences)

                for idx, v in enumerate(ss.variables):
                    mag_history[v].append(all_mags[idx])

        if convergence_test(mag_history, flux_bias_history):
            # The data is not used to update the flux_biases
            # This can be included as part of the test evaluation (if required)
            break

        if inclusion_by_update is None:
            exp_av_mags = {
                v: np.mean(mag_history[v][-num_experiments:]) for v in shimmed_variables
            }
        else:
            exp_av_mags = {
                v: np.mean(
                    [
                        mag_history[v][-num_experiments + i]
                        for i in inclusion_by_update[v]
                    ]
                )
                for v in shimmed_variables
            }
        if use_hypergradient:
            magnetizations = np.array([exp_av_mags[v] for v in shimmed_variables])
            if step > 0:
                norm = np.linalg.norm(magnetizations) * np.linalg.norm(last_mags)
                if math.isclose(norm, 0):
                    # When magnetization norms are zero the paper method is ill defined.
                    # One could choose to convergence test to exit at zero magnetization
                    # as an alternative.
                    alpha *= 1 - beta_hypergradient
                else:
                    alpha *= (
                        1
                        + beta_hypergradient * np.dot(magnetizations, last_mags) / norm
                    )
            last_mags = magnetizations
        else:
            alpha = learning_schedule[step]

        for v in shimmed_variables:
            flux_biases[v] -= alpha * exp_av_mags[v]
            flux_bias_history[v].append(flux_biases[v])

    if pop_fb:
        sampling_params["flux_biases"] = flux_biases

    return flux_biases, flux_bias_history, mag_history


def once_iterated_tanh_fit(
    bqm: dimod.BinaryQuadraticModel,
    sampler: dimod.Sampler,
    sampling_params: dict[str, Any],
    sampling_params_updates: list[dict[str, Any]],
    inclusion_by_update: dict[Variable, tuple[int, ...]],
    num_programmings: int = 10,
    basis_points: np.ndarray = np.linspace(-2e-4, 2e-4, 11),
    iterate: bool = True,
    verbose: bool = True,
    update_sampling_params: bool = True,
) -> list[Bias]:
    """Estimate flux-bias offsets by fitting a tanh curve to magnetization data.

    THIS METHOD IS EXPERIMENTAL AND LACKS PROPER TESTING.

    For each experimental setting in ``sampling_params_updates``, this function
    sweeps over ``basis_points``, collects magnetization data across
    ``num_programmings`` QPU programmings, and fits a tanh function
    :math:`m = \\tanh(p_1 (\\Phi - p_0))` to the resulting
    magnetization-versus-flux-bias curve. The fitted offset :math:`p_0`
    corresponds to the flux value that drives the mean magnetization to zero.
    When ``iterate`` is ``True``, each shimmed variable's flux bias is updated
    to the fitted :math:`p_0`.

    Unlike :func:`shim_flux_biases`, which uses iterative gradient descent,
    this approach estimates the correction in a single pass over the basis
    points, making it useful as an initial condition or a cross-check.

    Args:
        bqm: A :class:`~dimod.binary.BinaryQuadraticModel` describing the
            problem. This should include only detector and target qubits.
        sampler: A :class:`~dwave.system.samplers.DWaveSampler`.
        sampling_params: Base sampling parameters passed to the sampler.
            Updated in place for each entry in ``sampling_params_updates``.
            This should not include flux_biases. If this includes flux_biases,
            those values are taken as a baseline.
        sampling_params_updates: A list of dictionaries; each dictionary is
            applied as an update to ``sampling_params`` before collecting data
            for that experimental setting. These two updates are expected to
            swap the x_annealing_lines between detectors and targets. 
        inclusion_by_update: Maps each shimmed variable to a tuple of update
            indices (into ``sampling_params_updates``) from which its
            magnetization data is drawn.
        num_programmings: Number of independent QPU programmings per basis
            point. Averaging over programmings reduces programming-induced
            noise. Default is 10.
        basis_points: An array of flux-bias values used as the independent
            variable for the tanh fit. Default spans
            :math:`[-2 \\times 10^{-4},\\, 2 \\times 10^{-4}]` in 11 steps.
        iterate: If ``True``, updates each variable's entry in ``flux_biases``
            to the fitted tanh center :math:`p_0`.  Set to ``False`` to perform
            the scan and fit without modifying the flux biases.
        flux_biases: Initial flux-bias list (one entry per physical qubit).
            If ``None``, all biases are initialised to zero.
        verbose: If ``True``, plots the tanh fits and data points for each
            detected variable.
        update_sampling_params: If ``True``, updates the ``sampling_params`` 
            dictionary with the final flux biases under the "flux_biases" key.
    Returns:
        Flux biases in the :ref:`parameter_qpu_flux_biases` format, updated
        with the fitted offsets when ``iterate`` is ``True``.
    """
    from scipy.optimize import curve_fit
    _flux_biases = sampling_params.pop("flux_biases", None)
    
    if _flux_biases is None:
        flux_biases = [0.0] * sampler.properties["num_qubits"]
    else:
        flux_biases = _flux_biases.copy()  
    for idx_spu, spu in enumerate(sampling_params_updates):
        mags = {}
        sampling_params.update(spu)

        flux_biases_perturbed = flux_biases.copy()

        detected_variables = {
            k for k, v in inclusion_by_update.items() if v[0] == idx_spu
        }
        for fb in basis_points:
            for v in detected_variables:
                flux_biases_perturbed[v] = flux_biases[v]
            for p in range(num_programmings):
                ss = sampler.sample(
                    bqm, **sampling_params, flux_biases=flux_biases_perturbed
                )
                for idx_v, v in enumerate(ss.variables):
                    mags[(fb, p, v)] = np.sum(
                        ss.record.num_occurrences * ss.record.sample[:, idx_v]
                    ) / np.sum(ss.record.num_occurrences)
        

        def f_tanh(x, p0, p1):
            return np.tanh(p1 * (x - p0))

        for v in detected_variables:
            xdata = basis_points
            ydata = [
                np.mean([mags[(fb, p, v)] for p in range(num_programmings)])
                for fb in basis_points
            ]
            p = curve_fit(f_tanh, xdata, ydata, p0=(0.0, 1e3))
            if verbose:
                plt.figure(f"{idx_spu}")
                plt.plot(xdata, ydata, label=f"{v} {p[0][0]:.3g}")
                plt.plot(xdata, f_tanh(xdata, *p[0]))
                plt.legend()
                plt.xlabel('Flux biases (Phi0)')
                plt.ylabel('Magnetization (m)')
            if iterate:
                flux_biases[v] = p[0][0]
       
    if verbose:
        plt.show()
    if update_sampling_params:
        sampling_params["flux_biases"] = flux_biases
    elif _flux_biases is not None:
        # reset to original flux biases
        sampling_params["flux_biases"] = _flux_biases
    return flux_biases


def shim_tds_flux_biases(
    bqm: dimod.BinaryQuadraticModel,
    sampler: dimod.Sampler,
    target_lines: set,
    detector_lines: set,
    *,
    line_assignments: dict,
    sampling_params: dict[str, Any],
    learning_schedule: Optional[Iterable[float]] = None,
    convergence_test: Optional[Callable] = None,
    symmetrize_experiments: bool = False,
    beta_hypergradient: float = 0.4,
    num_steps: int = 10,
    alpha: Optional[float] = None,
    shimmed_variables: Optional[Iterable[Variable]] = None,
    method: Literal['standard', 'Iterated tanh'] = "standard",
) -> tuple[list[Bias], dict, dict]:
    """Shim flux biases using paired target and detector annealing lines.

    THIS METHOD IS EXPERIMENTAL AND LACKS PROPER TESTING.
    
    Targets and detectors when paired 1:1 can act as complementary
    detectors. If we assume the required flux biases do not depend
    on the waveform applied, then we can alternate the role of detector
    and target, shimming only the detector flux to arrive at a unique
    fixed point.

    We assume flux biases required for symmetric outcomes are independent
    of the ``x_anneal_schedules``. We can measure the coupled system(s)
    of qubits alternating the role of detector. Assuming
    :math:`d\langle m_D\rangle/d\Phi_D > -\mathrm{sign}(J_{SD})\,d\langle m_D\rangle/d\Phi_S > 0`,
    we can pursue paired experiments and update only detector qubit fluxes
    in each with a convergence guarantee.

    Args:
        bqm: A :class:`~dimod.binary.BinaryQuadraticModel` describing the
            coupled target–detector system.
        sampler: A :class:`~dwave.system.samplers.DWaveSampler`.
        target_lines: Indices of annealing lines whose qubits act as targets
            (the system whose state we wish to measure).
        detector_lines: Indices of annealing lines whose qubits act as
            detectors (the qubits whose flux biases are shimmed).
        line_assignments: Maps each variable (qubit index) to its annealing
            line index.
        sampling_params: Base sampling parameters passed to the sampler.
            Must include ``x_anneal_schedules``.
        learning_schedule: An iterable of gradient-descent prefactors for the
            ``standard`` method. When not provided, the hypergradient-descent
            method is used. Ignored for the ``Iterated tanh`` method.
        convergence_test: A callable that takes the history of magnetizations
            and flux biases and returns ``True`` to exit the search early.
            Ignored for the ``Iterated tanh`` method.
        symmetrize_experiments: If ``True``, symmetry-breaking elements in the
            experiment are inverted for a second run and magnetizations are
            averaged. Passed through to :func:`.shim_flux_biases` for the
            ``standard`` method. Default is ``False``.
        beta_hypergradient: Controls the learning-rate evolution for the
            hypergradient-descent method. Supported values are in
            :math:`(0, 1)`. Ignored when ``learning_schedule`` is provided or
            ``method`` is not ``"standard"``.
        num_steps: Number of gradient-descent steps for the ``standard``
            method. Default is 10.
        alpha: Initial learning rate for the hypergradient-descent method.
            See :func:`.shim_flux_biases`. Ignored for the ``Iterated tanh``
            method.
        shimmed_variables: Variables to shim. Defaults to all variables in
            ``bqm``.
        method: Shimming algorithm to use. ``"standard"`` delegates to
            :func:`.shim_flux_biases` with appropriately constructed
            ``sampling_params_updates`` and ``inclusion_by_update``.
            ``"Iterated tanh"`` uses :func:`.once_iterated_tanh_fit`.
    Returns:
        A tuple of three parts mirroring the return value of
        :func:`.shim_flux_biases`:

        1.  Flux biases in the :ref:`parameter_qpu_flux_biases` format
            (``None`` for the ``"Iterated tanh"`` method).
        2.  History of flux-bias assignments per shimmed component
            (``None`` for the ``"Iterated tanh"`` method).
        3.  History of magnetizations per shimmed component, or the
            flux-bias list returned by :func:`.once_iterated_tanh_fit`.
    """
    if "x_anneal_schedules" not in sampling_params:
        raise ValueError("x_anneal_schedules should be specified in sampling_params")


    if shimmed_variables is None:
        shimmed_variables = bqm.variables
    num_lines = len(sampling_params["x_anneal_schedules"])
    if any(
        len(lines) < 1 or not all(0 <= l < num_lines for l in lines)
        for lines in [target_lines, detector_lines]
    ):
        raise ValueError(
            "target_lines and detector_lines should be a non-empty iterable of line indices"
        )
    x_anneal_schedules_reversed = deepcopy(sampling_params["x_anneal_schedules"])
    dl = next(iter(detector_lines))
    for line in target_lines:
        x_anneal_schedules_reversed[line] = sampling_params["x_anneal_schedules"][dl]
    tl = next(iter(target_lines))
    for line in detector_lines:
        x_anneal_schedules_reversed[line] = sampling_params["x_anneal_schedules"][tl]
    inclusion_by_update = {
        v: (0,) if line_assignments[v] in detector_lines else (1,)
        for v in shimmed_variables
    }
    sampling_params_update = [
        {"x_anneal_schedules": sampling_params["x_anneal_schedules"]},
        {"x_anneal_schedules": x_anneal_schedules_reversed},
    ]
    if method == "standard":
        return shim_flux_biases(
            bqm,
            sampler,
            sampling_params=sampling_params,
            learning_schedule=learning_schedule,
            convergence_test=convergence_test,
            symmetrize_experiments=symmetrize_experiments,
            beta_hypergradient=beta_hypergradient,
            num_steps=num_steps,
            alpha=alpha,
            sampling_params_updates=sampling_params_update,
            inclusion_by_update=inclusion_by_update,
            shimmed_variables=shimmed_variables,
        )
    else:
        return (
            None,
            None,
            once_iterated_tanh_fit(
                bqm,
                sampler,
                sampling_params=sampling_params,
                sampling_params_updates=sampling_params_update,
                inclusion_by_update=inclusion_by_update,
            ),
        )


if __name__ == "__main__":
    import sys

    sys.path.append("../../../")
    from examples.mca_shim_AO_FB import plot_shim, _make_anneal_schedules
    from dwave.system import DWaveSampler
    from dwave.experimental.multicolor_anneal import get_properties
    import matplotlib.pyplot as plt

    print(
        "Test source detector shimming. This module code can be moved to tests and examples after development"
    )
    use_larmour_precession_documented_waveforms = (
        True  # Set False for Majid's waveforms
    )
    method = "standard"  # Can switch to 'Iterated Tanh' for Majid's method
    reference_test = (
        1  # 1 is single edge (majid), 2 is 5 edges (majid), 0 is first QPU edge
    )
    solver = "Advantage2_prototype2_x_internal"
    if reference_test > 0:
        # A 2-stage tanh() fitting method using 11 basis points, 8 programmings per point, of 1000 reads yields
        phi_qs = [(1.2600874e-05, 5.15008e-06)]
        es = [
            (1048, 1049),
        ]
        # sampling_params = These differ very slightly from my defaults, but not in a manner that should impact convergence.
        # Success of the method requires weak dependence on the waveforms given s_target.
        target_s = 0.292314
        if reference_test > 1:
            target_q = [1048, 400, 1024, 1023, 1028]
            detector_q = [1049, 1101, 1025, 1029, 1022]
            T_flux_offsets = [
                1.2600874e-05,
                -1.187981e-06,
                -6.139937e-06,
                -3.130485e-06,
                -1.749384e-05,
            ]
            D_flux_offsets = [
                5.15008e-06,
                1.7151358e-05,
                4.392779e-06,
                -1.852948e-06,
                7.622289e-06,
            ]
            es = [(v1, v2) for v1, v2 in zip(target_q, detector_q)]

        def calc_anneal_schedules(
            annealing_line_dicts, target_s, source_lines, target_lines, detector_lines
        ):
            """Majid's function"""
            tar_schedule = [
                [0.0, 0.0],
                [1.0, 0.0],
                [2.0, target_s],
                [8.0, target_s],
                [25.0, target_s],
            ]
            if source_lines:
                src_s_for_maxbeta = min(
                    [annealing_line_dicts[i]["maxC"] for i in source_lines]
                )  # 5.07
                src_s_for_minbeta = max(
                    [annealing_line_dicts[i]["minC"] for i in source_lines]
                )
                src_schedule = [
                    [0.0, src_s_for_minbeta],
                    [1.0, src_s_for_maxbeta],
                    [21.0, src_s_for_maxbeta],
                    [21.01, src_s_for_minbeta],
                    [25.0, src_s_for_minbeta],
                ]

            det_s_for_maxbeta = min([ald["maxC"] for ald in annealing_line_dicts])
            det_s_for_minbeta = max([ald["minC"] for ald in annealing_line_dicts])
            det_schedule = [
                [0.0, det_s_for_minbeta],
                [10.0, det_s_for_minbeta],
                [21.0, det_s_for_minbeta],
                [21.01, det_s_for_maxbeta],
                [25.0, det_s_for_maxbeta],
            ]

            do_nothing_schedule = [
                [0.0, 0.0],
                [1.0, 0.0],
                [2.0, 0.0],
                [8.0, 0.0],
                [25.0, 0.0],
            ]

            anneal_schedules = np.repeat(
                np.array(do_nothing_schedule).reshape(1, 5, 2), 6, axis=0
            )
            for line in source_lines:
                anneal_schedules[line] = src_schedule
            for line in target_lines:
                anneal_schedules[line] = tar_schedule
            for line in detector_lines:
                anneal_schedules[line] = det_schedule
            return anneal_schedules

    else:
        target_s = None
    qpu = DWaveSampler(solver="Advantage2_prototype2_x_internal")
    zephyr_shape = qpu.properties["topology"]["shape"]
    exp_feature_info = get_properties(qpu)

    line_assignments = {
        n: al_idx for al_idx, al in enumerate(exp_feature_info) for n in al["qubits"]
    }
    if es is None:
        es = qpu.edgelist[:1]
    variables = sorted(v for e in es for v in e)
    bqm = dimod.BinaryQuadraticModel("SPIN").from_ising(
        {v: 0 for v in variables}, {e: -1 for e in es}
    )
    detector_lines = {line_assignments[e[1]] for e in es}
    target_lines = {line_assignments[e[0]] for e in es}
    source_lines = {}  # Remaining lines are set to minC
    if use_larmour_precession_documented_waveforms:
        x_anneal_schedules = calc_anneal_schedules(
            annealing_line_dicts=exp_feature_info,
            target_s=target_s,
            source_lines=source_lines,
            target_lines=target_lines,
            detector_lines=detector_lines,
        )
    else:
        x_anneal_schedules = _make_anneal_schedules(
            exp_feature_info,
            detector_lines=detector_lines,
            target_lines=target_lines,
            source_lines=source_lines,
            target_c=target_s,
        )

    sampling_params = dict(
        num_reads=500,
        answer_mode="raw",
        x_disable_filtering=True,
        x_anneal_schedules=x_anneal_schedules,
    )

    # Detector only method
    flux_biases, flux_history, mag_history = shim_flux_biases(
        bqm, qpu, sampling_params=sampling_params, shimmed_variables=[e[1] for e in es]
    )
    plot_shim(mag_history, flux_history, label="Det. only")

    # New method
    flux_biases, flux_history, mag_history = shim_tds_flux_biases(
        bqm,
        qpu,
        target_lines,
        detector_lines,
        sampling_params=sampling_params,
        line_assignments=line_assignments,
        num_steps=40,
        method=method,  ## Temporary parameter to test Majid's proposal
    )
    plot_shim(mag_history, flux_history)
    plt.figure(2)
    plt.legend()
    plt.show()
