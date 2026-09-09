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
"""
An example to show coarse-grained calibration refinement for multi-color annealing.

Calibration can be refined relative to the baseline since certain static or low
frequency control errors are a function of the specific waveforms and programmed
Hamiltonian.
flux_biases, x_anneal_delays and anneal_offsets are refined sequentially by simple iterative methods
that typically succeed to improve calibration on a plurality of qubits.

This example builds many parallel target-detector-source (T-D-S) embeddings that
are refined in parallel.
Optionally, the embeddings can be made consistent with embedding of a loop,
in which case the final plot demonstrates an interference pattern arising from
pi/2-pulse initialization (up to limitations of decoherence and control error).
"""

import argparse
import hashlib
import json
import os
import re
from typing import Collection, Iterable, Literal, Sequence

import pickle
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
import networkx as nx
import numpy as np
from tqdm import tqdm

import dimod
from dimod.typing import Variable
from dwave.system import DWaveSampler
from dwave.system.testing import MockDWaveSampler
from dwave.system.composites import ParallelEmbeddingComposite
from minorminer.utils.parallel_embeddings import find_multiple_embeddings
from dwave.experimental.multicolor_anneal import (
    get_properties,
    make_tds_graph,
    make_tds_x_schedules,
    #   make_tds_x_schedule_delays,
    SOLVER_FILTER,
)
from dwave.experimental.shimming import shim_flux_biases, shim_tds_flux_biases


def _figure_path(
    figures_dir: str, figure_label: str, cache_str: str | None = None
) -> str:
    """Create a stable file path for a matplotlib figure label."""
    safe_label = re.sub(r"[^0-9A-Za-z._-]+", "_", figure_label).strip("_")
    suffix = f"_{cache_str}" if cache_str else ""
    return os.path.join(figures_dir, f"{safe_label}{suffix}.png")


def _apply_tight_layout() -> None:
    """Apply tight_layout to all open figures so axis labels are not truncated."""
    for fig_num in plt.get_fignums():
        plt.figure(fig_num).tight_layout()


def _save_open_figures(figures_dir: str, cache_str: str | None = None) -> None:
    """Save currently open matplotlib figures to disk."""
    os.makedirs(figures_dir, exist_ok=True)
    _apply_tight_layout()
    for fig_num in plt.get_fignums():
        fig = plt.figure(fig_num)
        figure_label = fig.get_label() or f"Figure_{fig_num}"
        fig.savefig(_figure_path(figures_dir, figure_label, cache_str))


def _calc_anneal_offsets(
    frequencies: np.ndarray,
    psd: np.ndarray,
    target_A: float,
    dAdc: float,
    dAfit: float = 0.5,
    fit_to_target_A: bool = True,
) -> np.ndarray:
    """Determine the anneal_offset necessary to synchronize frequency.

    After fully decoupling from the source, the signal is expected to be
    well described by a cosine subject to an exponentially decaying envelope,
    controlled by the T2 coherence time.
    The power spectral density is therefore a Lorentzian peaked at the
    oscillating frequency. The peak can be efficiently estimated with small
    bias as the expectation on a symmetric interval about the anticipated
    frequency. This is a simple heuristic estimator, reasonably robust to
    experimental non-idealities.

    root(A(c)^2 + [B(c) delta h]^2) is expected to describe the frequency,
    where delta h is non-zero owing to flux_biases. If flux biases are small
    we can correct the frequency accounting for a only using
    delta c = (A(c) - <A(c)>)/ [dA/dc], where dA/dc is known approximately
    from the schedule.

    Args:
        frequencies: Frequencies at which power is provided (GHz).
        psd: Power spectral density, the absolute discrete fourier transform
            value squared at each frequency.
        target_A: Expected/desired peak position (GHz).
        dAdc: Approximate rate of change of A with c (anneal offset).
        dAfit: Fractional range around target_A to use for estimation.
        fit_to_target_A: If True, use target_A as the reference; if False, use the mean.

    Returns:
        Estimated anneal offsets per qubit/embedding in units of normalized c.
    """

    # NB a symmetric window only works for frequencies in the range,
    # and some bias is introduced as a function of the window.
    Amin = target_A * dAfit
    Amax = target_A * (1 + dAfit)

    Afilter = np.logical_and(frequencies < Amax, frequencies > Amin)
    mean_A_est = np.sum(
        psd[:, Afilter] * frequencies[Afilter][np.newaxis, :], axis=1
    ) / np.sum(psd[:, Afilter], axis=1)
    mu = np.mean(mean_A_est)
    print(
        f"A: target={target_A:.3g}, estimated_mean={np.mean(mean_A_est):.3g}, and standard deviation={np.std(mean_A_est):.3g}",
    )
    if fit_to_target_A:
        mu = target_A
    dcs = (mean_A_est - mu) / dAdc
    return dcs


def make_yC(
    delays: np.ndarray,
    A: float,
    *,
    T2: float = 0.0101,
    phi_s: float = 0.0,
    phi_d: float = 0.0,
) -> np.ndarray:
    """Make a noise-free model signal for computational basis preparation and detection.

    Eq 6. from arXiv 2603.15534 at theta_d = theta_s = pi/2
    Note that this model assumes decoupling from the source, and ideally
    controlled flux biases (in particular note phi_q=0 on the target).
    The source polarization is assumed to be instantaneously removed at delay t=0,
    with measurement performed at delay t.

    y(t) = cos(2* pi * A * t + phi_s - phi_d) exp(-t/T2)

    Args:
        delays: Measurement times (microseconds)
        A: frequency
        T2: exponential envelope time scale. Defaulted as
            from T_phi = 12ns and T1 = 32ns, typical of Advantage2 research.
        phi_s: Bloch sphere rotation (azimuthal angle) for the source.
        phi_d: Bloch sphere rotation (azimuthal angle) for the detector.
    Returns:
        A model signal
    """
    return np.exp(-delays / T2) * np.cos(2 * np.pi * A * delays + phi_s - phi_d)


def make_yE(
    delays: np.ndarray, *, T1: float = 0.032, theta_s: float = np.pi / 2
) -> np.ndarray:
    """Make a noise-free model signal for energy basis preparation and detection.

    Eq 6. from arXiv 2603.15534 at theta_d = 0, with theta_s = pi/2 by default.
    Note that this model assumes decoupling from the source, and ideally
    controlled flux biases (in particular note phi_q=0 on the target, and the
    flux bias on the detector must be small compared to the polarizing signal).
    The source polarization is assumed to be instantaneously removed.

    y(t) = 1 - exp(-t / T1) * (1 - cos(theta_s))

    Args:
        delays: Measurement times (microseconds)
        T1: Exponential envelope time scale (microseconds). Defaulted to a value
            typical of Advantage2 research systems.
        theta_s: Bloch sphere rotation (polar angle) for the source (default: pi/2).
    Returns:
        A model signal
    """
    return 1 - np.exp(-delays / T1) * (1 - np.cos(theta_s))


def make_y(
    delays: np.ndarray,
    A: float,
    *,
    T1: float = 0.032,
    T2: float = 0.0101,
    theta_d: float = np.pi / 2,
    theta_s: float = np.pi / 2,
    phi_d: float = 0.0,
    phi_s: float = 0.0,
    t0: float = 0.0,
) -> np.ndarray:
    """Make a noise-free model signal for arbitrary basis preparation and detection.

    The source polarization is fixed to sign(Jts * fbs):
    y(t) = sign(J) y(theta_d, theta_s, phi_d, phi_s) for t > t0, and
           sign(J) for t < t0   # Constrained by source polarization

    where y(theta_d, theta_s, phi_d, phi_s) matches Eq. 6 of arXiv:2603.15534.

    Assume that the pinning magnetic field from the target ~|Ip(s_tar) Jtd| is large
    compared to the fb_d whilst the source is coupled (the target is pinned), so that
    the detector energetically prefers a state aligned with the target (thence the source)
    during the coupled phase. The source decoupling is approximated as instantaneously.

    Args:
        delays: Measurement times (microseconds)
        A: frequency
        T1: assumed coherence time (microseconds)
        T2: assumed coherence time (microseconds)
        theta_d: detection basis angle
        theta_s: source basis angle
        phi_d: detection phase
        phi_s: source phase
        t0: Source decoupling time (default: 0.0)
    Returns:
        A model signal assuming ideal detection and instantaneous source decoupling.

    """
    preparation_orientation = np.sign(np.sin(theta_s))
    rd = delays - t0
    return preparation_orientation * (rd <= 0) + (rd > 0) * (
        np.cos(theta_d) * make_yE(rd, T1=T1, theta_s=theta_s)
        + np.sin(theta_d)
        * np.sin(theta_s)
        * make_yC(rd, A=A, T2=T2, phi_d=phi_d, phi_s=phi_s)
    )


def dyC_dt0(
    delays: np.ndarray, A: float, T2: float = 0.0101, sign_Jts_fbs: int = 1
) -> np.ndarray:
    """Calculate derivative of signal model with respect to time delay.

    Computes the derivative of the model signal produced by ``make_y`` with
    respect to the time delay. For ``t > 0`` the underlying signal is
    ``sign_Jts_fbs * exp(-t/T2) * cos(2*pi*A*t)``; for ``t < 0`` it is
    constant so the derivative is zero.

    Args:
        delays: Time delays at which to evaluate the derivative (microseconds).
        A: Frequency (GHz).
        T2: Exponential envelope time scale (microseconds).
        sign_Jts_fbs: Sign convention for the Josephson coupling term (default: -1).

    Returns:
        Array of derivatives evaluated at each delay.
    """
    y0 = np.clip(np.exp(-delays / T2), a_min=0, a_max=1)
    dy0_dt0 = -y0 / T2 * (delays > 0)  # Only contributes when not clipped
    y1 = np.cos(2 * np.pi * A * delays)
    dy1_dt0 = -2 * np.pi * A * np.sin(2 * np.pi * A * delays)

    return sign_Jts_fbs * (dy0_dt0 * y1 + y0 * dy1_dt0)


def dyC_dA(
    delays: np.ndarray, A: float, T2: float = 0.0101, sign_Jts_fbs: int = 1
) -> np.ndarray:
    """Calculate derivative of signal model with respect to frequency.

    EFFECTIVELY OBSOLETE FUNCTION

    Computes the derivative of the ``make_y`` signal
    ``sign_Jts_fbs * exp(-t/T2) * cos(2*pi*A*t)`` (for ``t > 0``) with respect
    to the frequency parameter ``A``.

    Args:
        delays: Time delays at which to evaluate the derivative (microseconds).
        A: Frequency (GHz).
        T2: Exponential envelope time scale (microseconds).
        sign_Jts_fbs: Sign convention for the Josephson coupling term (default: -1).

    Returns:
        Array of frequency derivatives evaluated at each delay.
    """
    y0 = np.clip(np.exp(-delays / T2), a_min=0, a_max=1)
    dy1_dA = -2 * np.pi * delays * np.sin(2 * np.pi * A * delays)

    return sign_Jts_fbs * (y0 * dy1_dA)


def artificial_data(
    delays: np.ndarray,
    A: float,
    *,
    T1: float = 0.032,
    T2: float = 0.0101,
    theta_d: float = np.pi / 2,
    theta_s: float = np.pi / 2,
    phi_d: float = 0.0,
    phi_s: float = 0.0,
    t0: float = 0.0,
    num_independent_samples: float = float("Inf"),
    prng: np.random.Generator | int | None = None,
) -> np.ndarray:
    """Create an artificial data set

    y(t) = ideal signal produced by ``make_y``
    with variance of (1 - y(t)^2) in the measured state. Given independent
    and identically distributed samples we can model noise as normally
    distributed. The signal is clipped so that the value is always physical
    within the range [-1, 1] (only impacts small num_independent_samples).

    Args:
        delays: Time of measurement (microseconds).
        A: Frequency (GHz).
        T1: Assumed coherence time (microseconds).
        T2: Exponential envelope time scale (microseconds).
        theta_d: Detection basis angle (default: pi/2).
        theta_s: Source basis angle (default: pi/2).
        phi_d: Detection phase (default: 0.0).
        phi_s: Source phase (default: 0.0).
        t0: Reference time for the start of the measurement (default: 0.0).
        num_independent_samples: Number of samples to model.
        prng: Pseudo random number generator or seed.

    Returns:
        A model signal with sampling noise.
    """

    y = make_y(
        delays,
        A,
        T1=T1,
        T2=T2,
        theta_d=theta_d,
        theta_s=theta_s,
        phi_d=phi_d,
        phi_s=phi_s,
        t0=t0,
    )

    if num_independent_samples != float("Inf"):
        prng = np.random.default_rng(prng)
        return np.clip(
            y
            + np.sqrt((1 - y**2) / num_independent_samples) * prng.normal(size=len(y)),
            a_min=-1,
            a_max=1,
        )
    else:
        return y


def run_parallel_experiment(
    sampler: ParallelEmbeddingComposite,
    bqm: dimod.BinaryQuadraticModel,
    sampling_params: dict,
    delays: np.ndarray | list,
    detector_lines: Iterable[int],
    detected_vars: None | Sequence[Variable] = None,
) -> np.ndarray:
    """Collect detector magnetization for a set of independent embeddings

    Runs a Target-Detector-Source quench experiment on many parallel
    embeddings with the specified delays applied to detector lines.
    Sample averaged magnetization are calculated on detected qubits in
    each embedding and returned as a numpy array.

    Args:
        sampler: Parallel embedding composite sampler wrapping the QPU sampler.
        bqm: Binary Quadratic Model.
        sampling_params: Parameters passed to the QPU sampler.
        delays: Detector x_schedule_delays (microseconds).
        detector_lines: Iterable of detector line indices.
        detected_vars: Iterable of bqm variables to keep for
           detector magnetization calculation.

    Raises:
        ValueError if 'x_anneal_schedules' is not a key of
        `sampling_params`

    Returns:
        Numpy array of detector magnetizations (delays x embeddings).
    """
    if "x_anneal_schedules" not in sampling_params:
        raise ValueError("No multi-color anneal specified")

    if detected_vars is None:
        detected_vars = [
            v for v in bqm.variables if type(v) == tuple and v[0] == "detector"
        ]

    reset_delay = "x_schedule_delays" in sampling_params
    x_schedule_delays = sampling_params.pop(
        "x_schedule_delays", [0.0] * len(sampling_params["x_anneal_schedules"])
    )
    baseline_delays = x_schedule_delays.copy()
    mean_Z_detector = []
    for delay in tqdm(delays, disable=len(delays) == 1):
        for line in detector_lines:
            x_schedule_delays[line] = baseline_delays[line] + delay
        # Return as a list of samplesets, instead of aggregated:
        samplesets, _ = sampler.sample_multiple(
            [bqm] * len(sampler.embeddings),
            x_schedule_delays=x_schedule_delays,
            **sampling_params,
        )
        # Extract detector magnetization from each sampleset
        detector_samples = [
            dimod.keep_variables(sampleset, detected_vars).record.sample
            for sampleset in samplesets
        ]
        if len(detected_vars) == 1:
            mean_Z_detector.append([np.mean(sample) for sample in detector_samples])
        else:
            mean_Z_detector.append(
                [np.mean(sample, axis=0) for sample in detector_samples]
            )
    if reset_delay:
        sampling_params["x_schedule_delays"] = baseline_delays
    return np.array(mean_Z_detector)


def plot_shim(
    mag_history: dict,
    flux_history: dict,
    num_experiments: int = 1,
    label: str = "",
    max_qubit_labels: int = 10,
    plt_show_block: None | bool = None,
) -> None:
    """Plot the iterative flux-bias calibration refinement process.

    Args:
        mag_history: the magnetizations estimated throughout the iterative
            process for every embedding.
        flux_history: the flux_biases assignments throughout the iterative
            process for every embedding.
        num_experiments: Number of programmings per flux iteration. Using 1
            by default it should be noted that 2 magnetizations may be
            be measured per step in flux_biases.
        label: a label for the plots, used in legends.
        max_qubit_labels: maximum number of qubit labels to include in legend,
            if larger, defaults to no labels.
        plt_show_block: If not None (default), then execute
            :code:`plt.show(block=plt_show_block)` to display the figure.
    """
    mag_array = np.array(list(mag_history.values()))
    flux_array = np.array(list(flux_history.values()))

    mag_array = np.reshape(
        mag_array,
        (mag_array.shape[0], mag_array.shape[1] // num_experiments, num_experiments),
    )

    plt.figure("All_Qubit_Magnetization_by_calibration_refinement_iteration")
    plt.title(r"Magnetization by iteration, $\langle Z\rangle_{detector}$")
    y0 = 0
    for experiment_sign in range(num_experiments):
        y = mag_array[:, :, experiment_sign].transpose()
        if num_experiments > 1:
            plt.plot(
                y,
                label=f"Initial state all {-1 + 2*experiment_sign}",
            )
        else:
            plt.plot(
                y,
            )
        y0 = y0 + y / num_experiments

    plt.figure("Bulk_Magnetization_by_calibration_refinement_iteration")
    n_qubits = y0.shape[1]
    rms = np.sqrt(np.mean(y0**2, axis=1))
    # Jackknife standard error: recompute RMS leaving out one qubit at a time.
    loo_ss = np.sum(y0**2, axis=1, keepdims=True) - y0**2
    rms_loo = np.sqrt(loo_ss / (n_qubits - 1))
    rms_se = np.sqrt(
        (n_qubits - 1)
        / n_qubits
        * np.sum((rms_loo - rms_loo.mean(axis=1, keepdims=True)) ** 2, axis=1)
    )
    plt.errorbar(np.arange(len(rms)), rms, yerr=rms_se, capsize=3)
    plt.xlabel("Calibration refinement iteration")
    plt.ylabel("Root Mean Square Magnetization")

    plt.figure("All_Qubit_Magnetization_by_calibration_refinement_iteration")
    if num_experiments > 1:
        plt.plot(
            np.mean(mag_array, axis=2).transpose(),
            color="black",
            label="Experiment average",
        )
        plt.legend()
        plt.xlabel("Calibration refinement iteration")
    else:
        plt.xlabel("Programming")
        if mag_array.shape[0] <= max_qubit_labels:
            plt.legend(flux_history.keys(), title=f"{label} Qubit index")
    plt.ylabel("Magnetization")

    plt.figure("Flux_bias_by_calibration_refinement_iteration")
    plt.title("All refined flux_biases")
    plt.plot(flux_array.transpose())
    plt.xlabel("Calibration refinement iteration")
    plt.ylabel("Flux bias ($\\Phi_0$)")
    if mag_array.shape[0] <= max_qubit_labels:
        plt.legend(flux_history.keys(), title=f"{label} Qubit index")
    _apply_tight_layout()
    if plt_show_block is not None:
        plt.show(block=plt_show_block)


def _plot_tds_schedules(
    x_polarizing_schedule: list[list[float]],
    x_anneal_schedules: list[list[list[float]]],
    plt_show_block: None | bool = None,
) -> None:
    """Plots the piecewise linear schedules used

    Args:
        x_polarizing_schedule: The polarization signal.
        x_anneal_schedules: The list of anneal schedules, one per line.
        plt_show_block: If not None (default), then execute
            :code:`plt.show(block=plt_show_block)` to display the figure.
    """
    plt.figure("PWL multi-color annealing schedules")
    plt.title("PWL schedules")
    for line, schedule in enumerate(x_anneal_schedules):
        plt.plot(
            [x for x, _ in schedule], [y for _, y in schedule], label=f"Line {line}"
        )
    plt.plot(
        [x for x, _ in x_polarizing_schedule],
        [y for _, y in x_polarizing_schedule],
        label="Polarizing schedule",
        linestyle="dashed",
        color="black",
    )
    plt.xlabel("Time (microseconds)")
    plt.ylabel("Schedule value")
    plt.legend()
    _apply_tight_layout()
    if plt_show_block is not None:
        plt.show(block=plt_show_block)


def imshow_data(
    mean_Z_detector: np.ndarray,
    delays: np.ndarray,
    colormap_type: Literal["default", "divergent"] = "divergent",
    linthresh: float = 1.0,
    first: int = 0,
    last: int | None = None,
    context_str: str = "",
    plt_show_block: None | bool = None,
    ax=None,
) -> None:
    """Display detector magnetization data as a heatmap.

    Creates an image plot of detector magnetization values organized by delay times,
    with optional divergent or default colormaps.

    Args:
        mean_Z_detector: 2D array of detector magnetizations (delays x embeddings).
        delays: Array of time delay values (microseconds); ytick labels are
            rendered in nanoseconds (values are multiplied by 1000).
        colormap_type: Type of colormap to use ("default" or "divergent").
        linthresh: Threshold for the symmetric logarithmic colormap
            (only relevant if colormap_type is "divergent").
        first: Index whose delay value is highlighted as an additional ytick
            label. Does not restrict the plotted range.
        last: Index one past the last delay value highlighted as an additional
            ytick label. If None, uses ``mean_Z_detector.shape[0]``. Does not
            restrict the plotted range.
        context_str: Optional context string to append to figure title.
        plt_show_block: If not None (default), then execute
            :code:`plt.show(block=plt_show_block)` to display the figure.
        ax: Optional matplotlib axes instance for plotting into an existing figure.
    """
    fig_title = f"Timeseries_{colormap_type}_colormap{context_str}"
    if colormap_type == "divergent":
        norm, cmap = SymLogNorm(linthresh=linthresh, vmin=-1, vmax=1), "RdBu"
    else:
        norm, cmap = None, None
    if ax is None:
        ax = plt.figure(fig_title).gca()
        ax.set_title(f"Real-space magnetization {context_str}".strip())
    ax.imshow(mean_Z_detector, norm=norm, cmap=cmap)
    if last is None:
        last = mean_Z_detector.shape[0]
    yticks_dict = {
        first: f"{1000 * delays[first]:.3g}",
        last - 1: f"{1000 * delays[last-1]:.3g}",
    }
    yticks_dict.update(
        {
            0: f"{1000 * delays[0]:.3g}",
            mean_Z_detector.shape[0] - 1: f"{1000 * delays[-1]:.3g}",
        }
    )
    ax.set_yticks(
        list(yticks_dict.keys()),
        list(yticks_dict.values()),
    )
    ax.set_xlabel("Target-Detector-Source embedding")
    ax.set_ylabel("Delay, nanoseconds")
    _apply_tight_layout()
    if plt_show_block is not None:
        plt.show(block=plt_show_block)


def _get_experiment_id(
    args: argparse.Namespace, num_char: int = 8, verbose: bool = True
) -> str:
    """Generate a unique hash identifier for the current experiment parameters.

    Creates a reproducible hash of the experiment arguments (excluding save_figures)
    to enable consistent caching and figure naming.

    Args:
        args: Argument namespace containing experiment parameters.
        num_char: Number of characters to use from the hash (default: 8).
        verbose: If True, print experiment parameters and identifier.

    Returns:
        Hash string identifier for the experiment.
    """
    vars_args = vars(args).copy()
    vars_args.pop(
        "save_figures", None
    )  # save_figures is not relevant to the experiment data, so we exclude it from the hash
    if vars_args.get("solver_name", None) == SOLVER_FILTER:
        vars_args["solver_name"] = "DefaultSolver"
        print(
            "NB: The default solver is used; the experiment identifier is computed "
            "as if solver_name='DefaultSolver', so it does not distinguish between "
            "different physical solvers resolved at runtime."
        )
    args_string = json.dumps(vars_args, sort_keys=True)
    identifier = hashlib.sha256(args_string.encode("utf-8")).hexdigest()[:num_char]
    if verbose:
        print("Demo parameters:")
        print(vars(args))
        print(
            "Demo identifier (labels cached data and saved figures, employ --use-cache for data reuse):",
            identifier,
        )
    return identifier


def _plot_time_series(
    embs: list,
    line_assignments: dict[int, int],
    mean_Z_detector: np.ndarray,
    delays: np.ndarray,
    line_color: list | None = None,
    plotted_emb_idxs: Collection[int] | None = None,
    label_emb_idxs: Collection[int] | None = None,
    xlabel: str = "Delay, nanoseconds",
    ylabel: str = "Detector magnetizations",
    plt_show_block: None | bool = None,
    ax=None,
) -> None:
    """Plot time series data for selected embeddings with line-based coloring.

    Creates a line plot of detector magnetization or other signals across delay times,
    with each embedding colored by its assigned annealing line.

    Args:
        embs: List of embeddings, each containing qubit assignments.
        line_assignments: Dict mapping qubits to annealing line indices.
        mean_Z_detector: 2D array of data (delays x embeddings).
        delays: Array of delay time values.
        line_color: List of colors indexed by annealing line.
        plotted_emb_idxs: Set of embedding indices to plot. If None, plots all.
        label_emb_idxs: Set of embedding indices to label in legend. If None, uses plotted_emb_idxs.
        xlabel: Label for the x-axis.
        ylabel: Label for the y-axis.
        plt_show_block: If not None (default), then execute
            :code:`plt.show(block=plt_show_block)` to display the figure.
        ax: Optional matplotlib axes instance for plotting into an existing figure.
    """

    if ax is None:
        ax = plt.gca()

    if plotted_emb_idxs is None:
        plotted_emb_idxs = set(range(len(embs)))
    if label_emb_idxs is None:
        label_emb_idxs = plotted_emb_idxs

    for emb_idx in plotted_emb_idxs:
        q = embs[emb_idx][0][0]
        line_idx = line_assignments[q]
        if emb_idx in label_emb_idxs:
            if len(label_emb_idxs) == len(plotted_emb_idxs):
                label = f"line {line_idx}(qubit {q})"
            else:
                label = f"line {line_idx}"
        else:
            label = None
        if line_color is not None:
            color = line_color[line_idx]
        else:
            color = None
        ax.plot(delays, mean_Z_detector[:, emb_idx], color=color, label=label)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.legend()
    ax.grid()
    if plt_show_block is not None:
        plt.show(block=plt_show_block)


def estimate_decoupling_timescale(
    sampler: ParallelEmbeddingComposite,
    bqm: dimod.BinaryQuadraticModel,
    sampling_params: dict,
    detector_lines: Iterable[int],
    detected_vars: Sequence[Variable] = None,
    preparation_orientation: float = 1.0,
    t_guess: float = 0.0,
    t_min: float | None = None,
    t_max: float | None = None,
    target_A: float = 2000,
    T2: float = 0.0101,
    threshold_cycle_av: float = 0.9,
    verbose: bool = True,
) -> tuple[float, list]:
    """Estimate bulk decoupling delay

    Source and detector waveforms typically overlap, in order that the detection
    is not of a source-coupled system a delay is required.
    When the source is coupled the magnetization plateaus, after decoupling the
    magnetization follows a damped oscillation given pi/2-pulse initialization.
    Departure from the plateau to zero is rapid and persistent in a proxy signal
    with significantly reduced oscillation amplitude (provided target_A is not
    too inaccurate) y_proxy(t) = y(t) + y(t + 1/(2*target_A)).

    A bound on the decoupling timescale is first established to O(T2).
    The interval is then searched by bisection to obtain a value that characterizes
    decoupling in the bulk (mean) to accurace O(1/target_A), variation between lines
    and qubits on a given line can be of a comparable scale. Series for separated
    lines or qubits can be evaluated by similar principles, and deviations reduced
    by application of anneal offsets on detectors and sources.

    Args:
        sampler: Parallel embedding composite sampler wrapping the QPU sampler.
        bqm: Binary Quadratic Model defining the target-detector-source system.
        sampling_params: Parameters passed to the QPU sampler.
        detector_lines: Iterable of detector line indices.
        detected_vars: Iterable of bqm variables to keep for detector
            magnetization calculation (default: (("detector", 0),)).
        preparation_orientation: Orientation of the preparation pulse (default: 1).
        t_guess: Initial delay guess used to seed the search (microseconds, default: 0.0).
        t_min: Known lower bound on the decoupling delay. If None, it is estimated.
        t_max: Known upper bound on the decoupling delay. If None, it is estimated.
        target_A: Target amplitude for the decoupling sequence (MHz)
        T2: Decoherence time (milliseconds)
        threshold_cycle_av: Threshold characterizing the decoupled regime (default: 0.9).
            The half-cycle average (t and t+1/(2*target_A)) magnetization is only larger than
            the threshold, in absolute value, in the source-coupled regime.
        verbose: Whether to print progress information (default: True).

    Returns:
        A tuple containing:
        - The final t_guess value representing the estimated decoupling timescale.
        - A list of (t_guess, mag) pairs representing the decoupling timescale search data.
    """
    if verbose and (t_min is None or t_max is None):
        print("Estimate bounds O(T1) on decoupling time scale (two programmings)")
        # Estimate upper and lower bound to precision T1:
    data = []
    while t_min is None or t_max is None:
        mag = run_parallel_experiment(
            sampler=sampler,
            bqm=bqm,
            sampling_params=sampling_params,
            delays=[t_guess],
            detector_lines=detector_lines,
            detected_vars=detected_vars,
        )
        data.append((t_guess, mag[0, :]))
        if np.median(mag) * preparation_orientation < threshold_cycle_av:
            t_max = t_guess
            t_guess -= T2
        else:
            mag_half = mag
            mag = run_parallel_experiment(
                sampler=sampler,
                bqm=bqm,
                sampling_params=sampling_params,
                delays=[t_guess - 1 / target_A / 2.0],
                detector_lines=detector_lines,
                detected_vars=detected_vars,
            )
            data.append((t_guess - 1 / target_A / 2.0, mag[0, :]))
            if (
                preparation_orientation / 2 * (np.median(mag + mag_half))
                < threshold_cycle_av
            ):
                t_max = t_guess - 1 / target_A / 2.0
                t_guess -= T2
            else:
                t_min = t_guess
                t_guess += T2
    if verbose:
        print(
            "Estimate decoupling timescale to accuracy better than 1/target_A, with ~log(T2*target_A) programmings"
        )
    while t_max - t_min > 1 / target_A:
        t_guess = (t_min + t_max) / 2
        mag = run_parallel_experiment(
            sampler=sampler,
            bqm=bqm,
            sampling_params=sampling_params,
            delays=[t_guess],
            detector_lines=detector_lines,
            detected_vars=detected_vars,
        )
        data.append((t_guess, mag[0, :]))
        if np.median(mag) * preparation_orientation < threshold_cycle_av:
            t_max = t_guess
        else:
            mag_half = mag
            mag = run_parallel_experiment(
                sampler=sampler,
                bqm=bqm,
                sampling_params=sampling_params,
                delays=[t_guess - 1 / target_A / 2.0],
                detector_lines=detector_lines,
                detected_vars=detected_vars,
            )
            data.append((t_guess - 1 / target_A / 2.0, mag[0, :]))
            if (
                preparation_orientation / 2 * np.median(mag + mag_half)
                < threshold_cycle_av
            ):
                t_max = t_guess - 1 / target_A / 2.0
            else:
                t_min = t_guess
    return t_guess, data


def _to_independent_tds(embs):
    """Convert embeddings to independent T-D-S systems. Assume the label scheme returned by :code:`tds_graph`"""
    independent_tds = []
    for emb in embs:
        independent_tds += [
            {
                0: emb[i],
                ("source", 0): emb[("source", i)],
                ("detector", 0): emb[("detector", i)],
            }
            for i in range(len(emb) // 3)
        ]
    return independent_tds


def main(
    cache_str: str | None = None,
    solver: dict | str | None = None,
    detector_lines: Iterable[int] | None = None,
    source_lines: Iterable[int] | None = None,
    target_lines: Iterable[int] | None = None,
    seed: int | None = None,
    max_num_embeddings: int | None = None,
    target_c: float | None = None,
    target_A: float | None = None,
    target_B: float | None = None,
    dAdc: float | None = None,
    flux_biases_method: Literal["None", "Detector", "Target-Detector"] = "Detector",
    t_decoupled: float | None = None,
    num_anneal_offset_iterations: int = 2,
    schedule_fn: str = "09-1323A-D_Advantage2_system4_annealing_schedule.xlsx",
    num_reads: int = 500,
    use_common_c_bounds: bool = True,
    use_overshoot: bool = True,
    save_figures: bool = False,
    T2: float = 0.0101,
    Jtd: float = 1.0,
    Jts: float = 1.0,
    Jtt: float = 0.1,
    loop_length: int | None = None,
    preparation_orientation: float = 1.0,
    embedding_timeout: int = 60,
    colormap_type: Literal["divergent", "default"] = "divergent",
    dt_div_A_final: float = 0.25,
) -> None:
    """Demonstrate T-D-S variability and mitigation strategies.

    An ideal single-qubit target system might be prepared in
    a polarized state |1> whose evolution is subsequently
    described by H(c) = A(c) + B(c) h.
    Control limitations dictate that the A(s), B(s) and h realized
    by different qubits at a common c varies. An h error in the detector
    qubit can also contribute to errors in measurement.
    Methods are demonstrated for synchronization of frequency with
    use of anneal offsets incorporating a simple decoherence model, and
    calibration refinement of a detector flux bias to restore symmetry.

    Higher accuracy calibration refinement, and calibration refinement of target
    flux_biases may also be desirable, but are beyond the scope of the example.
    Note that we can use simple statistics to determine flux-bias assignment on a
    detector relative to a target. E.g. a) when decoupled from the source and
    detector a 1 qubit model frequency omega=root(A(s)^2 + B(s)^2 h^2) is a convex
    monotonic function of the linear field, b) when decoupled from the source the
    response of the detector magnetization to a flux_bias perturbation is maximized.

    Args:
        cache_str:
            A unique experimental identifier. If not None, a directory
            cache/ is created which is populated with experimental data. The cache
            is checked for compatible experimental data before running an experiment,
            and if compatible data is present the data is reloaded rather than
            running new jobs through the client.
        solver:
            Name of the solver, or dictionary of characteristics.
        detector_lines:
            An iterable of integer indices of the detector lines.
        source_lines:
            An iterable of integer indices of the source lines.
        seed:
            Random seed used for embedding generation.
        max_num_embeddings:
            Maximum number of embeddings to find. If None, search for all available embeddings.
        target_c:
            Normalized control bias at which the target qubits are held.
            target_c is inferred from target_A, or defaulted such that A(target_c)=B(target_c).
            Either target_A or target_c should be specified, not both.
        target_A:
            The expected qubit frequency in GHz.
            When None target_A is inferred from target_c (and the given processor schedule).
            Either target_A or target_c should be specified, not both.
        target_B:
            The expected qubit frequency in GHz for the B field.
            When None target_B is inferred from target_c (and the given processor schedule).
            target_B is not used to parameterize experiments.
        dAdc:
            Approximate rate of change of A(c) with the normalized control bias c
            near target_c (GHz per unit c), used to convert a frequency discrepancy
            into an anneal offset. If None, it is estimated from the schedule file.
        flux_biases_method:
            When set to "None", flux_biases are not modified. When "Detector",
            flux_biases are modified on detector qubits to achieve zero expected
            magnetization at long delay. When "Target-Detector", flux_biases are
            modified on both target and detector qubits. Target-Detector calibration
            refinement can diverge, particularly for fast quenches of sources and
            detectors, large |Jtd| or |Jts|, and small target_A.
        source_decoupling_detection:
            When True, estimate the delay required for the detector to decouple from
            the source before collecting timeseries data. When False, this detection
            stage is skipped.
        num_anneal_offset_iterations:
            Number of anneal-offset refinement iterations to run. The first iteration
            estimates the required offsets; subsequent iterations re-measure and
            refine them. This verification stage is skipped when the value is 0.
            Anneal offsets are modified so that the peak power-spectral density is
            centered at a common target frequency for all qubits. This peak value
            characterizes the frequency of the target qubit in simple, well-calibrated
            models.
        schedule_fn: A schedule file that is used to estimate an appropriate sampling interval for delay
            time and an appropriate scale for anneal_offset synchronization. This should be matched to the
            solver.
        save_figures:
            When True, save generated figures to a ``figures`` folder.
        num_reads:
            The number of reads to perform for each measurement.
        use_common_c_bounds:
            When True, align the c-bounds of the generated schedules across annealing lines.
        use_overshoot:
            When True, use overshoot transitions for the source and detector quenches
            when building the multi-color annealing schedules. By default this is True
            to maximize the quench rate on source and detector.
        T2:
            Assumed decoherence (envelope) time scale used by the simple signal model
            (microseconds).
        Jtd: The coupling strength between target and detector qubits.
        Jts: The coupling strength between target and source qubits.
        preparation_orientation: The orientation of the target qubit whilst coupled
            to the source.
        dt_div_A_final: Sampling rate for the final (loop) experiment. Defaults
            to the same value used in anneal_offset shimming (1/4), smaller values
            can be used for pretty plots.
    Raises:
        ValueError: If the fit window (``delay_min_fit``, ``delay_max_fit``)
            is incompatible with the data window (``delay_min``, ``delay_max``)
            or empty; if neither or both of ``target_c`` and ``target_A`` are
            specified; if ``exp_feature_info`` has an unexpected (legacy)
            format; or if the fit window contains fewer than one sample.
        FileNotFoundError: If the QPU is offline and the fallback pickle
            caches for solver properties or experimental feature info are
            missing.
        RuntimeError: If the QPU is unavailable and no cached data is found
            for a stage that requires new sampling.
    """

    print()
    print(
        "This example currently serves as a guide to available functionality, and "
        "some simple heuristics, in a calibration refinement context. "
        "It is not a general purpose or canonical methodology. "
        "Correction of flux_biases, x_schedule_delays and anneal_offsets are correlated, "
        "and a high performance outcome may require an application-specific iterative "
        "approach with a larger number of programmings and reads than used in these examples."
    )
    print()
    print(
        "The example proceeds in several stages, per configurable parameters. "
        "Success in each stage is contingent on previous stages and a reasonable baseline calibration. "
        "The main stages are: "
        "embedding Target-Detector-Source systems and multi-color annealing sampling parameters; "
        "refining flux_biases to restore an anticipated Z2 symmetry at equilibrium; "
        "determining delays that characterize the preparation time (decoupling from the source); "
        "determining anneal_offsets that improve the accuracy of the target qubit frequencies."
    )
    print()
    print(
        "Note: To save experiment embeddings and data for replotting add the --use-cache flag."
    )
    # Schedule based approximations, target_A and dA/dc are approximated.
    stage_idx = -1

    if schedule_fn is None:
        if target_A is None or target_c is None or dAdc is None:
            raise ValueError(
                "Schedule file is required if target_A, target_c, or dAdc is not specified."
            )
    else:
        stage_idx += 1
        print()
        print(
            f"Stage {stage_idx}: Estimate consistent target_A, target_c and dA/dc from the given qpu schedule. "
            "These are used to parameterize the anneal_offset refinement method, and the accuracy threshold for x_schedule_delays refinements."
        )
        print(f"Schedule file used: {schedule_fn} ")
        print(
            "For calibration refinement of anneal offsets, dA/dc must be a reasonable match in the vicinity of target_A."
        )
        qpu_anneal_schedule = pd.read_excel(
            schedule_fn, sheet_name="Fast-Annealing Schedule"
        )

        plt.figure("Schedule")
        plt.title("Annealing Schedule")
        delta_vs_s = qpu_anneal_schedule[::-1]
        plt.plot(delta_vs_s["s"], delta_vs_s["A(s) (GHz)"], label="A(s)")
        plt.plot(delta_vs_s["s"], delta_vs_s["B(s) (GHz)"], label="B(s)")
        if target_c is None:
            if target_A is None:
                target_c = np.interp(
                    0.0,
                    -delta_vs_s["B(s) (GHz)"] + delta_vs_s["A(s) (GHz)"],
                    delta_vs_s["s"],
                )  # Expected normalized control bias at which to hold the qubits
            else:
                target_c = np.interp(
                    target_A, delta_vs_s["A(s) (GHz)"], delta_vs_s["s"]
                )  # Expected normalized control bias at which to hold the qubits
        elif target_A is not None:
            raise ValueError(
                "Specification of both target_c and target_A is not permitted when also specifying schedule_fn."
            )
        if target_A is None:
            target_A = np.interp(
                1 - target_c, 1 - delta_vs_s["s"], delta_vs_s["A(s) (GHz)"]
            )  # Expected frequency of detector magnetization oscillations
        target_B = np.interp(
            1 - target_c, 1 - delta_vs_s["s"], delta_vs_s["B(s) (GHz)"]
        )

        dtarget_c = 0.01
        target_Aminus = np.interp(
            1 - (target_c - dtarget_c), 1 - delta_vs_s["s"], delta_vs_s["A(s) (GHz)"]
        )
        target_Aplus = np.interp(
            1 - (target_c + dtarget_c), 1 - delta_vs_s["s"], delta_vs_s["A(s) (GHz)"]
        )
        if dAdc is None:
            dAdc = (target_Aplus - target_Aminus) / (2 * dtarget_c)
        plt.plot(
            [target_c, target_c],
            [0, np.max(delta_vs_s["A(s) (GHz)"])],
            label=f"c={target_c:.3g}",
        )
        plt.plot(
            [0, target_c - 0.01],
            [target_Aminus, target_Aminus],
            linestyle="dotted",
            color="black",
        )
        plt.plot(
            [target_c - 0.01, target_c - 0.01],
            [0, target_Aminus],
            linestyle="dotted",
            color="black",
        )
        plt.plot(
            [0, target_c + 0.01],
            [target_Aplus, target_Aplus],
            linestyle="dotted",
            color="black",
        )
        plt.plot(
            [target_c + 0.01, target_c + 0.01],
            [0, target_Aplus],
            linestyle="dotted",
            color="black",
        )
        plt.xlabel("Normalized control bias, c")
        plt.ylabel("Energy scale, GHz")
        plt.ylim([0, 2 * max(target_A, target_B)])
        plt.xlim([0, 1])
        plt.legend()
        print(
            f"target_c = {target_c:.3g} ",
            f"A(target_c) = {target_A:.3g} GHz ",
            f"B(target_c) = {target_B:.3g} GHz",
        )

    stage_idx += 1
    print()
    print(
        f"Stage {stage_idx}: Build multi color annealing waveforms compatible with the "
        f"research QPU."
    )

    if cache_str:
        qpu_fn = f"cache/qpu_{cache_str}.pkl"
    try:
        qpu = DWaveSampler(solver=solver)
        print(
            f"Solver connected to (check matches schedule file): {qpu.solver.identity}"
        )
        exp_feature_info = get_properties(qpu)
        if cache_str:
            with open(qpu_fn, "wb") as f:
                pickle.dump((qpu.properties, exp_feature_info), f)
        online = True
    except Exception as error:
        if not cache_str:
            raise (error)
        elif not os.path.isfile(qpu_fn):
            raise FileNotFoundError(
                f"use_cache=True, but cache files are missing and no client "
                f"is available: {error}"
            )
        else:
            with open(qpu_fn, "rb") as f:
                properties, exp_feature_info = pickle.load(f)
        qpu = MockDWaveSampler(
            properties=properties,
            nodelist=properties["qubits"],
            edgelist=properties["couplers"],
        )
        online = False
    if len(exp_feature_info) != 2:
        raise ValueError("Legacy format")
    line_assignments = {
        n: al_idx for al_idx, al in enumerate(exp_feature_info[1]) for n in al["qubits"]
    }
    num_lines = len(exp_feature_info[1])
    # 2/3 of lines reserved for the target (typical applications)
    if (detector_lines is None) != (source_lines is None):
        raise ValueError(
            "detector_lines and source_lines must either both be specified or both be None."
        )
    elif detector_lines is None:
        if num_lines <= 6:
            # 6 line convention
            detector_lines = (0,)
            source_lines = (num_lines // 2,)
        else:
            # 12 line convention
            detector_lines = (0, num_lines // 4 + 1)
            source_lines = (num_lines // 2, (3 * num_lines) // 4 + 1)
    if target_lines is None:
        target_lines = set(range(num_lines)) - set(detector_lines) - set(source_lines)

    cmap = plt.colormaps.get_cmap("plasma")

    line_color = [cmap(i / (num_lines - 1)) for i in range(num_lines)]

    x_anneal_schedules, x_polarizing_schedule = make_tds_x_schedules(
        exp_feature_info=exp_feature_info,
        target_lines=target_lines,
        target_c=target_c,
        detector_lines=detector_lines,
        source_lines=source_lines,
        use_common_bounds=use_common_c_bounds,
        use_overshoot=use_overshoot,
        sign_polarization=-np.sign(Jts * Jtd * preparation_orientation),
    )
    _plot_tds_schedules(
        x_polarizing_schedule,
        x_anneal_schedules,
    )
    if save_figures:
        _save_open_figures("figures/", cache_str)
    print("Close figures to proceed to next stages.")
    _apply_tight_layout()
    plt.show()

    # x_schedule_delays = make_tds_x_schedule_delays(
    #    x_anneal_schedules=x_anneal_schedules,
    #    quenched_lines=set(detector_lines) | set(source_lines),
    #    target_c=target_c,
    #    decimal_places=6,
    # )  # Quench rates implied by linear PWL are unreliable, especially with overshoot.
    x_schedule_delays = [0.0] * num_lines

    anneal_offsets = [0.0] * qpu.properties["num_qubits"]
    flux_biases = [0.0] * qpu.properties["num_qubits"]
    sampling_params = dict(
        num_reads=num_reads,
        answer_mode="raw",
        x_disable_filtering=True,
        x_schedule_delays=x_schedule_delays,
        x_anneal_schedules=x_anneal_schedules,
        x_polarizing_schedule=x_polarizing_schedule,
        flux_biases=flux_biases,
        anneal_offsets=anneal_offsets,
        auto_scale=False,
    )
    stage_idx += 1
    print()
    print(f"Stage {stage_idx}: Find T-D-S embeddings for parallel programming")
    print("(see mca_embedding.py example).")
    print(
        "Finding embeddings is NP-hard in general. The routine attempts to find many "
        f"such embeddings that can be programmed in parallel. If no embedding is found "
        f"within the (CLI configurable) timeout of {embedding_timeout} seconds, the search is terminated."
    )

    T = qpu.to_networkx_graph()

    def _target_assignments(n: int) -> str:
        """Classify a qubit as "detector", "source", or "target" by its line."""
        line = line_assignments[n]
        if line in detector_lines:
            return "detector"
        elif line in source_lines:
            return "source"
        elif line in target_lines:
            return "target"
        else:
            return "unknown"

    Tnode_to_tds = {n: _target_assignments(n) for n in qpu.nodelist}
    target_graph = nx.Graph()
    target_graph.add_node(0)
    # Embeddings are reprocessed, to be treated as independent.
    S, Snode_to_tds = make_tds_graph(target_graph)
    if loop_length is not None:
        if not (loop_length >= 4 and loop_length % 2 == 0):
            raise ValueError(
                "loop_length, when specified, must be an even number greater than or equal to 4."
            )

        target_graph_experiment = nx.Graph()
        target_graph_experiment.add_nodes_from(range(loop_length))
        target_graph_experiment.add_edges_from(
            (i, (i + 1) % loop_length) for i in range(loop_length)
        )
        S_experiment, Snode_to_tds = make_tds_graph(target_graph_experiment)
    else:
        S_experiment = S

    fn_cache = f"cache/emb_{cache_str}.pkl"
    if cache_str:
        os.makedirs(os.path.dirname(fn_cache), exist_ok=True)

    if cache_str and os.path.isfile(fn_cache):
        with open(fn_cache, "rb") as f:
            embs_experiment = pickle.load(f)
    else:
        subgraph_kwargs = dict(
            node_labels=(Snode_to_tds, Tnode_to_tds),
            as_embedding=True,
            timeout=embedding_timeout,
        )
        embs_experiment = find_multiple_embeddings(
            S_experiment,
            T,
            max_num_emb=max_num_embeddings,
            embedder_kwargs=subgraph_kwargs,
            one_to_iterable=True,
            seed=seed,
            timeout=embedding_timeout,
        )
        if len(embs_experiment) == 0:
            raise RuntimeError(
                "No embeddings were found, try a larger embedding_timeout, simpler target (smaller or no num_loops) or new line combination."
            )
        else:
            with open(fn_cache, "wb") as f:
                pickle.dump(embs_experiment, f)

    if loop_length is not None:
        print(
            f"{len(embs_experiment)} independent length-{loop_length} target loops were found, where each target qubit is connected to both a source and a detector. "
            f"In the calibration refinement stages these are treated as {len(embs_experiment)}x{loop_length} independent S-D-T systems."
        )
        embs = _to_independent_tds(embs_experiment)
    else:
        embs = embs_experiment
    print(
        f"{len(embs)} T-D-S placements were found; each is calibrated with parallelized data collection. "
        "These are ordered by target line for purposes of visualization."
    )
    embs_by_line = {i: [] for i in target_lines}
    for i, emb in enumerate(embs):
        q = emb[0][0]
        embs_by_line[line_assignments[q]].append(emb)

    embs = [emb for i in target_lines for emb in embs_by_line[i]]
    n_embs = len(embs)

    sampler = ParallelEmbeddingComposite(qpu, embeddings=embs)

    bqm = dimod.BinaryQuadraticModel("SPIN").from_ising(
        {n: 0 for n in S.nodes()},
        {e: Jtd for e in S.edges() if any(Snode_to_tds[v] == "detector" for v in e)}
        | {e: Jts for e in S.edges() if any(Snode_to_tds[v] == "source" for v in e)},
    )  # bqm restricted to decoupled source and target nodes
    bqm_td = dimod.BinaryQuadraticModel("SPIN").from_ising(
        {n: 0 for n in S.nodes() if Snode_to_tds[n] != "source"},
        {e: Jtd for e in S.edges() if any(Snode_to_tds[v] == "detector" for v in e)},
    )  # Relevant to flux shimmi
    if flux_biases_method != "None":
        stage_idx += 1
        print()
        print(f"Stage {stage_idx}: Refine flux_biases")

        x_polarizing_schedule = sampling_params.pop("x_polarizing_schedule")
        fn_cache = f"cache/FB_{cache_str}.npy"
        if cache_str and os.path.isfile(fn_cache):
            with open(fn_cache, "rb") as f:
                flux_biases, flux_history, mag_history = pickle.load(f)
        else:
            if not online:
                raise RuntimeError("QPU not available, and no cached data found.")
            # Require zero magnetization in the limit of long delay (where
            # source impact has decayed away.
            bqm_embedded = dimod.BinaryQuadraticModel("SPIN").from_ising(
                {emb[n][0]: h for emb in embs for n, h in bqm_td.linear.items()},
                {
                    tuple(emb[n][0] for n in e): J
                    for emb in embs
                    for e, J in bqm_td.quadratic.items()
                },
            )
            shimmed_variables = {
                n
                for n in bqm_embedded.variables
                if line_assignments[n] in detector_lines
            }
            x_polarizing_schedule = sampling_params.pop(
                "x_polarizing_schedule", None
            )  # Remove polarizing signal
            if flux_biases_method == "Target-Detector":
                print(
                    "Refine target and detector flux_biases for unbiased target and detector "
                    "qubits at equilibrium with sources decoupled and depolarized."
                )

                flux_biases, flux_history, mag_history = shim_tds_flux_biases(
                    bqm=bqm_embedded,
                    sampler=qpu,
                    sampling_params=sampling_params,
                    target_lines=set(target_lines),
                    detector_lines=set(detector_lines),
                    line_assignments=line_assignments,
                )
            elif flux_biases_method == "Detector":
                print(
                    "Refine detector flux_biases for unbiased detector magnetization with"
                    " sources decoupled and depolarized."
                )
                flux_biases, flux_history, mag_history = shim_flux_biases(
                    bqm=bqm_embedded,
                    sampler=qpu,
                    sampling_params=sampling_params,
                    shimmed_variables=shimmed_variables,
                )
            else:
                raise ValueError("Unknown method")

            polarization_candidates = [
                (i, flux_biases[i])
                for i in range(len(flux_biases))
                if abs(flux_biases[i]) > 1e-4
            ]
            if polarization_candidates:
                print(
                    "WARNING: Anomalously large flux biases could indicate "
                    "a calibration issue, check magnetization plots "
                    "for evidence of polarization and report bad qubits."
                )
                print(polarization_candidates)

            if x_polarizing_schedule is not None:
                sampling_params["x_polarizing_schedule"] = x_polarizing_schedule
            if cache_str:
                with open(fn_cache, "wb") as f:
                    pickle.dump((flux_biases, flux_history, mag_history), f)
        plot_shim(
            mag_history,
            flux_history,
        )
        sampling_params["flux_biases"] = flux_biases
        if save_figures:
            _save_open_figures("figures/", cache_str)
        print("Close figures to proceed to next stages.")
        _apply_tight_layout()
        plt.show()
        sampling_params["x_polarizing_schedule"] = x_polarizing_schedule

    if t_decoupled is None:
        stage_idx += 1
        print()
        print(
            f"Stage {stage_idx}: Detect delay for source decoupling (for bulk of {n_embs} parallel embeddings)"
        )
        print(
            "Whilst the source is coupled a polarized signal is detected. "
            "Larmor precession proceeds from the point where decoupling occurs; "
            "determine this processor- and anneal-schedule-specific value."
        )
        if len(detector_lines) > 1 or len(source_lines) > 1:
            print(
                "WARNING: Multiple detector or source lines detected. Delays could vary by line combination. "
                "A feature to handle this branching is not yet implemented, so the bulk delay calculated might "
                "be an inappropriate middle ground."
            )

        fn_cache = f"cache/source_decoupling_{cache_str}.pkl"
        if cache_str and os.path.isfile(fn_cache):
            with open(fn_cache, "rb") as f:
                t_decoupled, t_mags = pickle.load(f)
        else:
            t_decoupled, t_mags = estimate_decoupling_timescale(
                sampler=sampler,
                bqm=bqm,
                sampling_params=sampling_params,
                detector_lines=detector_lines,
                target_A=target_A * 1000,
            )
            if cache_str:
                with open(fn_cache, "wb") as f:
                    pickle.dump((t_decoupled, t_mags), f)
        plt.figure("source_decoupling")
        x = [t for t, _ in t_mags]
        y = [mag for _, mag in t_mags]
        plt.plot(
            x,
            y,
            linestyle="None",
            marker=".",
        )
        plt.xlabel("Time")
        plt.ylabel("Magnitude")
        plt.title("Source Decoupling Detection")
        plt.axvline(
            t_decoupled, color="r", linestyle="--", label="Estimated Decoupling Time"
        )
        plt.legend()
        if save_figures:
            _save_open_figures("figures/", cache_str)
        print("Close figures to proceed to next stages.")
        _apply_tight_layout()
        plt.show()

    # Collecting regularly spaced data on the interval [0, T2], post
    # source decoupling, is not optimal for inference of calibration
    # errors, but (IMO) allows intuitive estimators and data series.
    # Nyquist frequency in MHz, resolving up to 2*target_A.
    nyquist_frequency = target_A * 1000 * 2

    delay_min_fit = delay_min = t_decoupled + 1 / (
        target_A * 1000
    )  # Ignore first cycle.
    delay_max_fit = delay_max = delay_min + T2
    delays = np.linspace(
        delay_min,
        delay_max,
        round((delay_max - delay_min) * nyquist_frequency * 2) + 1,
    )
    dt = delays[1] - delays[0]

    if num_anneal_offset_iterations > 0:
        stage_idx += 1
        print()
        print(f"Stage {stage_idx}: Anneal offset refinement")
        print(
            "Collecting data in the interval ~[0, T2] at a rate close to twice the Nyquist frequency. "
            "This allows the power-spectral density to be estimated in good agreement with a Lorentzian. "
            "The model target frequency can be approximated from the peak to reasonable precision. "
            "Methods of higher accuracy are available, for example by exploiting phase information and "
            "collecting data at higher sampling rates, but this power spectral density method is "
            "sufficient to resolve discrepancies up to a scale O(0.01GHz) relevant to calibration refinement. "
        )
        print()
        print(f"Stage {stage_idx}a: Some model data.")
        print(
            "In figures, ideal error-free high density data is shown as dashed lines, with (coarsely) sampled "
            "data intended to mimic some practical experimental limitations."
        )
        delay_max_art = T2
        delay_min_art = 0.0
        delays_art = np.linspace(
            delay_min_art,
            delay_max_art,
            round((delay_max_art - delay_min_art) * (2 * nyquist_frequency)) + 1,
            endpoint=True,
        )
        dt_art = delays_art[1] - delays_art[0]  # Rounding

        dt_hd = 0.00001  # 0.01 nanoseconds, close to practical limit.
        high_density_delays = np.linspace(
            delay_min_art,
            delay_max_art,
            round((delay_max_art - delay_min_art) / dt_hd),
            endpoint=False,
        )
        dt_hd = high_density_delays[1] - high_density_delays[0]  # Rounding

        # theta_s = theta_d = pi/2, other parameters are pertured (should revisit for completeness).
        # note that the random delay, and random phi_s-phi_d deviations are qualitatively
        # captured by a single time delay.
        for A in [target_Aminus, target_A, target_Aplus]:
            delay_perturbation = 1 / (A * 1000) * np.random.random()
            T2_perturbed = 0.0101 * (1 + 0.1 * np.random.random())
            label = f"A={A:.3g}"
            theta_s = np.pi / 2 * preparation_orientation
            signal = artificial_data(
                delays_art + delay_perturbation,
                A * 1000,
                num_independent_samples=num_reads,
                theta_s=theta_s,
                T2=T2_perturbed,
            )
            ideal_high_density_signal = artificial_data(
                high_density_delays + delay_perturbation,
                A * 1000,
                num_independent_samples=float("Inf"),  # No noise
                theta_s=theta_s,
            )
            fig = plt.figure("artificial_timeseries")
            next_color = fig.gca()._get_lines.get_next_color()
            plt.title("y=cos(2pi A t)exp(-t/T)+sampling error")
            plt.plot(
                (delays_art + delay_perturbation) * 1000,
                signal,
                label=label,
                marker=".",
                linestyle=None,
                color=next_color,
            )
            plt.plot(
                (high_density_delays + delay_perturbation) * 1000,
                ideal_high_density_signal,
                linestyle="dotted",
                color=next_color,
            )
            plt.xlabel("Time, nanoseconds")
            plt.ylabel(r"Magnetization, $\langle y \rangle_{detector}$")
            plt.legend()

            plt.figure("artificial_psd")
            psd_title = "Approximate Lorentzian PSD ~ A/((f-A)^2 + A^2)"
            plt.title(psd_title)
            frequencies = (
                np.arange(len(delays_art) // 2) / dt_art / len(delays_art) / 1000
            )
            psd = np.abs(np.fft.fft(signal)) ** 2 / len(signal) ** 2
            frequencies_hd = (
                np.arange(len(high_density_delays) // 2)
                / dt_hd
                / len(high_density_delays)
                / 1000
            )
            psd_hd = (
                np.abs(np.fft.fft(ideal_high_density_signal)) ** 2
                / len(ideal_high_density_signal) ** 2
            )
            plt.plot(
                frequencies,
                psd[: len(psd) // 2],
                label=label,
                marker=".",
                linestyle=None,
                color=next_color,
            )
            plt.plot(
                frequencies_hd,
                psd_hd[: len(psd_hd) // 2],
                linestyle="dotted",
                color=next_color,
            )
            plt.ylabel(r"Power Spectral Density, $|\langle {\hat y}\rangle(\omega)|^2$")
            plt.xlabel(r"Frequency ($\omega$), GHz")
            plt.xlim([0, frequencies[-1]])
            plt.legend()

        if save_figures:
            _save_open_figures("figures/", cache_str)
        print("Close figures to proceed to next (experimental) stages.")
        _apply_tight_layout()
        plt.show()

        print()
        print(
            f"Stage {stage_idx}b: Estimate anneal offsets required to achieve the target frequency {target_A:.3g}GHz (for all {n_embs} parallel embeddings)."
        )
        print("Anneal offset using a linear model based upon the provided schedule.")

        fn_cache = f"cache/AO_It0_{cache_str}.npy"
        if cache_str and os.path.isfile(fn_cache):
            mean_Z_detector = np.load(fn_cache)
        else:
            if not online:
                raise RuntimeError("QPU not available, and no cached data found.")

            mean_Z_detector = run_parallel_experiment(
                sampler, bqm, sampling_params, delays, detector_lines
            )
            if cache_str:
                np.save(fn_cache, mean_Z_detector)

        first = np.argmax(delays >= delay_min_fit)
        last = np.argmax(delays >= delay_max_fit) + 1
        ld = last - first
        if ld < 1:
            raise ValueError("Fit window is empty: t-fit range too small for target_A")

        frequencies = np.arange(ld) / dt / 1000 / ld  # GHz
        psd = np.array(
            [
                np.abs(np.fft.fft(mean_Z_detector[first:last, i])) ** 2
                for i in range(len(embs))
            ]
        ) / (last - first)

        # Plot data #
        line_exemplars = {
            line_assignments[emb[0][0]]: idx for idx, emb in enumerate(embs)
        }
        timeseries_fig, (ax_first_iter_timeseries, ax_second_iter_timeseries) = (
            plt.subplots(
                1, 2, figsize=(12, 5), num="Timeseries", constrained_layout=True
            )
        )
        timeseries_fig.suptitle(
            "Time series for several qubits using distinct target lines"
        )
        ax_first_iter_timeseries.set_title("Before anneal-offset refinement")
        ax_second_iter_timeseries.set_title("After anneal-offset refinement")
        _plot_time_series(
            embs,
            line_assignments,
            mean_Z_detector,
            delays * 1000,
            line_color,
            plotted_emb_idxs=line_exemplars.values(),
            label_emb_idxs=line_exemplars.values(),
            ax=ax_first_iter_timeseries,
        )
        heatmap_after_axes = {}

        heatmap_fig, (ax_first_iter_heatmap, ax_second_iter_heatmap) = plt.subplots(
            1,
            2,
            figsize=(12, 5),
            num=f"Timeseries_{colormap_type}_colormap",
            constrained_layout=True,
        )
        heatmap_fig.suptitle(f"Detector magnetization ({colormap_type} colormap)")
        ax_first_iter_heatmap.set_title("Before anneal-offset refinement")
        ax_second_iter_heatmap.set_title("After anneal-offset refinement")
        imshow_data(
            mean_Z_detector=mean_Z_detector,
            delays=delays,
            colormap_type=colormap_type,
            first=first,
            last=last,
            ax=ax_first_iter_heatmap,
        )
        heatmap_after_axes[colormap_type] = ax_second_iter_heatmap

        psd_fig, (ax_first_iter_psd, ax_second_iter_psd) = plt.subplots(
            1, 2, figsize=(12, 5), num="PSD", constrained_layout=True
        )
        psd_fig.suptitle("Power associated with magnetization time series")
        ax_first_iter_psd.set_title("Before anneal-offset refinement")
        ax_second_iter_psd.set_title("After anneal-offset refinement")
        _plot_time_series(
            embs,
            line_assignments,
            psd[:, : ld // 2].T,
            frequencies[: ld // 2],
            line_color,
            label_emb_idxs=line_exemplars.values(),
            xlabel=r"Frequency ($\omega$), GHz",
            ylabel=r"Power Spectral Density, $|\langle Z\rangle(\omega)|^2$",
            ax=ax_first_iter_psd,
        )
        ax_first_iter_psd.plot(
            [target_A, target_A],
            [0, np.max(psd)],
            color="black",
            linestyle="dashed",
            label="Schedule prediction",
        )
        ax_first_iter_psd.legend()

        # Calculate anneal_offsets for synchronization
        anneal_offsets = y = _calc_anneal_offsets(
            frequencies, psd, target_A, dAdc
        )  # Per embedding
        # anneal offsets can be realized line-wise by changing target_c,
        # or qubit-wise by modification of anneal_offset. We can
        # correct for the mean with a line_offset, and then qubit-wise
        # variation with the anneal offset.

        plt.figure("Proposed anneal_offsets")
        plt.plot(
            sorted(y),
            np.arange(len(y)) / len(y),
            label=f"RMS(It=1)={np.sqrt(np.mean(np.array(y)**2)):.3g}",
        )
        plt.xlabel(f"Proposed anneal offset")
        plt.ylabel("Cumulative distribution function")
        plt.legend()
        if num_anneal_offset_iterations == 1:
            if save_figures:
                _save_open_figures("figures/", cache_str)
            print("Close figures to proceed to next stages.")
            _apply_tight_layout()
            plt.show()

    if num_anneal_offset_iterations > 1:
        anneal_offsets0 = anneal_offsets
        print()
        print(
            f"Stage {stage_idx}c: Apply second iterative stage (and demonstrate improvements in target_A homogeneity)."
        )
        fn_cache = f"cache/AO_It1_{cache_str}.npy"
        if cache_str and os.path.isfile(fn_cache):
            mean_Z_detector = np.load(fn_cache)
        else:
            if not online:
                raise RuntimeError("QPU not available, and no cached data found.")
            for emb, ao in zip(embs, anneal_offsets):
                sampling_params["anneal_offsets"][
                    emb[0][0]
                ] -= ao  # Apply correction to target on each embedding
            mean_Z_detector = run_parallel_experiment(
                sampler, bqm, sampling_params, delays, detector_lines
            )
            if cache_str:
                np.save(fn_cache, mean_Z_detector)
        psd = np.array(
            [
                np.abs(np.fft.fft(mean_Z_detector[first:last, i])) ** 2
                for i in range(len(embs))
            ]
        ) / (last - first)
        y = anneal_offsets = _calc_anneal_offsets(
            frequencies, psd, target_A, dAdc
        )  # Per embedding

        for emb, ao in zip(embs, anneal_offsets):
            sampling_params["anneal_offsets"][
                emb[0][0]
            ] -= ao  # Apply correction to target on each embedding

        plt.figure("Proposed anneal_offsets")
        plt.plot(
            sorted(y),
            np.arange(len(y)) / len(y),
            label=f"RMS(It=2)={np.sqrt(np.mean(np.array(y)**2)):.3g}",
        )
        plt.legend()

        _plot_time_series(
            embs,
            line_assignments,
            mean_Z_detector,
            delays * 1000,
            line_color,
            plotted_emb_idxs=line_exemplars.values(),
            label_emb_idxs=line_exemplars.values(),
            ax=ax_second_iter_timeseries,
        )

        imshow_data(
            mean_Z_detector=mean_Z_detector,
            delays=delays,
            colormap_type=colormap_type,
            first=first,
            last=last,
            ax=heatmap_after_axes[colormap_type],
        )

        _plot_time_series(
            embs,
            line_assignments,
            psd[:, : ld // 2].T,
            frequencies[: ld // 2],
            line_color,
            label_emb_idxs=line_exemplars.values(),
            xlabel=r"Frequency ($\omega$), GHz",
            ylabel=r"Power Spectral Density, $|\langle Z\rangle(\omega)|^2$",
            ax=ax_second_iter_psd,
        )
        ax_second_iter_psd.plot(
            [target_A, target_A],
            [0, np.max(psd)],
            color="black",
            linestyle="dashed",
            label="Schedule prediction",
        )
        ax_second_iter_psd.legend()

        plt.figure("AnnealOffsets")
        legend_idxs = set(line_exemplars.values())
        for emb_idx, emb in enumerate(embs):
            q = emb[0][0]
            line_target = line_assignments[q]
            if emb_idx in legend_idxs:
                label = f"target-qubit line={line_target}"
            else:
                label = None
            plt.plot(
                anneal_offsets0[emb_idx],
                anneal_offsets[emb_idx],
                color=line_color[line_target],
                marker="x",
                label=label,
            )
        plt.xlabel("Estimated anneal-offset correction (baseline)")
        plt.ylabel("Estimated anneal-offset correction (after refinement)")
        plt.grid(True)
        plt.legend()

    if save_figures:
        _save_open_figures("figures", cache_str)
    _apply_tight_layout()
    plt.show()
    print(
        "If --loop_length was specified, close figures to proceed to next stages: pi/2-pulse propagation."
    )

    if loop_length:
        stage_idx += 1
        print()
        print(f"Stage {stage_idx}: pi/2 pulse propagation.")
        print(
            f"Target qubits are coupled with Jtt={Jtt} in loops of length {loop_length}. "
        )
        print(
            "A pi/2 pulse is applied to the first (0 indexed) qubit. "
            "targets are then measured in the (same) computational basis subject to delay. "
        )
        print(
            "We anticipate excitation propagation around the loop subject to interference. "
        )
        if target_B is not None:
            print(
                f"Propagation occurs on a time scale  ~ 1/(B(s) |Jtt|) = {1/np.abs(target_B * Jtt):.3g}ns"
            )
        sampler_experiment = ParallelEmbeddingComposite(qpu, embeddings=embs_experiment)
        bqm_experiment = dimod.BinaryQuadraticModel("SPIN").from_ising(
            {n: 0 for n in S_experiment.nodes()},
            {
                e: Jtd
                for e in S_experiment.edges()
                if any(Snode_to_tds[v] == "detector" for v in e)
            }
            | {
                e: Jts
                for e in S_experiment.edges()
                if any((Snode_to_tds[v] == "source" and v[1] == 0) for v in e)
            }
            | {
                e: Jtt
                for e in S_experiment.edges()
                if all(Snode_to_tds[v] == "target" for v in e)
            },
        )  # bqm restricted to decoupled source and target nodes
        fn_cache = f"cache/LoopExperiment_{cache_str}.npy"

        if dt_div_A_final != 0.25:
            experiment_dt = dt_A_final / A_target / 1000
            delays = np.linspace(
                delay_min,
                delay_max,
                round((delay_max - delay_min) / experiment_dt) + 1,
            )
            dt = delays[1] - delays[0]

        if cache_str and os.path.isfile(fn_cache):
            mean_Z_detector = np.load(fn_cache)
        else:
            mean_Z_detector = run_parallel_experiment(
                sampler_experiment,
                bqm_experiment,
                sampling_params,
                delays,
                detector_lines,
            )
            if cache_str:
                np.save(fn_cache, mean_Z_detector)
        square_data = mean_Z_detector.reshape(mean_Z_detector.shape[0], -1)
        print(
            "Note: The x-axis is ordered by position on the ring, rather than target line assignment (per earlier plots). "
            "The excitation propagates in two directions (left and right, modulo the periodic boundary condition)"
        )
        if len(embs_experiment) > 1:
            print(
                f"The plot is divided into {len(embs_experiment)} panels, reflecting independent (decoupled) embeddings that were programmed in parallel"
            )
        imshow_data(
            square_data,
            delays=delays,
            context_str=f"coupled loop length={loop_length}",
        )
        ax = plt.gca()
        rows, cols = square_data.shape
        line_x = np.arange(0.5 + loop_length, cols - 0.5, loop_length)
        ax.vlines(x=line_x, ymin=-0.5, ymax=rows - 0.5, colors="black", linewidth=1.5)
        plt.title(
            f"{len(embs_experiment)} target loops of length {loop_length}, each with pi/2-pulse at origin."
        )
        if save_figures:
            _save_open_figures("figures", cache_str)
        _apply_tight_layout()
        plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Target-detector-source embedding demo with optional flux-bias "
            "calibration refinement and anneal-offset refinement."
        )
    )
    parser.add_argument(
        "--use-cache",
        dest="use_cache",
        action="store_true",
        help=(
            "Cache and reload experiment artifacts keyed by CLI parameters. "
            "If solver graph_id changes, cached embeddings may be invalid."
        ),
    )
    parser.add_argument(
        "--solver-name",
        dest="solver_name",
        type=str,
        help="QPU solver name. Default research system with fast reverse anneal.",
        default=SOLVER_FILTER,
    )
    parser.add_argument(
        "--detector-lines",
        dest="detector_lines",
        type=int,
        nargs="+",
        help="Detector lines (one or more integer indices).",
        default=None,  # First vertical qubit line
    )
    parser.add_argument(
        "--source-lines",
        dest="source_lines",
        type=int,
        nargs="+",
        help="Source lines (one or more integer indices).",
        default=None,  # First horizontal qubit line under 6-line control
    )
    parser.add_argument(
        "--target-lines",
        dest="target_lines",
        type=int,
        nargs="+",
        help="Target lines (one or more integer indices).",
        default=None,  # First horizontal qubit line under 6-line control
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for embedding generation.",
        default=None,
    )
    parser.add_argument(
        "--max-num-embeddings",
        dest="max_num_embeddings",
        type=int,
        help="Max embeddings to find (default: all available).",
        default=None,
    )
    parser.add_argument(
        "--target-A",
        dest="target_A",
        type=float,
        help="Expected qubit frequency (GHz). "
        "(default A(target_c) such that A(target_c)=B(target_c), which is approximately 2 GHz)",
        default=None,
    )
    parser.add_argument(
        "--schedule-fn",
        dest="schedule_fn",
        type=str,
        help="Path to the annealing schedule Excel file (.xlsx). Should be matched to the solver.",
        default="09-1323A-D_Advantage2_system4_annealing_schedule.xlsx",
    )
    parser.add_argument(
        "--Jts",
        dest="Jts",
        type=float,
        help="Coupling strength between target and source qubits. "
        "The AFM value maximizing the effective quench rate is used. "
        "Note that, an FM value (-2.0) can achieve higher quench rates still but at risk of enhanced control errors",
        default=1.0,
    )
    parser.add_argument(
        "--Jtd",
        dest="Jtd",
        type=float,
        help="Coupling strength between target and detector qubits. "
        "The AFM value maximizing the effective quench rate is used. "
        "Note that, an FM value (-2.0) can achieve higher quench rates still but at risk of enhanced control errors",
        default=1.0,
    )
    parser.add_argument(
        "--Jtt",
        dest="Jtt",
        type=float,
        help="Coupling strength between target and target qubits. The parameter is ignored unless a loop length is specified. "
        "Note that a sufficiently weak coupling limit [B(s_target) J_tt << A(s_target)] is required for independent sourcing and detection of qubits.",
        default=-0.1,
    )
    parser.add_argument(
        "--loop-length",
        dest="loop_length",
        type=int,
        help="Length of the loop model (even and >=4 if specified). "
        "By default, independent T-D-S systems are modeled (a Larmour precession example)."
        "If N is specified then T-D-S systems compatible with unfrustrated loops of length N are embedded.",
        default=None,
    )
    parser.add_argument(
        "--flux-bias-method",
        dest="flux_biases_method",
        type=str,
        choices=["None", "Detector", "Target-Detector"],
        default="Detector",
        help="Flux-bias calibration refinement mode: 'None' disables calibration refinement; "
        "'Detector' refines detector qubit flux_biases to achieve zero measured magnetization; "
        "'Target-Detector' alternates detector/target roles to refine detector and target flux_biases "
        " (this can cause divergences, particularly at small frequencies and "
        "when target_c is desynchronized).",
    )
    parser.add_argument(
        "--t-decoupling",
        dest="t_decoupled",
        type=float,
        default=None,
        help="Delay required on the detector relative to the source in order that the target is decoupled.",
    )
    parser.add_argument(
        "--num_anneal_offset_iterations",
        dest="num_anneal_offset_iterations",
        type=int,
        help="Number of anneal-offset iterations to perform. Note that a second iteration is useful to verify the first iteration (thus 2 by default).",
        default=2,
    )
    parser.add_argument(
        "--common-c-bounds",
        dest="use_common_c_bounds",
        action="store_true",
        default=False,
        help="Enable common c-bounds alignment across annealing lines. "
        "This is necessary for the Target-Detector/TDS calibration-refinement method.",
    )
    parser.add_argument(
        "--no-overshoot",
        dest="use_overshoot",
        action="store_false",
        default=True,
        help="Disable overshoot transitions for the source and detector quenches.",
    )
    parser.add_argument(
        "--disable-save-figures",
        dest="save_figures",
        action="store_false",
        help="Disable saving figures to figures/ folder with hash-based names.",
    )

    args = parser.parse_args()
    if args.use_cache:
        cache_str = _get_experiment_id(args, num_char=8)
    else:
        cache_str = None
    main(
        cache_str=cache_str,
        solver=args.solver_name,
        detector_lines=args.detector_lines,
        source_lines=args.source_lines,
        target_lines=args.target_lines,
        target_A=args.target_A,
        schedule_fn=args.schedule_fn,
        t_decoupled=args.t_decoupled,
        num_anneal_offset_iterations=args.num_anneal_offset_iterations,
        flux_biases_method=args.flux_biases_method,
        use_common_c_bounds=args.use_common_c_bounds,
        use_overshoot=args.use_overshoot,
        save_figures=args.save_figures,
        Jts=args.Jts,
        Jtd=args.Jtd,
        Jtt=args.Jtt,
        loop_length=args.loop_length,
    )
