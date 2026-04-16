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
An example to show embedding for multicolor annealing.
"""
import argparse
import hashlib
import json
import os
from typing import Iterable

import pickle
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from dwave.experimental.shimming.flux_biases import shim_flux_biases
from dwave.experimental.shimming.flux_biases import shim_tds_flux_biases
import numpy as np
from tqdm import tqdm

import dimod
from dwave.system import DWaveSampler
from dwave.system.composites import ParallelEmbeddingComposite

from minorminer.utils.parallel_embeddings import find_multiple_embeddings
from dwave.experimental.multicolor_anneal import (
    get_properties,
    SOLVER_FILTER,
    make_tds_graph,
    qubit_to_Advantage2_annealing_line,
)
from dwave.experimental.shimming import shim_flux_biases


def _get_experiment_id(args):
    args_string = json.dumps(vars(args), sort_keys=True)
    return hashlib.sha256(args_string.encode("utf-8")).hexdigest()


def _make_anneal_schedules(
    exp_feature_info: list,
    target_c: float = 0.37,
    times: list[float] | tuple[float] = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
    detector_lines: Iterable[int] = (0,),
    source_lines: Iterable[int] = (3,),
    target_lines: Iterable[int] | None = None,
    use_common_bounds: bool = True,
):
    """Set annealing schedules suitable for Larmour precision.

    See documentation for Larmour precession example, the same
    schedule is used.
    """
    source_lines = set(source_lines)
    detector_lines = set(detector_lines)
    num_lines = len(exp_feature_info)

    assert (
        source_lines.isdisjoint(detector_lines)
        and source_lines.issubset(set(range(num_lines)))
        and detector_lines.issubset(set(range(num_lines)))
    ), "Source and detector lines must be disjoint and valid line indices."

    min_time_step = max(
        exp_feature_info[line]["minAnnealingTimeStep"] for line in range(num_lines)
    )
    if len(times) != 7 or np.min(np.diff(times)) < 2 * min_time_step:
        raise ValueError("Format assumes 7 times each separated by atleast 2 minStep")

    if use_common_bounds:
        maxC = min(exp_feature_info[line]["maxC"] for line in range(num_lines))
        minC = max(exp_feature_info[line]["minC"] for line in range(num_lines))
        maxCs = {line: maxC for line in range(num_lines)}
        minCs = {line: minC for line in range(num_lines)}
        min_time_steps = {line: min_time_step for line in range(num_lines)}
    else:
        maxCs = {line: exp_feature_info[line]["maxC"] for line in range(num_lines)}
        minCs = {line: exp_feature_info[line]["minC"] for line in range(num_lines)}
        min_time_steps = {
            line: exp_feature_info[line]["minAnnealingTimeStep"]
            for line in range(num_lines)
        }
    # By default, lines are chosen to minimize interference
    anneal_schedules = [
        [
            [times[0], minCs[line]],
            [times[1], minCs[line]],
            [times[2], minCs[line]],
            [times[3], minCs[line]],
            [times[4], minCs[line]],
            [times[5], minCs[line]],
            [times[6], 1.0],
        ]
        for line in range(num_lines)
    ]
    if target_lines is None:
        target_lines = set(range(num_lines)) - source_lines - detector_lines
    elif not target_lines.issubset(
        set(range(num_lines)) - source_lines - detector_lines
    ):
        raise ValueError(
            "Target lines must be a subset of available lines excluding source and detector lines."
        )
    for line in target_lines:
        anneal_schedules[line] = [
            [times[0], 0.0],
            [times[1], 0.0],
            [times[2], 0.0],
            [times[3], target_c],
            [times[4], target_c],
            [times[4] + min_time_steps[line], target_c],
            [times[5], target_c],
            [times[6], 1.0],
        ]

    for line in source_lines:
        anneal_schedules[line] = [
            [times[0], 0.0],
            [times[1], maxCs[line]],
            [times[2], maxCs[line]],
            [times[3], maxCs[line]],
            [times[4], maxCs[line]],
            [times[4] + min_time_steps[line], minCs[line]],
            [times[5], minCs[line]],
            [times[6], 1.0],
        ]
    for line in detector_lines:  # No delay relative to source.
        anneal_schedules[line] = [
            [times[0], 0.0],
            [times[1], minCs[line]],
            [times[2], minCs[line]],
            [times[3], minCs[line]],
            [times[4], minCs[line]],
            [times[4] + min_time_steps[line], maxCs[line]],
            [times[5], maxCs[line]],
            [times[6], 1.0],
        ]
    return anneal_schedules


def _make_polarizing_schedules(
    source_lines: Iterable[int],
    num_lines: int = 6,
    *,
    sign_polarization: int = 1,
    times: list[float] | tuple[float] = (0.0, 1.0, 2.0, 6.0),
):
    """Set polarizing schedules suitable for Larmour precision.

    See documentation for Larmour precession example, the same
    schedule is used."""
    if len(times) != 4:
        raise ValueError(
            "Expecting 2 unpolarized times, followed by two polarized times"
        )
    polarization_schedules = [[[t, 0] for t in times] for _ in range(num_lines)]
    for line in source_lines:
        polarization_schedules[line] = [
            [times[0], sign_polarization],
            [times[1], sign_polarization],
            [times[2], 0],
            [times[3], 0],
        ]
    return polarization_schedules


def _calc_anneal_offsets(
    frequencies: np.ndarray,
    psd: np.ndarray,
    target_A: float,
    dAdc: float,
    Amin: float | None = None,
    Amax: float | None = None,
):
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
        frequencies: frequencies at which power provided.
        psd: power spectral density, the absolute discrete fourier transform
            value squared at each frequency.
        target_A: expected/desired peak position.
        dAdc: Approximate rate of change of A with c (anneal offset).
        Amin: A lower bound on the frequency range used in estimation,
        Amax: An upper bound on the frequency range used in estimation,

    Returns:
        Estimated error on c relative to the mean value for a collection of

    """

    # NB a symmetric window only works for frequencies in the range,
    # and some bias is introduced by use of a target.
    if Amin is None:
        Amin = target_A / 2
    if Amax is None:
        Amax = target_A * 1.5

    Afilter = np.logical_and(frequencies < Amax, frequencies > Amin)
    mean_A_est = np.sum(
        psd[:, Afilter] * frequencies[Afilter][np.newaxis, :], axis=1
    ) / np.sum(psd[:, Afilter], axis=1)
    mu = np.mean(mean_A_est)
    print("Standard deviation in A estimates", np.sqrt(np.var(mean_A_est)))
    print()
    dcs = (mean_A_est - mu) / dAdc
    return dcs


def artificial_data(
    delays: np.ndarray,
    A: float,
    decay_time: float,
    num_independent_samples: int = float("Inf"),
    prng: np.random.Generator | int | None = None,
):
    """Create an artificial data set

    y(t) = np.exp(-delays / decay_time) * np.cos(2* np.pi * A * delays)
    with variance of (1 - y(t)^2) in the measured state. Given independent
    and identically distributed samples we can model noise as normally
    distributed.

    Args:
        delays: time of measurement
        A: frequency
        decay_time: exponential envelope time scale
        num_independent_samples: number of samples to model
        prng: pseudo random number generator or seed.
    Returns:
        A model signal:
    """
    y = np.exp(-delays / decay_time) * np.cos(2 * np.pi * A * delays)
    if num_independent_samples != float("Inf"):
        prng = np.random.default_rng(prng)
        return y + np.sqrt((1 - y**2) / num_independent_samples) * prng.normal(
            size=len(y)
        )
    else:
        return y


def run_parallel_experiment(
    sampler: ParallelEmbeddingComposite,
    bqm: dimod.BinaryQuadraticModel,
    qpu_parameters: dict,
    delays: np.ndarray | list,
    detector_lines: Iterable[int],
) -> np.ndarray:
    """Collect detector magnetization for a set of independent embeddings

    See documentation example, here we simply parallelize.

    Args:
        sampler: A parallel embedding composite sampler, wrapping the qpu sampler.
        bqm: Binary Quadratic Model
        qpu_parameters: parameters passed to the QPU sampler.
        delays: detector x_schedule_delays
        detector_lines: detector lines.

    Returns:
        A numpy array of detector magnetizations

    """
    mean_Z_detector = []
    for delay in tqdm(delays):
        for line in detector_lines:
            qpu_parameters["x_schedule_delays"][line] = delay
        # Return as a list of samplesets, instead of aggregated:
        samplesets, _ = sampler.sample_multiple(
            [bqm] * len(sampler.embeddings), **qpu_parameters
        )
        # Extract detector magnetization from each sampleset
        detector_samples = [
            dimod.keep_variables(sampleset, [("detector", 0)]).record.sample
            for sampleset in samplesets
        ]
        mean_Z_detector.append([np.mean(sample) for sample in detector_samples])
    return np.array(mean_Z_detector)


def plot_shim(
    mag_history: dict,
    flux_history: dict,
    num_experiments: int = 1,
    fname: str | None = None,
    label: str = "",
    max_qubit_labels: int = 10,
):
    """Plot the iterative flux_bias_shim process.

    Args:
        mag_history: the magnetizations estimated throughout the iterative
            process for every embedding.
        flux_history: the flux_biases assignments throughout the iterative
            process for every embedding.
        num_experiments: Number of programmings per flux iteration. Using 1
            by default it should be noted that 2 magnetizations may be
            be measured per step in flux_biases.
        fname: a filename to which to save data, mag or fb is prepended for
            the two plot types. By default no plots are created.
        label: a label for the plots, used in legends.
    """
    mag_array = np.array(list(mag_history.values()))
    flux_array = np.array(list(flux_history.values()))

    mag_array = np.reshape(
        mag_array,
        (mag_array.shape[0], mag_array.shape[1] // num_experiments, num_experiments),
    )

    plt.figure()
    plt.title(r"Magnetization by iteration, $\langle Z\rangle_{detector}$")
    for experiment_sign in range(num_experiments):
        if num_experiments > 1:
            plt.plot(
                mag_array[:, :, experiment_sign].transpose(),
                label=f"Initial state all {-1 + 2*experiment_sign}",
            )
        else:
            plt.plot(
                mag_array[:, :, experiment_sign].transpose(),
            )
    if num_experiments > 1:
        plt.plot(
            np.mean(mag_array, axis=2).transpose(),
            color="black",
            label="Experiment average",
        )
        plt.legend()
        plt.xlabel("Shim iteration")
    else:
        plt.xlabel("Programming")
        if mag_array.shape[0] <= max_qubit_labels:
            plt.legend(flux_history.keys(), title=f"{label} Qubit index")
    plt.ylabel("Magnetization")
    if fname is not None:
        plt.savefig(f"mag_{fname}")

    plt.figure()
    plt.title("All detector flux_biases")
    plt.plot(flux_array.transpose())
    plt.xlabel("Shim iteration")
    plt.ylabel("Flux bias ($\\Phi_0$)")
    if mag_array.shape[0] <= max_qubit_labels:
        plt.legend(flux_history.keys(), title=f"{label} Qubit index")
    if fname is not None:
        plt.savefig(f"fb_{fname}")


def main(
    cache_str: str | None = None,
    solver: dict | str | None = None,
    detector_lines: Iterable[int] = (0,),
    source_lines: Iterable[int] = (3,),
    target_c: float = 0.37,
    no_flux_biases: bool = False,
    no_anneal_offsets: bool = False,
    delay_min: float = 0.005,
    delay_max: float = 0.015,
    delay_min_fit: float | None = None,
    delay_max_fit: float | None = None,
    fn_schedule: str = "09-1317A-D_Advantage2_research1_4_annealing_schedule.xlsx",
    shim_detector_and_target: bool = False,
    target_lines: set | None = None,
):
    """Demonstrate t-d-s variability and mitigation strategies

    An ideal single-qubit target system might be prepared in
    a polarized state |1> whose evolution is subsequently
    described by H(c) = A(c) + B(c) h.
    Control limitations dictate that the A(s), B(s) and h realized
    by different qubits at a common c varies. An h error in the detector
    qubit can also contribute to errors in measurement.
    Methods are demonstrated for synchronization of frequency with
    use of anneal offsets incorporating a simple decoherence model, and
    shimming of a detector flux bias to restore symmetry.

    Higher accuracy shimming, and shimming of target flux_biases may also be
    desirable, but are beyond the scope of the example. Note that we can
    use simple statistic to determine flux_bias assignment on a detector
    relative to a target. E.g. a) when decoupled from the source and detector
    a 1 qubit model frequency omega=root(A(s)^2 + B(s)^2 h^2) is a convex
    monotonic function of the linear field, b) When decoupled from the source
    the response of the detector magnetization to a flux_bias perturbation is
    maximized.

    Args:
        cache_str:
            A unique experimental idenfier. If not None a
            cache/ is created which is populated with experimental data. The cache
            is checked for compatible experimental data before running an experiment,
            and if compatible data is present the data is reloaded rather than
            running new jobs through the client.
        solver:
            Name of the solver, or dictionary of characteristics.
        detector_lines:
            The integer indices of the detector lines.
        source_lines:
            The integer indices of the source lines.
        target_c:
            normalized control bias at which the target qubits are held
        no_flux_biases:
            When set to True, flux_biases are not modified. When False flux_biases
            are modified on detector qubits to achived zero expected magnetization at
            long delay.
        no_anneal_offsets:
            When set to True, anneal_offsets are not modified. When False anneal_offsets
            are modified so that the peak power-spectral density is peaked at a common
            value for all qubits. This peak values characterizes the frequency of the target
            qubit in simple well-calibratied models.
        delay_min: The delay on the detector line for which data is collected.
        delay_max: The maximum delay on the detector line for which data is collected. Between
            delay_min and delay_max the spacing in time reflects the target frequency that we
            are seeking to resolve for anneal_offset refinement.
        delay_min_fit:
            A lower bound on the timeseries window used for inference of the target power spectral density.
            A value that is too small can bias the estimator by introduction of effects related
            to coupling to the source line.
        delay_max_fit:
            An upper bound on the timeseries window used for inference of the target power spectral density.
            Too large a value reduces the efficiency of the estimator, since delays much larger than the
            T1 coherence time are dominated by noise.
        fn_schedule: A schedule file that is used to estimate an appropriate sampling interval for delay
            time and an appropriate scale for anneal_offset synchronization. This should be matched to the
            solver.
        target_lines: The integer indices of the target lines. By default all lines not reserved for the source and detector are available as target lines, but this can be used to specify a subset of target lines for analysis.
            By default all lines not reserved for the source and detector are available as target lines, but this can be used to specify a subset of target lines for analysis.

    Raises:
        ValueError: If the number of lines is less than 3, or
        if the detector_lines or source_lines are not
        disjoint and in set(range(num_lines))
    """
    print(
        "A variety of plots are shown to demonstrate heuristic correction of "
        "flux_biases on detectors, and target qubit frequency "
        "desynchronization, from small amounts of data. "
    )
    if delay_max_fit is None:
        delay_max_fit = delay_max  # Can be automated for SNR in principle.
    elif delay_max_fit > delay_max:
        raise ValueError("Fit window exceeds data window")
    if delay_min_fit is None:
        delay_min_fit = delay_min  # Can be automated for SNR in principle.
    elif delay_min_fit < delay_min:
        raise ValueError("Fit window exceeds data window")
    if delay_min_fit > delay_max_fit:
        raise ValueError("Fit window is empty")
    if len(detector_lines) == 0 or len(source_lines) == 0:
        raise ValueError("Must have at least one source and one detector line")
    detector_lines = set(detector_lines)
    source_lines = set(source_lines)

    if detector_lines & source_lines:
        raise ValueError("Detector and source lines must be disjoint")

    # Schedule based approximations, target_A and dA/dc are approximated.
    qpu_anneal_schedule = pd.read_excel(
        fn_schedule, sheet_name="Fast-Annealing Schedule"
    )
    plt.figure()
    plt.title("Schedule")
    delta_vs_s = qpu_anneal_schedule[::-1]
    plt.plot(delta_vs_s["s"], delta_vs_s["A(s) (GHz)"], label="A(s)")
    plt.plot(delta_vs_s["s"], delta_vs_s["B(s) (GHz)"], label="B(s)")
    target_A = np.interp(
        1 - target_c, 1 - delta_vs_s["s"], delta_vs_s["A(s) (GHz)"]
    )  # Expected frequency of detector magnetization oscillations
    target_B = np.interp(1 - target_c, 1 - delta_vs_s["s"], delta_vs_s["B(s) (GHz)"])
    print("Schedule predictions: ", "A(c)", target_A, "B(c)", target_B)
    print()
    dc = 0.01
    target_Aminus = np.interp(
        1 - (target_c - dc), 1 - delta_vs_s["s"], delta_vs_s["A(s) (GHz)"]
    )
    target_Aplus = np.interp(
        1 - (target_c + dc), 1 - delta_vs_s["s"], delta_vs_s["A(s) (GHz)"]
    )
    dAdc = (target_Aplus - target_Aminus) / (2 * dc)
    plt.plot(
        [target_c, target_c],
        [0, np.max(delta_vs_s["A(s) (GHz)"])],
        label=f"c={target_c}",
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

    qpu = DWaveSampler(solver=solver)
    zephyr_shape = qpu.properties["topology"]["shape"]
    exp_feature_info = get_properties(qpu)
    line_assignments = {
        n: al_idx for al_idx, al in enumerate(exp_feature_info) for n in al["qubits"]
    }
    num_lines = len(exp_feature_info)
    if target_lines is None:
        target_lines = set(range(num_lines)) - set(source_lines) - set(detector_lines)
    cmap = plt.colormaps.get_cmap("plasma")
    line_color = [cmap(i / (num_lines - 1)) for i in range(num_lines)]

    x_anneal_schedules = _make_anneal_schedules(
        exp_feature_info,
        source_lines=source_lines,
        detector_lines=detector_lines,
        target_lines=target_lines,
        target_c=target_c,
    )
    x_anneal_schedules_fb = _make_anneal_schedules(
        exp_feature_info,
        source_lines={},
        detector_lines=detector_lines,
        target_lines=target_lines,
        target_c=target_c,
    )
    x_polarizing_schedules = _make_polarizing_schedules(
        source_lines=source_lines, num_lines=num_lines
    )
    x_polarizing_schedules_fb = _make_polarizing_schedules(
        source_lines={}, num_lines=num_lines
    )
    x_schedule_delays = [0.0] * num_lines

    anneal_offsets = [0.0] * qpu.properties["num_qubits"]
    flux_biases = [0.0] * qpu.properties["num_qubits"]

    # See documented Larmour precession example
    qpu_parameters = dict(
        num_reads=500,
        answer_mode="raw",
        x_disable_filtering=True,
        x_schedule_delays=x_schedule_delays,
        x_anneal_schedules=x_anneal_schedules,
        x_polarizing_schedules=x_polarizing_schedules,
        flux_biases=flux_biases,
        anneal_offsets=anneal_offsets,
    )

    print(
        "Determine many T-D-S embeddings appropriate for parallel programming (see mca_embedding.py example)."
    )
    print()
    T = qpu.to_networkx_graph()

    def _target_assignments(n: int):
        line = line_assignments[n]
        if line in detector_lines:
            return "detector"
        elif line in source_lines:
            return "source"
        elif line in target_lines:
            return "target"
        else:
            return "unused"

    Tnode_to_tds = {n: _target_assignments(n) for n in qpu.nodelist}
    target_graph = nx.Graph()
    target_graph.add_node(0)
    S, Snode_to_tds = make_tds_graph(target_graph)

    subgraph_kwargs = dict(node_labels=(Snode_to_tds, Tnode_to_tds), as_embedding=True)
    fn_cache = f"cache/emb_{cache_str}.pkl"
    if cache_str:
        os.makedirs(os.path.dirname(fn_cache), exist_ok=True)
    if cache_str and os.path.isfile(fn_cache):
        with open(fn_cache, "rb") as f:
            embs = pickle.load(f)
    else:
        embs = find_multiple_embeddings(
            S,
            T,
            max_num_emb=None,
            embedder_kwargs=subgraph_kwargs,
            one_to_iterable=True,
        )
        with open(fn_cache, "wb") as f:
            pickle.dump(embs, f)

    # Reorder by target line for ease of analysis:
    embs_by_line = {i: [] for i in range(num_lines)}
    for i, emb in enumerate(embs):
        q = emb[0][0]
        embs_by_line[qubit_to_Advantage2_annealing_line(q, zephyr_shape)].append(emb)
    embs = [emb for i in range(num_lines) for emb in embs_by_line[i]]

    sampler = ParallelEmbeddingComposite(qpu, embeddings=embs)

    dt = 1 / target_A / 1000 / 4  # Appropriate scale for frequency resolution.
    delays = np.linspace(delay_min, delay_max, round((delay_max - delay_min) / dt) + 1)

    # Demonstrate some data for simple model y(t) = cos(2 pi A [t + t0]) exp(- [t + t0]/d):
    delays_ns = 5 * np.random.random() + 1000 * delays
    ld = len(delays_ns)
    frequencies = np.arange(ld) / dt / 1000 / ld
    decay_time_ns = 20
    for idx, A in enumerate([target_Aminus, target_A, target_Aplus]):
        for num_independent_samples in [100, float("Inf")]:
            signal = artificial_data(
                delays_ns,
                A,
                decay_time=decay_time_ns,
                num_independent_samples=num_independent_samples,
            )
            if num_independent_samples == float("Inf") and idx == 1:
                label = f"A={A:.3g}, no sample err."
            elif num_independent_samples == 100:
                label = f"A={A:.3g}"
            else:
                continue

            plt.figure("y=cos(2pi A t)exp(-t/T)+sampling error")
            plt.title("y=cos(2pi A t)exp(-t/T)+sampling error")
            plt.plot(delays_ns, signal, label=label)
            plt.xlabel("Time, microseconds")
            plt.ylabel(r"Magnetization, $\langle Z \rangle_{detector}$")
            plt.legend()

            plt.figure("Simple model data (PSD) ~ A/((f-A)^2 + A^2)")
            plt.title("Approx Lorentzian power spectral density ~ A/((f-A)^2 + A^2)")
            psd = np.abs(np.fft.fft(signal)) ** 2 / len(signal)
            plt.plot(frequencies[: ld // 2], psd[: ld // 2], label=label)
            plt.ylabel(rf"Power Spectral Density, $|\langle Z\rangle(\omega)|^2$")
            plt.xlabel(r"Frequency ($\omega$), GHz")
            plt.legend()

    bqm = dimod.BinaryQuadraticModel("SPIN").from_ising(
        {n: 0 for n in S.nodes()}, {e: -1 for e in S.edges()}
    )

    if not no_flux_biases:
        shimstr = "_FBshim"
        print(
            "Shim flux biases for zero detector magnetization in"
            " the limit of long delay."
        )
        print()
        fn_cache = f"cache/FB_{cache_str}.npy"
        if cache_str and os.path.isfile(fn_cache):
            with open(fn_cache, "rb") as f:
                flux_biases, flux_history, mag_history = pickle.load(f)
        else:
            # Require zero magnetization in the limit of long delay (where
            # source impact has decayed away.
            bqm_embedded = dimod.BinaryQuadraticModel("SPIN").from_ising(
                {emb[n][0]: h for emb in embs for n, h in bqm.linear.items()},
                {
                    tuple(emb[n][0] for n in e): J
                    for emb in embs
                    for e, J in bqm.quadratic.items()
                },
            )
            shimmed_variables = {
                n
                for n in bqm_embedded.variables
                if qubit_to_Advantage2_annealing_line(n, zephyr_shape) in detector_lines
            }

            for line in detector_lines:
                qpu_parameters["x_schedule_delays"][line] = 0.1  # Documented limit.
            bqm_td = bqm_embedded.copy()
            bqm_td.fix_variables(
                {
                    n: 0
                    for n in bqm_embedded.variables
                    if line_assignments[n] in source_lines
                }
            )
            qpu_parameters_fb = (
                qpu_parameters.copy()
            )  # NB, values replaced not modified
            qpu_parameters_fb["x_anneal_schedules"] = x_anneal_schedules_fb
            qpu_parameters_fb["x_polarizing_schedules"] = x_polarizing_schedules_fb

            if shim_detector_and_target:
                shimmed_variables = {
                    n
                    for n in bqm_embedded.variables
                    if qubit_to_Advantage2_annealing_line(n, zephyr_shape)
                    in target_lines
                    or n in detector_lines
                }
                flux_biases, flux_history, mag_history = shim_tds_flux_biases(
                    bqm=bqm_td,  # bqm_embedded,  #
                    sampler=qpu,
                    sampling_params=qpu_parameters_fb,
                    target_lines=target_lines,
                    detector_lines=detector_lines,
                    line_assignments=line_assignments,
                    num_steps=60,
                    shimmed_variables=shimmed_variables,
                )
            else:
                flux_biases, flux_history, mag_history = shim_flux_biases(
                    bqm=bqm_td,  # bqm_embedded,
                    sampler=qpu,
                    sampling_params=qpu_parameters,
                    shimmed_variables=shimmed_variables,
                )

            if cache_str:
                with open(fn_cache, "wb") as f:
                    pickle.dump((flux_biases, flux_history, mag_history), f)
        plot_shim(mag_history, flux_history)
        qpu_parameters["flux_biases"] = flux_biases
    else:
        shimstr = ""
    plt.show()
    print(f"Collect data for {len(embs)} parallel embeddings")
    fn_cache = f"cache/AO_It0_{cache_str}.npy"
    if cache_str and os.path.isfile(fn_cache):
        mean_Z_detector = np.load(fn_cache)
    else:
        mean_Z_detector = run_parallel_experiment(
            sampler, bqm, qpu_parameters, delays, detector_lines
        )
        if cache_str:
            os.makedirs(os.path.dirname(fn_cache), exist_ok=True)
            np.save(fn_cache, mean_Z_detector)

    first = np.argmax(delays >= delay_min_fit)
    last = np.argmax(delays >= delay_max_fit) + 1
    ld = last - first
    if ld < 1:
        raise ValueError("Fit window is empty: t-fit range too small for target_A")

    frequencies = np.arange(ld) / dt / 1000 / ld
    psd = np.array(
        [
            np.abs(np.fft.fft(mean_Z_detector[first:last, i])) ** 2
            for i in range(len(embs))
        ]
    ) / (last - first)

    print(
        "Plot real space data in 3 formats, and the power spectral density estimated by a discrete Fourier transform"
    )

    plt.figure()
    plt.title("Time series for several qubits using distinct target lines")
    plotted_lines = set()
    for idx, emb in enumerate(embs):
        q = emb[0][0]
        line_target = qubit_to_Advantage2_annealing_line(q, zephyr_shape)
        if line_target not in plotted_lines:
            plt.plot(
                delays * 1000,
                mean_Z_detector[:, idx],
                color=line_color[line_target],
                label=f"target line {line_target}",
            )
            plotted_lines.add(line_target)
    plt.ylabel("Detector magnetizations")
    plt.xlabel("Detector delay, ns")
    plt.legend()
    plt.grid()

    plt.figure()
    plt.title("Real space magnetizations (divergent color scheme)")
    plt.imshow(mean_Z_detector, vmin=-1, vmax=1, cmap="RdBu")
    yticks_dict = {
        first: f"{1000 * delays[first]:.3g}",
        last - 1: f"{1000 * delays[last-1]:.3g}",
    }
    yticks_dict.update(
        {0: str(1000 * delays[0]), mean_Z_detector.shape[0] - 1: str(1000 * delays[-1])}
    )
    plt.yticks(
        list(yticks_dict.keys()),
        list(yticks_dict.values()),
    )
    plt.xlabel("Target-Detector-Source embedding")
    plt.ylabel("Delay, nanoseconds")

    plt.figure()
    plt.title("Real space magnetizations (higher contrast color scheme)")
    plt.imshow(mean_Z_detector[first:last, :])
    yticks_dictN = {
        0: f"{1000 * delays[first]:.3g}",
        last - first - 1: f"{1000 * delays[last-1]:.3g}",
    }
    plt.yticks(
        list(yticks_dictN.keys()),
        list(yticks_dictN.values()),
    )
    plt.xlabel("Target-Detector-Source embedding")
    plt.ylabel("Delay, nanoseconds")

    plt.figure()
    plt.title("Power associated to magnetization time series")
    lines_represented = set()
    for i, emb in enumerate(embs):
        q = emb[0][0]
        line = qubit_to_Advantage2_annealing_line(q, zephyr_shape)
        if line in lines_represented:
            label = None
        else:
            label = f"target-qubit line={line}"
            lines_represented.add(line)
        plt.plot(
            frequencies[: ld // 2],
            psd[i, : ld // 2],
            color=line_color[line],
            label=label,
        )
    plt.plot(
        [target_A, target_A],
        [0, np.max(psd)],
        color="black",
        linestyle="dashed",
        label="Schedule prediction",
    )
    plt.legend()
    plt.ylabel(rf"Power Spectral Density, $|\langle Z\rangle(\omega)|^2$")
    plt.xlabel(r"Frequency ($\omega$), GHz")
    plt.grid(True)

    # Calculate anneal_offsets for synchronization
    if not no_anneal_offsets:
        anneal_offsets = _calc_anneal_offsets(
            frequencies, psd, target_A, dAdc
        )  # Per embedding

        print("Collect data with anneal offset compensation of frequency variation")
        fn_cache = fn_cache = f"cache/AO_It1_{cache_str}.npy"
        if cache_str and os.path.isfile(fn_cache):
            mean_Z_detector = np.load(fn_cache)
        else:
            for emb, ao in zip(embs, anneal_offsets):
                qpu_parameters["anneal_offsets"][
                    emb[0][0]
                ] -= ao  # Apply correction to target on each embedding
            mean_Z_detector = run_parallel_experiment(
                sampler, bqm, qpu_parameters, delays, detector_lines
            )
            if cache_str:
                np.save(fn_cache, mean_Z_detector)
        psd = np.array(
            [
                np.abs(np.fft.fft(mean_Z_detector[first:last, i])) ** 2
                for i in range(len(embs))
            ]
        ) / (last - first)

        print(
            "Plot real space data in 3 formats, and the power spectral density estimated by a discrete Fourier transform"
        )

        plt.figure()
        plt.title("Time series after anneal_offsets")
        plotted_lines = set()
        for i, emb in enumerate(embs):
            q = emb[0][0]
            line = qubit_to_Advantage2_annealing_line(q, zephyr_shape)
            if line not in plotted_lines:
                plt.plot(
                    delays * 1000,
                    mean_Z_detector[:, i],
                    color=line_color[line],
                    label=f"target line {line}",
                )
                plotted_lines.add(line)
        plt.ylabel("Detector magnetizations")
        plt.xlabel("Detector delay, ns")
        plt.legend()
        plt.grid()

        plt.figure()
        plt.title("Real space magnetizations after anneal offsets")
        plt.imshow(mean_Z_detector, vmin=-1, vmax=1, cmap="RdBu")
        plt.yticks(
            list(yticks_dict.keys()),
            list(yticks_dict.values()),
        )
        plt.xlabel("Target-Detector-Source embedding")
        plt.ylabel("Delay, nanoseconds")

        plt.figure()
        plt.title("Real space magnetizations after anneal offsets")
        plt.imshow(mean_Z_detector[first:last, :])
        plt.yticks(
            list(yticks_dictN.keys()),
            list(yticks_dictN.values()),
        )
        plt.xlabel("Target-Detector-Source embedding")
        plt.ylabel("Delay, nanoseconds")

        plt.figure()
        plt.title("Power associated to magnetization time series after anneal offsets")
        lines_represented = set()
        for i, emb in enumerate(embs):
            q = emb[0][0]
            line = qubit_to_Advantage2_annealing_line(q, zephyr_shape)
            if line in lines_represented:
                label = None
            else:
                label = f"target-qubit line={line_target}"
                lines_represented.add(line_target)
            plt.plot(
                frequencies[: ld // 2],
                psd[i, : ld // 2],
                color=line_color[line_target],
                label=label,
            )
        plt.plot(
            [target_A, target_A],
            [0, np.max(psd)],
            color="black",
            linestyle="dashed",
            label="Schedule prediction",
        )
        plt.legend()
        plt.ylabel(rf"Power Spectral Density, $|\langle Z\rangle(\omega)|^2$")
        plt.xlabel(r"Frequency ($\omega$), GHz")
        plt.grid(True)

        plt.figure()
        anneal_offsets0 = anneal_offsets
        anneal_offsets = _calc_anneal_offsets(
            frequencies, psd, target_A, dAdc
        )  # Per embedding
        lines_represented = set()
        for i, emb in enumerate(embs):
            q = emb[0][0]
            line_target = qubit_to_Advantage2_annealing_line(q, zephyr_shape)
            if line_target in lines_represented:
                label = None
            else:
                label = f"target-qubit line={line_target}"
                lines_represented.add(line_target)
            plt.plot(
                anneal_offsets0[i],
                anneal_offsets[i],
                color=line_color[line_target],
                marker="x",
                label=label,
            )
        plt.xlabel(
            "Frequency discrepancy (proposed c-<c> change) before anneal_offset shim"
        )
        plt.ylabel(
            "Frequency discrepancy (proposed c-<c> change) after anneal_offset shim"
        )
        plt.grid(True)
        plt.legend()
    plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="A target-detector-source embedding example"
    )
    parser.add_argument(
        "--use_cache",
        action="store_true",
        help="Add this flag to save experimental data, and reload when available at matched parameters",
    )
    parser.add_argument(
        "--solver_name",
        type=str,
        help="Option to specify QPU solver, by default an experimental system supporting fast reverse anneal",
        default=SOLVER_FILTER,
    )
    parser.add_argument(
        "--line_detector",
        type=int,
        help="Detector line",
        default=0,  # First vertical qubit line
    )
    parser.add_argument(
        "--line_source",
        type=int,
        help="Source line",
        default=3,  # First horizontal qubit line under 6-line control
    )
    parser.add_argument(
        "--fn_schedule",
        type=str,
        help="fn_schedule labels an Excel QPU schedule file that should be in the path. The schedules for specific processors can be found on the website. The schedule is used to parameterize an approximate frequency target, and the susceptibility of that frequency to a change in the anneal offset.",
        default=None,
    )
    parser.add_argument(
        "--target_c",
        type=float,
        help="The normalized phi_cjj value for the target. This should correspond to a frequency A(c_target) of approximately 1 to 3GHz for reasonable performance. The fn_schedule file can be used to infer an approximate relationship between A(c_target) and c_target.",
        default=0.37,
    )
    parser.add_argument(
        "--delay_min",
        type=float,
        help="Initial delay time (us) for data collection. Ideally this matches to the earliest delay for which the signal is depolarized.",
        default=0.005,
    )
    parser.add_argument(
        "--delay_max",
        type=float,
        help="Final delay time (us) for data collection.",
        default=0.015,
    )
    parser.add_argument(
        "--delay_min_fit",
        type=float,
        help="Initial delay time (us) for frequency estimation, by default matches delay_min. This is ideally chosen to match the smallest delay for which the signal is not polarized.",
        default=None,
    )
    parser.add_argument(
        "--delay_max_fit",
        type=float,
        help="Final delay time (us) for frequency estimation, by default matches delay_max. This is ideally chosen such that the number of oscillations captured (delay_max-delay_min)*A(target_c) is large, yet small enough that data collection is efficient and signal is not dominated by the decohered portion of the evolution at large delay.",
        default=None,
    )
    parser.add_argument(
        "--line_target",
        type=int,
        help="Target line: by default all lines not reserved for the source and detector are available. Choosing a specific line can allow improved parameterization and accuracy (e.g. customization of detector delay relative to a specific target line, rather than an average).",
        default=None,  #
    )
    parser.add_argument(
        "--no_flux_biases",
        action="store_true",
        help="Add this flag to omit the data analsis with recalibrated flux_biases",
    )
    parser.add_argument(
        "--no_anneal_offsets",
        action="store_true",
        help="Add this flag to omit the data analsis with recalibrated anneal_offsets",
    )
    parser.add_argument(
        "--shim_detector_and_target",
        action="store_true",
        help="Add this flag to shim both the detector and target lines (experimental implementation, not yet producing good outcomes)",
    )

    args = parser.parse_args()
    print(args.fn_schedule)
    if args.use_cache:
        cache_str = _get_experiment_id(args)
    else:
        cache_str = None
    if args.line_target is None and args.shim_detector_and_target:
        raise ValueError('The alternating method used is very sensitive to '
                         'relative line delays, therefore it is used with a '
                         'restriction to 3 lines at a time')
    main(
        cache_str=cache_str,
        solver=args.solver_name,
        detector_lines=(args.line_detector,),
        source_lines=(args.line_source,),
        target_c=args.target_c,
        delay_min=args.delay_min,
        delay_max=args.delay_max,
        delay_min_fit=args.delay_min_fit,
        delay_max_fit=args.delay_max_fit,
        no_anneal_offsets=args.no_anneal_offsets,
        no_flux_biases=args.no_flux_biases,
        target_lines={args.line_target} if args.line_target is not None else None,
        shim_detector_and_target=args.shim_detector_and_target,
        fn_schedule=args.fn_schedule,
    )
