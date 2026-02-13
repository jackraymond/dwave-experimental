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

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from dwave.system import DWaveSampler
from dwave.system.testing import MockDWaveSampler
from minorminer.subgraph import find_subgraph
from minorminer.utils.parallel_embeddings import find_multiple_embeddings
from dwave.experimental.multicolor_anneal import (
    get_properties,
    SOLVER_FILTER,
    qubit_to_Advantage2_annealing_line,
    make_tds_graph,
)
from dwave_networkx import (
    draw_zephyr,
    draw_parallel_embeddings,
    draw_chimera_embedding,
    chimera_graph,
)


def main(
    use_client: bool = False,
    solver: dict | str | None = None,
    detector_line: int = 0,
    source_line: int = 3,
    biclique_target_lines: set | None = None,
):
    """Examples of multicolor annealing.

    The first example demonstrates the embedding of a single
    qubit problem in many places on the processor, such that
    every qubit is connected to a source and detector qubit on the
    specified lines.
    The second example embeds a loop of 64 target qubits with each target coupled
    to a source and detector qubit.
    The third example shows embedding of a chimera like graph where every
    target node has a coupling available to a source and target line. Sources
    might be shared in principle, but there are insufficient detectors to
    measure all target qubits simultaneously (a subset must be chosen per
    experiment).

    In the fourth example we embed a biclique on two lines, with the remaining
    lines used as detectors. There are no sources.

    Args:
        use_client: Whether or not to use a specific solver. If
            True, the solver field is used to establish a client
            connection, and embedding proceeds with respect to the
            associated solver graph. If False the code runs locally
            using a defect-free Advantage2 processor graph at a
            Zephyr[6] scale, using the standard 6-anneal line scheme.
        solver: Name of the solver, or dictionary of characteristics.
        detector_line: The integer index of the detector line.
        source_line: The integer index of the source line.
        biclique_target_lines: a pair of lines used for embedding a
            a biclique. The remaining lines are used as detectors.
            Subject to device yield limitations, a biclique of size
            up to K_{8,8} is possible. {2,3} by default.
    Raises:
        ValueError: If the number of lines is less than 3, or
            if ``detector_line`` or ``source_line`` is not in
            range [0, ``num_lines``).
    """

    # when available, use feature-based search to default the solver.
    if use_client:
        qpu = DWaveSampler(solver=solver)
        annealing_lines = get_properties(qpu)
        line_assignments = {
            n: al_idx for al_idx, al in enumerate(annealing_lines) for n in al["qubits"]
        }
        num_lines = len(annealing_lines)
    else:
        shape = [6, 4]
        qpu = MockDWaveSampler(topology_type="zephyr", topology_shape=[6, 4])
        line_assignments = {
            n: qubit_to_Advantage2_annealing_line(n, shape) for n in qpu.nodelist
        }
        num_lines = 6
    if biclique_target_lines is None:
        biclique_target_lines = {0, 3}
    # Plot the colored graph:
    T = qpu.to_networkx_graph()

    cmap = plt.colormaps.get_cmap("plasma")
    colors = [cmap(i / (num_lines - 1)) for i in range(num_lines)]

    def T_line_assignments(n: int):
        line = line_assignments[n]
        if line == detector_line:
            return "detector"
        elif line == source_line:
            return "source"
        else:
            return "target"

    Tnode_to_tds = {n: T_line_assignments(n) for n in qpu.nodelist}
    node_color = [colors[line_assignments[n]] for n in T.nodes()]
    plt.figure("Annealing lines")
    draw_zephyr(T, node_color=node_color, node_size=5, edge_color="grey")

    print(
        "Embed many single-target variable problems, with a source and detector"
        " associated to every qubit."
    )
    target_graph = nx.Graph()
    target_graph.add_node(0)
    S, Snode_to_tds = make_tds_graph(target_graph)
    subgraph_kwargs = dict(node_labels=(Snode_to_tds, Tnode_to_tds), as_embedding=True)

    embs = find_multiple_embeddings(
        S, T, max_num_emb=None, embedder_kwargs=subgraph_kwargs, one_to_iterable=True
    )
    used_nodes = {v[0] for emb in embs for v in emb.values()}
    node_color = [
        colors[line_assignments[n]] if n in used_nodes else "grey" for n in T.nodes()
    ]
    plt.figure("Parallel Embedding of many decoupled D-T-S models")
    draw_parallel_embeddings(T, embeddings=embs, S=S, node_color=node_color)

    print(
        "Embed a loop of length 64, with a source and detector "
        "associated to every qubit Embedding for a ring of length L."
    )
    L = 64
    target_graph = nx.from_edgelist((i, (i + 1) % L) for i in range(L))
    S, Snode_to_tds = make_tds_graph(target_graph)

    plt.figure("A loop decorated with sources and detectors")
    colors_S = {
        "target": "k",
        "source": colors[source_line],
        "detector": colors[detector_line],
    }

    def loop_pos(n, n_range):
        if isinstance(n, tuple):
            if n[0] == "detector":
                mult = 1.2
            else:
                mult = 0.8
            return (
                mult * np.cos(2 * np.pi * n[1] / n_range),
                mult * np.sin(2 * np.pi * n[1] / n_range),
            )

        else:
            return (np.cos(2 * np.pi * n / n_range), np.sin(2 * np.pi * n / n_range))

    pos = {n: loop_pos(n, L) for n in S.nodes()}
    node_color = [colors_S[Snode_to_tds[n]] for n in S.nodes()]
    nx.draw_networkx(
        S, node_color=node_color, pos=pos, with_labels=False, node_size=640 / L
    )

    subgraph_kwargs = dict(node_labels=(Snode_to_tds, Tnode_to_tds), as_embedding=True)
    emb = find_subgraph(S, T, **subgraph_kwargs)
    used_nodes = {v[0] for v in emb.values()}
    node_color = [
        colors[line_assignments[n]] if n in used_nodes else "grey" for n in T.nodes()
    ]
    plt.figure("The embedded loop with detectors and sources")
    draw_parallel_embeddings(T, embeddings=[emb], S=S, node_color=node_color)

    print(
        "Identify nodes that are each attached to at least one source and one"
        " detector. These might be shared amongst target nodes of a complex"
        " model (in this case a Chimera graph)."
    )

    def has_source_and_detector(T, n):
        return any(
            line_assignments[nn] == detector_line for nn in T.neighbors(n)
        ) and any(line_assignments[nn] == source_line for nn in T.neighbors(n))

    Tsub = T.subgraph(
        {n for n in T.nodes() if has_source_and_detector(T, n)}
    ).copy()  # Keep only nodes connected to both a source and detector
    emb = {n: (n,) for n in Tsub.nodes()}
    plt.figure("Chimera: a complex graph allowing reconfigurable detectors and sources")
    Tsub.add_nodes_from(T.nodes())
    node_color = [
        (
            colors[line_assignments[n]]
            if line_assignments[n] == detector_line
            or line_assignments[n] == source_line
            else "grey"
        )
        for n in T.nodes()
    ]
    draw_parallel_embeddings(T, embeddings=[emb], S=Tsub, node_color=node_color)

    # BEGIN ADDITION
    print("What follows, how to do an embedded model (a square or cubic lattice)")
    # Open boundary square lattice on Chimera. NB, L<=4 starts to look irregular.

    for L in [6, 7, 8]:
        embedding = {
            (x, y): ((x, y, 0, 0), (x, y, 1, 0)) for x in range(L) for y in range(L)
        }
        subgraph_target = chimera_graph(
            L, L, 4, coordinates=True
        )  # Reduced scale for clearer visualization
        plt.figure(f"Square {L}x{L}")
        draw_chimera_embedding(subgraph_target, embedding, node_size=10 / L)
        # Try embedding with find_subgraph and some time-out, hinting vertical -> vertical and horizontal -> horizontal can help.

        def displace2d(n: tuple, orientation: int):
            "Displace coordinate by row n[0] or column n[1]"
            if orientation == 0:
                return (n[0] + 1,) + n[1:]
            else:
                return (n[0],) + (n[1] + 1,) + n[2:]

        edgelist_logical = [
            (dimer[orientation], displace2d(dimer[orientation], orientation))
            for dimer in embedding.values()
            for orientation in [0, 1]
        ]
        # Delete (or wrap) boundary spanning:
        edgelist_square = [
            (n1, n2) for (n1, n2) in edgelist_logical if n2[0] < L and n2[1] < L
        ]
        S_t = nx.from_edgelist(edgelist_square + list(embedding.values()))
        emb_to_four_lines = find_subgraph(
            S_t, Tsub, as_embedding=True, timeout=20
        )  # Could try larger in a hypothetical hard case.
        if len(emb) > 0:
            plt.figure(f"Square {L}x{L} embedded: no specific source/target assignment")
            draw_parallel_embeddings(
                T, embeddings=[emb_to_four_lines], S=S_t, node_color=node_color
            )
        else:
            print("Square lattice target embedding not found (by this scheme)")
        # This assumes the 6 line scheme, first 3 lines are for vertical qubits, final 3 lines for horizontal qubits.
        S_tds_greedy = S_t.copy()  # shallow copy is fine
        assert detector_line < 3 and source_line > 2
        S_tds, Snode_to_tds = make_tds_graph(
            target_graph=S_t,
            detected_nodes=[
                n for n in S_t.nodes() if n[2] == 1
            ],  # Horizontal qubits, connect to vertical detector line
            sourced_nodes=[
                n for n in S_t.nodes() if n[2] == 0
            ],  # Vertical qubits, connect to horizontal source line
        )
        Snode_to_tds = {
            k: v if v != "target" else f"target{k[2]}" for k, v in Snode_to_tds.items()
        }  # Extra placement hinting is important for reducing time.
        from dwave_networkx import zephyr_coordinates

        linear_to_zephyr = zephyr_coordinates(
            *qpu.properties["topology"]["shape"]
        ).linear_to_zephyr
        Tnode_to_tds = {
            k: v if v != "target" else f"target{linear_to_zephyr(k)[0]}"
            for k, v in Tnode_to_tds.items()
        }
        emb_to_six_lines = find_subgraph(
            S_tds,
            T,
            node_labels=(Snode_to_tds, Tnode_to_tds),
            as_embedding=True,
            timeout=10,
        )
        if len(emb_to_six_lines) > 0:
            plt.figure(f"Square {L}x{L} embedded with source/detector per 2-chain")
            draw_parallel_embeddings(
                T, embeddings=[emb_to_six_lines], S=S_tds, node_color=node_color
            )
        else:
            print(f"{L}x{L} lattice with S/D per chain not found by this scheme")
    plt.show()
    # Open (or periodic in z) cubic lattices on Chimera.
    Lz = 3  # We stack up to 4-fold, could be less, cant be more (with 2-variable chains on Chimera):
    for L in range(3, 5):
        embedding = {
            (x, y, z): ((x, y, 0, z), (x, y, 1, z))
            for x in range(L)
            for y in range(L)
            for z in range(Lz)
        }
        # I've fixed the 3rd dimension to 2, can't be bigger than 2 with dimers
        # Using 4-chains allows up to 3m/4 x 3m/4 x 8 on a defect free Zephyr[m]
        target = chimera_graph(
            L, L, 4, coordinates=True
        )  # Reduced scale for visualization
        plt.figure(f"Cubic {L}x{L}x{Lz}")
        draw_chimera_embedding(target, embedding, node_size=10 / L)

    ## END OF ADDITION
    m = 8
    print(
        "Embed a K_{m,m} on two lines, with detectors on the other four lines."
        f" m={m} for this example."
    )
    target_graph = nx.from_edgelist([(i, j) for i in range(m) for j in range(m, 2 * m)])
    S, Snode_to_td = make_tds_graph(target_graph, sourced_nodes=[])  # No source nodes.

    Tnode_to_td = {
        n: "target" if line_assignments[n] in biclique_target_lines else "detector"
        for n in qpu.nodelist
    }

    subgraph_kwargs = dict(node_labels=(Snode_to_td, Tnode_to_td), as_embedding=True)
    emb = find_subgraph(S, T, **subgraph_kwargs)
    if emb:
        pos = {n: loop_pos(n, 2 * m) for n in S.nodes()}
        plt.figure("A fully detected clique, with no sources")
        nx.draw_networkx(S, pos)

        used_nodes = {v[0] for v in emb.values()}
        node_color = [
            colors[line_assignments[n]] if n in used_nodes else "grey"
            for n in T.nodes()
        ]
        plt.figure("A clique on two lines, detected by qubits on the remaining 4 lines")
        draw_parallel_embeddings(T, embeddings=[emb], S=S, node_color=node_color)
    else:
        print(
            "For the choice of biclique target lines and yield pattern "
            f"a K {m} {m} embedding was not found."
        )

    plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="A target-detector-source embedding example"
    )
    parser.add_argument(
        "--use_client", action="store_true", help="Add this flag to use the client"
    )
    parser.add_argument(
        "--solver_name",
        type=str,
        help="Option to specify QPU solver when use_client is True, by default an experimental system supporting fast reverse anneal",
        default=SOLVER_FILTER,
    )
    args = parser.parse_args()

    main(
        use_client=args.use_client,
        solver=args.solver_name,
    )
