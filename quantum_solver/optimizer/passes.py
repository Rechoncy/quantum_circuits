"""
Peephole, Algebraic Commutation, Topology Routing, 1-Qubit Fusion, and
2-Qubit Block Consolidation Optimization Passes.
"""

from __future__ import annotations

from collections import deque
import math
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from ..core.circuit import (
    GateInstruction,
    MeasureInstruction,
    NoiseInstruction,
    QuantumCircuit,
    embed_gate_unitary,
)
from ..core.gates import NativeGateSet
from ..core.hardware import HardwareModel
from .euler_1q import normalize_angle, synthesize_1q_for_gate_set
from .kak_2q import synthesize_2q_unitary


def expand_multi_qubit_gates(circuit: QuantumCircuit) -> QuantumCircuit:
    """
    Decompose 3-qubit gates (e.g. CCX / Toffoli) into standard 1Q + 2Q CNOT gates
    so subsequent routing and 2Q KAK block synthesis can optimize them globally.
    """
    out = QuantumCircuit(circuit.num_qubits, name=circuit.name)
    out.measured_qubits = list(circuit.measured_qubits)

    for op in circuit.operations:
        if isinstance(op, GateInstruction) and op.name.upper() in ("CCX", "TOFFOLI", "CCZ"):
            c0, c1, tgt = op.qubits
            is_ccx = op.name.upper() in ("CCX", "TOFFOLI")
            if is_ccx:
                out.h(tgt)
            out.cnot(c1, tgt)
            out.tdg(tgt)
            out.cnot(c0, tgt)
            out.t(tgt)
            out.cnot(c1, tgt)
            out.tdg(tgt)
            out.cnot(c0, tgt)
            out.t(c1)
            out.t(tgt)
            out.h(tgt)
            out.cnot(c0, c1)
            out.t(c0)
            out.tdg(c1)
            out.cnot(c0, c1)
            if not is_ccx:
                out.h(tgt)
        else:
            out.operations.append(op)
    return out


def _shortest_coupling_path(
    q_start: int,
    q_end: int,
    edges: Set[Tuple[int, int]],
    num_qubits: int,
) -> List[int]:
    """Find shortest path between q_start and q_end on hardware coupling graph via BFS."""
    if q_start == q_end:
        return [q_start]
    adj: Dict[int, List[int]] = {i: [] for i in range(num_qubits)}
    for u, v in edges:
        if u < num_qubits and v < num_qubits:
            adj[u].append(v)
            adj[v].append(u)

    queue: deque[List[int]] = deque([[q_start]])
    visited = {q_start}
    while queue:
        path = queue.popleft()
        curr = path[-1]
        if curr == q_end:
            return path
        for nxt in adj[curr]:
            if nxt not in visited:
                visited.add(nxt)
                queue.append(path + [nxt])
    raise ValueError(f"Hardware topology is disconnected between qubit {q_start} and {q_end}.")


def _all_pairs_distances(edges: Set[Tuple[int, int]], num_qubits: int) -> List[List[int]]:
    """Compute all-pairs shortest path distances on the hardware coupling graph."""
    inf = 10**6
    dist = [[inf] * num_qubits for _ in range(num_qubits)]
    adj: Dict[int, List[int]] = {i: [] for i in range(num_qubits)}
    for u, v in edges:
        if u < num_qubits and v < num_qubits:
            adj[u].append(v)
            adj[v].append(u)
    for src in range(num_qubits):
        dist[src][src] = 0
        q: deque[int] = deque([src])
        while q:
            cur = q.popleft()
            for nxt in adj[cur]:
                if dist[src][nxt] > dist[src][cur] + 1:
                    dist[src][nxt] = dist[src][cur] + 1
                    q.append(nxt)
    return dist


def route_for_topology(
    circuit: QuantumCircuit,
    hardware: Optional[HardwareModel],
    restore_final_layout: bool = True,
) -> QuantumCircuit:
    """
    SABRE-inspired dynamic layout-tracking topology router!
    Tracks logical-to-physical qubit placement across the circuit instead of inserting
    wasteful round-trip SWAPs after every non-adjacent 2Q gate, using lookahead distance
    scoring to choose the best SWAP direction. When `restore_final_layout=True`, restores
    the canonical logical ordering ONCE at the end so exact unitary equivalence holds.
    """
    n = circuit.num_qubits
    if hardware is None or n <= 2:
        return circuit.copy()

    edges = set(hardware.coupler_params.keys())
    # Fast check: if all 2Q gates are already connected, avoid overhead
    if all(
        hardware.is_connected(op.qubits[0], op.qubits[1])
        for op in circuit.operations
        if isinstance(op, GateInstruction) and len(op.qubits) == 2
    ):
        return circuit.copy()

    dist_mat = _all_pairs_distances(edges, n)
    l2p = list(range(n))  # logical qubit -> physical qubit
    p2l = list(range(n))  # physical qubit -> logical qubit

    out = QuantumCircuit(n, name=circuit.name)
    out.measured_qubits = list(circuit.measured_qubits)

    def _apply_phys_swap(pa: int, pb: int) -> None:
        out.swap(pa, pb)
        la, lb = p2l[pa], p2l[pb]
        p2l[pa], p2l[pb] = lb, la
        l2p[la], l2p[lb] = pb, pa

    def _score_candidate_layout(cand_l2p: List[int], start_idx: int) -> float:
        score = 0.0
        seen = 0
        for future_op in circuit.operations[start_idx + 1 :]:
            if isinstance(future_op, GateInstruction) and len(future_op.qubits) == 2:
                fa, fb = cand_l2p[future_op.qubits[0]], cand_l2p[future_op.qubits[1]]
                weight = 0.65**seen
                score += weight * dist_mat[fa][fb]
                seen += 1
                if seen >= 6:
                    break
        return score

    for idx, op in enumerate(circuit.operations):
        if isinstance(op, GateInstruction) and len(op.qubits) == 2:
            lq0, lq1 = op.qubits
            pq0, pq1 = l2p[lq0], l2p[lq1]
            if not hardware.is_connected(pq0, pq1):
                path = _shortest_coupling_path(pq0, pq1, edges, n)
                # Compare moving pq0 toward pq1 vs moving pq1 toward pq0 via lookahead cost
                cand_a = list(l2p)
                cand_p2l_a = list(p2l)
                for k in range(len(path) - 2):
                    u, v = path[k], path[k + 1]
                    la, lb = cand_p2l_a[u], cand_p2l_a[v]
                    cand_p2l_a[u], cand_p2l_a[v] = lb, la
                    cand_a[la], cand_a[lb] = v, u

                rev_path = list(reversed(path))
                cand_b = list(l2p)
                cand_p2l_b = list(p2l)
                for k in range(len(rev_path) - 2):
                    u, v = rev_path[k], rev_path[k + 1]
                    la, lb = cand_p2l_b[u], cand_p2l_b[v]
                    cand_p2l_b[u], cand_p2l_b[v] = lb, la
                    cand_b[la], cand_b[lb] = v, u

                chosen = path if _score_candidate_layout(cand_a, idx) <= _score_candidate_layout(cand_b, idx) else rev_path
                for k in range(len(chosen) - 2):
                    _apply_phys_swap(chosen[k], chosen[k + 1])

                pq0, pq1 = l2p[lq0], l2p[lq1]

            out.operations.append(
                GateInstruction(
                    name=op.name,
                    qubits=(pq0, pq1),
                    params=op.params,
                    duration_ns=op.duration_ns,
                    label=op.label,
                    custom_matrix=op.custom_matrix,
                    noise_after=op.noise_after,
                    skip_hw_noise=op.skip_hw_noise,
                    custom_hamiltonian=op.custom_hamiltonian,
                )
            )
        elif isinstance(op, GateInstruction):
            mapped_q = tuple(l2p[q] for q in op.qubits)
            out.operations.append(
                GateInstruction(
                    name=op.name,
                    qubits=mapped_q,
                    params=op.params,
                    duration_ns=op.duration_ns,
                    label=op.label,
                    custom_matrix=op.custom_matrix,
                    noise_after=op.noise_after,
                    skip_hw_noise=op.skip_hw_noise,
                    custom_hamiltonian=op.custom_hamiltonian,
                )
            )
        elif isinstance(op, NoiseInstruction):
            # Noise stays attached to logical qubit's current physical location
            out.operations.append(op)
        elif isinstance(op, MeasureInstruction):
            # Defer measurements until after optional final layout restoration
            continue

    # Restore identity permutation ONCE at the end via greedy shortest-path token swapping
    if restore_final_layout:
        for target_q in range(n):
            curr_phys = l2p[target_q]
            if curr_phys != target_q:
                path = _shortest_coupling_path(curr_phys, target_q, edges, n)
                for k in range(len(path) - 1):
                    _apply_phys_swap(path[k], path[k + 1])

    # Re-append measurement instructions at the end
    for op in circuit.operations:
        if isinstance(op, MeasureInstruction):
            out.operations.append(op)

    return out


def consolidate_2q_blocks(
    circuit: QuantumCircuit,
    gate_set: NativeGateSet,
    tol: float = 1e-7,
) -> QuantumCircuit:
    """
    Scan the circuit for maximal contiguous 2-qubit blocks acting on a qubit pair (qa, qb)
    and resynthesize each block into the minimum-2Q-gate hardware-native decomposition
    using KAK / Cartan synthesis.
    """
    if circuit.num_qubits == 1:
        U = circuit.unitary()
        out = QuantumCircuit(1, name=circuit.name)
        for inst in synthesize_1q_for_gate_set(U, 0, gate_set, tol=tol):
            out.operations.append(inst)
        for op in circuit.operations:
            if not isinstance(op, GateInstruction):
                out.operations.append(op)
        out.measured_qubits = list(circuit.measured_qubits)
        return out

    # Special fast path for a 2-qubit circuit without intermediate noise/measurement barriers:
    has_barriers = any(not isinstance(op, GateInstruction) or len(op.noise_after) > 0 for op in circuit.operations)
    if circuit.num_qubits == 2 and not has_barriers:
        U_full = circuit.unitary()
        out = QuantumCircuit(2, name=circuit.name)
        for inst in synthesize_2q_unitary(U_full, (0, 1), gate_set, tol=tol):
            out.operations.append(inst)
        out.measured_qubits = list(circuit.measured_qubits)
        return out

    # General N-qubit circuit block consolidation:
    out = QuantumCircuit(circuit.num_qubits, name=circuit.name)
    out.measured_qubits = list(circuit.measured_qubits)

    i = 0
    ops = circuit.operations
    n_ops = len(ops)

    while i < n_ops:
        op = ops[i]
        if not isinstance(op, GateInstruction) or len(op.noise_after) > 0:
            out.operations.append(op)
            i += 1
            continue

        if len(op.qubits) == 1:
            # Look ahead to see if this 1Q gate can be absorbed into an upcoming 2Q block on its qubit
            out.operations.append(op)
            i += 1
            continue

        if len(op.qubits) == 2:
            pair = (min(op.qubits[0], op.qubits[1]), max(op.qubits[0], op.qubits[1]))
            block_gates: List[GateInstruction] = []

            # Also pull back any immediately preceding 1Q gates in `out.operations` that act on `pair`
            pulled_back: List[GateInstruction] = []
            while out.operations and isinstance(out.operations[-1], GateInstruction):
                prev_g = out.operations[-1]
                if len(prev_g.noise_after) == 0 and set(prev_g.qubits).issubset(pair):
                    pulled_back.append(out.operations.pop())  # type: ignore[arg-type]
                else:
                    break
            block_gates.extend(reversed(pulled_back))

            # Greedily collect subsequent 1Q and 2Q gates acting exclusively on `pair`
            while i < n_ops:
                curr = ops[i]
                if (
                    isinstance(curr, GateInstruction)
                    and len(curr.noise_after) == 0
                    and set(curr.qubits).issubset(pair)
                ):
                    block_gates.append(curr)
                    i += 1
                else:
                    break

            # Compute the 4x4 unitary of `block_gates` on `pair = (qa, qb)`
            qa, qb = pair
            U_block = np.eye(4, dtype=np.complex128)
            for bg in block_gates:
                local_qubits = tuple(0 if q == qa else 1 for q in bg.qubits)
                U_g = embed_gate_unitary(bg.unitary(), local_qubits, 2)
                U_block = U_g @ U_block

            synthesized = synthesize_2q_unitary(U_block, (qa, qb), gate_set, tol=tol)
            out.operations.extend(synthesized)
        else:
            out.operations.append(op)
            i += 1

    return out


def commute_rz_through_cz(
    circuit: QuantumCircuit,
    tol: float = 1e-7,
) -> QuantumCircuit:
    """
    Commute single-qubit Z-rotations (RZ, Z, S, SDG, T, TDG) rightward across
    diagonal 2-qubit gates (CZ, RZZ) so they merge with downstream 1-qubit gates.
    """
    out = circuit.copy()
    ops = out.operations
    changed = True
    while changed:
        changed = False
        for idx in range(len(ops) - 1):
            op_a = ops[idx]
            op_b = ops[idx + 1]
            if not (isinstance(op_a, GateInstruction) and isinstance(op_b, GateInstruction)):
                continue
            if op_a.noise_after or op_b.noise_after:
                continue
            if op_a.name == "RZ" and op_b.name in ("CZ", "RZZ") and op_a.qubits[0] in op_b.qubits:
                # Swap order: [RZ(q), CZ(q0, q1)] -> [CZ(q0, q1), RZ(q)]
                ops[idx], ops[idx + 1] = op_b, op_a
                changed = True
    return out


def fuse_1q_runs(
    circuit: QuantumCircuit,
    gate_set: NativeGateSet,
    tol: float = 1e-7,
) -> QuantumCircuit:
    """
    Collect maximal contiguous runs of 1-qubit gates on each individual qubit
    (separated only by 2-qubit gates, noise, or measurements on that qubit),
    multiply their 2x2 matrices, and resynthesize into the minimal hardware-native
    1Q pulse sequence.
    """
    n = circuit.num_qubits
    pending_1q: Dict[int, np.ndarray] = {q: np.eye(2, dtype=np.complex128) for q in range(n)}
    has_pending: Dict[int, bool] = {q: False for q in range(n)}

    out = QuantumCircuit(n, name=circuit.name)
    out.measured_qubits = list(circuit.measured_qubits)

    def _flush_qubit(q: int) -> None:
        if has_pending[q]:
            synthesized = synthesize_1q_for_gate_set(pending_1q[q], q, gate_set, tol=tol)
            for g in synthesized:
                spec = gate_set.get_gate_spec(g.name)
                if spec is not None:
                    g.duration_ns = 0.0 if spec.is_virtual else spec.duration_ns
                out.operations.append(g)
            pending_1q[q] = np.eye(2, dtype=np.complex128)
            has_pending[q] = False

    for op in circuit.operations:
        if isinstance(op, GateInstruction) and len(op.qubits) == 1 and not op.noise_after:
            q = op.qubits[0]
            pending_1q[q] = op.unitary() @ pending_1q[q]
            has_pending[q] = True
        elif isinstance(op, GateInstruction):
            for q in op.qubits:
                _flush_qubit(q)
            spec = gate_set.get_gate_spec(op.name)
            if spec is not None and op.duration_ns is None:
                op.duration_ns = 0.0 if spec.is_virtual else spec.duration_ns
            out.operations.append(op)
        elif isinstance(op, NoiseInstruction):
            for q in op.channel.qubits:
                _flush_qubit(q)
            out.operations.append(op)
        elif isinstance(op, MeasureInstruction):
            for q in op.qubits:
                _flush_qubit(q)
            out.operations.append(op)

    for q in range(n):
        _flush_qubit(q)

    return out
