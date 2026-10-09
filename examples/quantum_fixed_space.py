#!/usr/bin/env python3
"""Bounded numerical fixed-density SDP research, outside the Rust runtime.

For d=2..4 and Phi(rho)=sum_j K_j rho K_j*, solve min/max Tr(E rho)
over Hermitian rho >= 0, Tr(rho)=1, Phi(rho)=rho. A full POVM consists of
1..8 PSD effects summing to I; E sums a selected outcome subset. Its exact
mathematical feasible set is compact for a CPTP map. Numeric tolerance-based
admission and SCS residuals do NOT certify that exact mathematical assertion.

For minimization the mathematical dual is max lambda subject to
E - lambda I + Phi*(Y) - Y >= 0, Y Hermitian. The report checks its numerical
slack as well as SCS's canonical residuals. Floating-point primal/dual evidence
is neither an exact feasibility certificate nor a rigorous enclosure of all
fixed states. In particular, tiny nonzero fixed constraints can be lost by a
solver. Labels explicitly remain estimates, uncertain or unsupported.

Reproduce on CPython3.12/Linux x86_64 in a fresh venv (no global installation):
  python3 -m venv /tmp/ouro-quantum
  python3 examples/quantum_fixed_space.py --requirements > /tmp/ouro-quantum.lock
  /tmp/ouro-quantum/bin/python -m pip install --only-binary=:all: \
      --require-hashes -r /tmp/ouro-quantum.lock
  /tmp/ouro-quantum/bin/python tests/quantum_research.py

The wheel lock below is the tested CPython3.12/Linux x86_64 wheel set. Other
platforms require a separately qualified lock, not silent hash substitution.
CLI JSON matrices use d rows of d [real,imag] pairs. Input is capped at256KiB;
outer OS/process bounds are provided by tests/quantum_research.py.

Primary implementation references:
https://www.cvxpy.org/tutorial/constraints/index.html
https://www.cvxgrp.org/scs/algorithm/index.html
https://www.cvxgrp.org/scs/api/settings.html
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import sys
from typing import Any
import warnings

for _thread_variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_thread_variable] = "1"

VERSION = 1
SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
MAX_INPUT_BYTES = 256 * 1024
MAX_KRAUS = 16
MAX_OUTCOMES = 8
QUALIFIED_REQUIREMENTS = """cffi==2.1.1 --hash=sha256:c1453022f490d2459a11819d83ad1d586e9ff65a12ac3e705ffebd46d3685dcf
clarabel==0.11.1 --hash=sha256:c8c41aaa6f3f8c0f3bd9d86c3e568dcaee079562c075bd2ec9fb3a80287380ef
cloudpickle==3.1.2 --hash=sha256:9acb47f6afd73f60dc1df93bb801b472f05ff42fa6c84167d25cb206be1fbf4a
cvxpy==1.9.3 --hash=sha256:88b28c6df62d49e7e59b1a08ae2f90caf4019d496208ec697aa018169220cdcf
highspy==1.15.1 --hash=sha256:9730647160a6481426729f46d9989a0507d05f3cf96f9fb180f4ab9891bea67b
jinja2==3.1.6 --hash=sha256:85ece4451f492d0c13c5dd7c13a64681a86afae63a5f347908daf103ce6d2f67
joblib==1.6.0 --hash=sha256:3dbbf9f6e4b592a2357b854608e980fe6390d131d7a82f011a377ef2ebef7aba
markupsafe==3.0.4 --hash=sha256:8e124f974786f831d6043728e38296969d3579db8896fe004682f5758e613581
numpy==2.5.3 --hash=sha256:b7e18c623bb5c95acb3b3328861272816ba199fb531921c5d6d0b675f1fde9e3
osqp==1.1.3 --hash=sha256:6ceef7fb4f332892b6e0bbc17323d5c9e028c3f9db726b62a15d876d0f81cc06
pycparser==3.0 --hash=sha256:b727414169a36b7d524c1c3e31839a521725078d7b2ff038656844266160a992
qdldl==0.1.9.post1 --hash=sha256:aa0e9721d272467c95a9748e6acea8911a12041ef3d8b20176aff829388ce57d
scipy==1.18.1 --hash=sha256:f55fa87b6c612ecd6b058f167c53231b1d14e412efe361d3d6e38b3631c73218
scs==3.3.1 --hash=sha256:38e8eeb2b8f43f3109569862d398ca787ab4e4a50e16d1148206ed1cecd1840f
setuptools==84.0.0 --hash=sha256:51a52592b3b99e102b609654876bd65f19f999935166d1352678931132b0c670
sparsediffpy==0.6.1 --hash=sha256:193618b27958588983134827e1b2137caf8e80fd9346f0a98c38e1fb938b2c40
"""


class ResearchError(ValueError):
    def __init__(self, category: str, message: str):
        super().__init__(message)
        self.category = category


@dataclass(frozen=True)
class Config:
    version: int = VERSION
    max_iters: int = 20_000
    time_limit_secs: float = 3.0
    solver_eps: float = 1e-8
    input_tolerance: float = 1e-10
    evidence_tolerance: float = 2e-6
    conditioning_floor: float = 1e-7
    analysis_margin: float = 1e-5
    accept_threshold: float = 2 / 3
    reject_threshold: float = 1 / 3

    def validate(self) -> None:
        if type(self.version) is not int or self.version != VERSION:
            raise ResearchError("unsupported", "configuration version")
        if type(self.max_iters) is not int or not 1 <= self.max_iters <= 20_000:
            raise ResearchError("unsupported", "iteration cap")
        for name, lower, upper in (
            ("time_limit_secs", 1e-6, 3.0),
            ("solver_eps", 1e-12, 1e-3),
            ("input_tolerance", 1e-14, 1e-8),
            ("evidence_tolerance", 1e-12, 1e-3),
            ("conditioning_floor", 1e-12, 1e-3),
            ("analysis_margin", 1e-12, 1e-3),
            ("accept_threshold", 0.0, 1.0),
            ("reject_threshold", 0.0, 1.0),
        ):
            value = getattr(self, name)
            # JSON integers need not fit a float. Reject their exact range
            # before math.isfinite performs a potentially overflowing cast.
            if type(value) not in (float, int) or not lower <= value <= upper or not math.isfinite(value):
                raise ResearchError("invalid_input", f"configuration {name}")
        if self.reject_threshold >= self.accept_threshold:
            raise ResearchError("invalid_input", "threshold ordering")


def _libraries():
    try:
        import cvxpy as cp
        import numpy as np
        import scs
    except ImportError as error:
        raise ResearchError("unsupported", "install the isolated hash-pinned dependency lock") from error
    return cp, np, scs


def runtime_manifest() -> dict[str, Any]:
    versions = {}
    for line in QUALIFIED_REQUIREMENTS.splitlines():
        package, required = line.split(" ")[0].split("==")
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            actual = None
        versions[package] = {"required": required, "actual": actual}
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "dependency_versions": versions,
        "dependency_lock_sha256": hashlib.sha256(QUALIFIED_REQUIREMENTS.encode()).hexdigest(),
        "prototype_source_sha256": SOURCE_SHA256,
        "qualified_versions": all(item["required"] == item["actual"] for item in versions.values()),
        "threads": {name: os.environ[name] for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
    }


def _matrix(value, dimension: int | None = None):
    _, np, _ = _libraries()
    if isinstance(value, np.ndarray):
        shape = value.shape
    elif isinstance(value, (list, tuple)):
        if not 2 <= len(value) <= 4 or any(not isinstance(row, (list, tuple)) or len(row) != len(value) for row in value):
            raise ResearchError("unsupported", "matrix dimension must be2..4")
        shape = (len(value), len(value))
    else:
        raise ResearchError("invalid_input", "matrix representation")
    if len(shape) != 2 or shape[0] != shape[1] or not 2 <= shape[0] <= 4:
        raise ResearchError("unsupported", "matrix dimension must be2..4")
    if dimension is not None and shape != (dimension, dimension):
        raise ResearchError("invalid_input", "matrix dimension mismatch")
    try:
        matrix = np.array(value, dtype=np.complex128, copy=True)
    except (TypeError, ValueError, OverflowError) as error:
        raise ResearchError("invalid_input", "finite IEEE matrix entries required") from error
    if not np.isfinite(matrix).all() or np.max(np.abs(matrix)) > 2:
        raise ResearchError("invalid_input", "nonfinite or unbounded matrix entry")
    return matrix


def validate(kraus, povm, selected_outcomes, config: Config):
    _, np, _ = _libraries()
    config.validate()
    if not isinstance(kraus, (list, tuple)) or not 1 <= len(kraus) <= MAX_KRAUS:
        raise ResearchError("unsupported", "Kraus count must be1..16")
    if not isinstance(povm, (list, tuple)) or not 1 <= len(povm) <= MAX_OUTCOMES:
        raise ResearchError("unsupported", "POVM outcome count must be1..8")
    first = _matrix(kraus[0])
    dimension = first.shape[0]
    operators = [first] + [_matrix(operator, dimension) for operator in kraus[1:]]
    effects = [_matrix(effect, dimension) for effect in povm]
    if not isinstance(selected_outcomes, (list, tuple)) or len(selected_outcomes) > len(effects):
        raise ResearchError("invalid_input", "selected outcome subset")
    if any(type(index) is not int or index < 0 or index >= len(effects) for index in selected_outcomes) or len(set(selected_outcomes)) != len(selected_outcomes):
        raise ResearchError("invalid_input", "selected outcome index or duplicate")
    selected = tuple(sorted(selected_outcomes))
    eye = np.eye(dimension)
    gram = sum((operator.conj().T @ operator for operator in operators), np.zeros_like(first))
    tp_residual = float(np.linalg.norm(gram - eye, "fro"))
    normalization = float(np.linalg.norm(sum(effects, np.zeros_like(first)) - eye, "fro"))
    hermiticity = max(float(np.linalg.norm(effect - effect.conj().T, "fro")) for effect in effects)
    eigenvalues = [np.linalg.eigvalsh((effect + effect.conj().T) / 2) for effect in effects]
    minimum = min(float(eigenvalue[0]) for eigenvalue in eigenvalues)
    maximum = max(float(eigenvalue[-1]) for eigenvalue in eigenvalues)
    if not all(math.isfinite(value) for value in (tp_residual, normalization, hermiticity, minimum, maximum)):
        raise ResearchError("invalid_input", "nonfinite validation arithmetic")
    tolerance = config.input_tolerance
    if tp_residual > tolerance:
        raise ResearchError("invalid_input", "Kraus trace preservation exceeds numeric admission tolerance")
    if normalization > tolerance or hermiticity > tolerance or minimum < -tolerance or maximum > 1 + tolerance:
        raise ResearchError("invalid_input", "POVM PSD/Hermiticity/normalization exceeds numeric admission tolerance")
    combined = sum((effects[index] for index in selected), np.zeros_like(first))
    validation = {
        "numeric_admission_only": True,
        "finite_binary_floating_matrix_data": True,
        "trace_preservation_fro": tp_residual,
        "povm_normalization_fro": normalization,
        "effect_hermiticity_max_fro": hermiticity,
        "minimum_effect_eigenvalue": minimum,
        "maximum_effect_eigenvalue": maximum,
    }
    return operators, effects, selected, combined, validation


def _conditioning(operators, config: Config):
    _, np, _ = _libraries()
    dimension = operators[0].shape[0]
    fixed = sum((np.kron(operator.conj(), operator) for operator in operators), np.zeros((dimension**2, dimension**2), dtype=complex)) - np.eye(dimension**2)
    singular = np.linalg.svd(fixed, compute_uv=False)
    magnitude = float(np.linalg.norm(fixed, "fro"))
    roundoff_floor = float(16 * dimension**2 * np.finfo(float).eps * max(1, magnitude))
    nonzero = singular[singular > roundoff_floor]
    return {
        "singular_values": [float(value) for value in singular],
        "fixed_constraint_fro": magnitude,
        "roundoff_floor": roundoff_floor,
        "heuristic_numerical_rank": int(np.count_nonzero(singular > config.conditioning_floor)),
        "smallest_resolved_singular_value": float(nonzero[-1]) if len(nonzero) else None,
        "ill_conditioned": bool((len(nonzero) and nonzero[-1] < config.conditioning_floor) or (magnitude != 0 and not len(nonzero))),
        "rank_is_not_an_exact_fixed_space_certificate": True,
    }


def _scalar(value) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _complex_json(matrix):
    return [[[float(entry.real), float(entry.imag)] for entry in row] for row in matrix]


def _solve(operators, effect, config: Config, sign: int) -> dict[str, Any]:
    cp, np, scs = _libraries()
    dimension = effect.shape[0]
    rho = cp.Variable((dimension, dimension), hermitian=True)
    mapped = sum((operator @ rho @ operator.conj().T for operator in operators))
    constraints = [rho >> 0, cp.trace(rho) == 1, mapped == rho]
    problem = cp.Problem(cp.Minimize(sign * cp.real(cp.trace(effect @ rho))), constraints)
    result: dict[str, Any] = {
        "direction": "minimum" if sign == 1 else "maximum",
        "claim": "numeric_only",
        "status": None,
        "estimate": None,
        "usable_estimate": False,
    }
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            problem.solve(
                solver=cp.SCS,
                eps_abs=config.solver_eps,
                eps_rel=config.solver_eps,
                max_iters=config.max_iters,
                time_limit_secs=config.time_limit_secs,
                warm_start=False,
                verbose=False,
                linear_solver=scs.LinearSolver.QDLDL,
            )
        result["warnings"] = [str(warning.message)[:512] for warning in caught[:8]]
    except (cp.error.SolverError, ValueError, ArithmeticError) as error:
        result["status"] = "solver_error"
        result["error"] = str(error)[:512]
        return result
    result["status"] = problem.status
    stats = problem.solver_stats
    raw = (stats.extra_stats or {}).get("info", {})
    result["scs"] = {name: _scalar(raw.get(name)) for name in ("pobj", "dobj", "res_pri", "res_dual", "gap", "iter", "solve_time", "setup_time")}
    result["scs"]["status"] = str(raw.get("status", "unavailable"))[:128]
    result["scs"]["status_val"] = _scalar(raw.get("status_val"))
    result["scs"]["canonical_objectives_are_signed"] = True
    if rho.value is None:
        return result
    state = np.asarray(rho.value)
    if state.shape != (dimension, dimension) or not np.isfinite(state).all():
        result["numeric_evidence_status"] = "nonfinite_state"
        return result
    # A density matrix has entry magnitudes <=1. This loose finite guard also
    # prevents corrupt solver iterates from overflowing subsequent evidence.
    if np.max(np.abs(state)) > 2:
        result["numeric_evidence_status"] = "unbounded_numeric_state"
        return result
    hermitian_state = (state + state.conj().T) / 2
    objective = np.trace(effect @ state)
    transformed = sum((operator @ state @ operator.conj().T for operator in operators), np.zeros_like(state))
    evidence = {
        "hermiticity_fro": float(np.linalg.norm(state - state.conj().T, "fro")),
        "trace_error": float(abs(np.trace(state) - 1)),
        "minimum_rho_eigenvalue": float(np.linalg.eigvalsh(hermitian_state)[0]),
        "fixed_residual_fro": float(np.linalg.norm(transformed - state, "fro")),
        "objective_imaginary": float(abs(objective.imag)),
        "reported_objective_disagreement": float(abs(sign * objective.real - problem.value)),
    }
    trace_dual = constraints[1].dual_value
    fixed_dual = constraints[2].dual_value
    if trace_dual is not None and fixed_dual is not None:
        lagrange = float(np.real(trace_dual))
        try:
            with np.errstate(over="raise", invalid="raise"):
                multiplier = (np.asarray(fixed_dual) + np.asarray(fixed_dual).conj().T) / 2
                dual_slack = sign * effect + lagrange * np.eye(dimension) + sum((operator.conj().T @ multiplier @ operator for operator in operators), np.zeros_like(multiplier)) - multiplier
                if math.isfinite(lagrange) and np.isfinite(dual_slack).all():
                    eigenvalues = np.linalg.eigvalsh((dual_slack + dual_slack.conj().T) / 2)
                    evidence.update({"dual_objective_readout": -sign * lagrange, "dual_slack_minimum_eigenvalue": float(eigenvalues[0]), "original_primal_dual_gap": float(abs(sign * objective.real + lagrange))})
        except (FloatingPointError, np.linalg.LinAlgError):
            result["dual_evidence_status"] = "nonfinite_or_unavailable"
    result["estimate"] = _scalar(objective.real)
    result["rho"] = _complex_json(state)
    result["matrix_evidence"] = evidence
    tolerance = config.evidence_tolerance
    required = ("hermiticity_fro", "trace_error", "fixed_residual_fro", "objective_imaginary", "reported_objective_disagreement", "original_primal_dual_gap")
    result["usable_estimate"] = bool(
        problem.status == cp.OPTIMAL
        and all(
            name in evidence
            and math.isfinite(evidence[name])
            and evidence[name] <= tolerance
            for name in required
        )
        and evidence["minimum_rho_eigenvalue"] >= -tolerance
        and evidence.get("dual_slack_minimum_eigenvalue", -math.inf) >= -tolerance
        and all(
            result["scs"].get(name) is not None
            and abs(result["scs"][name]) <= tolerance
            for name in ("res_pri", "res_dual", "gap")
        )
        and result["estimate"] is not None
        and -tolerance <= result["estimate"] <= 1 + tolerance
    )
    return result


def analyze(kraus, povm, selected_outcomes=(0,), config=Config()) -> dict[str, Any]:
    _, np, _ = _libraries()
    runtime = runtime_manifest()
    if not runtime["qualified_versions"]:
        raise ResearchError("unsupported", "dependency versions differ from the qualified wheel lock")
    operators, effects, selected, combined, validation = validate(kraus, povm, selected_outcomes, config)
    binding = hashlib.sha256(b"ourochronos.quantum-sdp-research/v1\0")
    binding.update(json.dumps({"version": VERSION, "config": asdict(config), "selected_outcomes": selected, "runtime": runtime}, sort_keys=True, allow_nan=False).encode())
    for group in (operators, effects):
        binding.update(len(group).to_bytes(4, "little"))
        for matrix in group:
            binding.update(matrix.shape[0].to_bytes(4, "little"))
            binding.update(np.asarray(matrix, dtype="<c16").tobytes(order="C"))
    conditioning = _conditioning(operators, config)
    minimum = _solve(operators, combined, config, 1)
    maximum = _solve(operators, combined, config, -1)
    usable = minimum["usable_estimate"] and maximum["usable_estimate"] and not conditioning["ill_conditioned"]
    label = "numerical_uncertain"
    if usable:
        lower, upper = minimum["estimate"], maximum["estimate"]
        margin = config.analysis_margin
        if lower > upper + config.evidence_tolerance:
            usable = False
        elif lower - margin >= config.accept_threshold:
            label = "numerical_accept_estimate"
        elif upper + margin <= config.reject_threshold:
            label = "numerical_reject_estimate"
        elif lower + margin < config.accept_threshold and upper - margin > config.reject_threshold:
            label = "numerical_ambiguous_estimate"
    return {
        "schema_version": VERSION,
        "model_config_runtime_sha256": binding.hexdigest(),
        "runtime": runtime,
        "config": asdict(config),
        "dimension": operators[0].shape[0],
        "kraus_count": len(operators),
        "povm_outcome_count": len(effects),
        "selected_outcomes": list(selected),
        "claim": "numeric_only_no_global_certificate",
        "input_validation": validation,
        "conditioning": conditioning,
        "minimum": minimum,
        "maximum": maximum,
        "usable_extrema_estimates": bool(usable),
        "threshold_label": label,
        "analysis_margin_is_not_a_rigorous_error_bound": True,
        "unsupported": [
            "exact_amplitude_certification",
            "dimension_above4",
            "maximum_entropy_selection",
            "exact_global_threshold_decision",
        ],
    }


def _json_matrix(value):
    if not isinstance(value, list) or not 2 <= len(value) <= 4 or any(not isinstance(row, list) or len(row) != len(value) for row in value):
        raise ResearchError("unsupported", "JSON matrix dimension must be2..4")
    result = []
    for row in value:
        parsed = []
        for pair in row:
            if not isinstance(pair, list) or len(pair) != 2 or any(type(number) not in (int, float) for number in pair):
                raise ResearchError("invalid_input", "IEEE [real,imag] pairs required; exact symbolic amplitudes unsupported")
            try:
                entry = complex(pair[0], pair[1])
            except (ValueError, OverflowError) as error:
                raise ResearchError("invalid_input", "matrix number overflow") from error
            if not math.isfinite(entry.real) or not math.isfinite(entry.imag):
                raise ResearchError("invalid_input", "nonfinite JSON matrix entry")
            parsed.append(entry)
        result.append(parsed)
    return result


def load_input(path: Path):
    with path.open("rb") as handle:
        contents = handle.read(MAX_INPUT_BYTES + 1)
    if len(contents) > MAX_INPUT_BYTES:
        raise ResearchError("unsupported", "JSON input byte cap")
    def unique_fields(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON field")
            result[key] = value
        return result

    def finite_constant(_):
        raise ValueError("nonfinite JSON constant")

    try:
        value = json.loads(
            contents,
            parse_constant=finite_constant,
            object_pairs_hook=unique_fields,
        )
    except (ValueError, UnicodeError, RecursionError) as error:
        raise ResearchError("invalid_input", "invalid bounded JSON input") from error
    if not isinstance(value, dict) or set(value) - {"kraus", "povm", "selected_outcomes", "config"} or "kraus" not in value or "povm" not in value:
        raise ResearchError("invalid_input", "JSON input fields")
    groups = []
    for name, cap in (("kraus", MAX_KRAUS), ("povm", MAX_OUTCOMES)):
        group = value[name]
        if not isinstance(group, list) or not 1 <= len(group) <= cap:
            raise ResearchError("unsupported", f"JSON {name} count")
        groups.append([_json_matrix(matrix) for matrix in group])
    try:
        config = Config(**value.get("config", {}))
    except TypeError as error:
        raise ResearchError("invalid_input", "JSON configuration fields") from error
    return *groups, value.get("selected_outcomes", [0]), config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requirements", action="store_true")
    parser.add_argument("--input", type=Path)
    arguments = parser.parse_args()
    if arguments.requirements:
        print(QUALIFIED_REQUIREMENTS, end="")
        return 0
    if arguments.input is None:
        parser.error("provide --input or --requirements")
    try:
        result = analyze(*load_input(arguments.input))
    except (ResearchError, OSError) as error:
        print(json.dumps({"schema_version": VERSION, "claim": "no_certificate", "status": getattr(error, "category", "input_io_error"), "error": str(error)}, allow_nan=False))
        return 2
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
