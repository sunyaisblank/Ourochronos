#!/usr/bin/env python3
"""Independent finite analytic/adversarial campaign for the quantum SDP example.

Runs the working process with120s wall/90s CPU/2GiB address-space bounds,
single-thread BLAS, no core files and16MiB output-file cap on Linux. At most80
named cases are recorded; each SDP direction has its own3s/20000-iteration cap.
No global Python/Rust dependencies are installed by this program.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
MAX_CASES = 80
WALL_SECONDS = 120
REPORT = []


def _limits():
    if sys.platform != "linux":
        raise RuntimeError("qualified research process limits require Linux")
    import resource

    resource.setrlimit(resource.RLIMIT_CPU, (90, 90))
    resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
    resource.setrlimit(resource.RLIMIT_FSIZE, (16 * 1024**2, 16 * 1024**2))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[variable] = "1"


def _load_prototype():
    spec = importlib.util.spec_from_file_location("quantum_fixed_space", ROOT / "examples/quantum_fixed_space.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _record(value):
    if len(REPORT) >= MAX_CASES:
        raise AssertionError("explicit finite case cap exhausted")
    REPORT.append(value)


def run_worker():
    _limits()
    prototype = _load_prototype()
    import numpy as np

    # Analytic references use spectral/diagonal/block formulas, never the
    # prototype's SDP/conditioning/range computation or solver witness.
    def effect(dimension):
        matrix = np.diag(np.linspace(0.2, 0.8, dimension)).astype(complex)
        for index in range(dimension - 1):
            matrix[index, index + 1] = 0.035 + 0.04j
            matrix[index + 1, index] = 0.035 - 0.04j
        return matrix

    def dephasing(dimension):
        matrices = []
        for index in range(dimension):
            projector = np.zeros((dimension, dimension), dtype=complex)
            projector[index, index] = 1
            matrices.append(projector)
        return matrices

    def mixed_reset(dimension):
        weights = np.arange(1, dimension + 1, dtype=float)
        weights /= weights.sum()
        matrices = []
        for destination in range(dimension):
            for source in range(dimension):
                matrix = np.zeros((dimension, dimension), dtype=complex)
                matrix[destination, source] = np.sqrt(weights[destination])
                matrices.append(matrix)
        return matrices, np.diag(weights)

    def pure_reset(vector):
        matrices = []
        for source in range(len(vector)):
            matrix = np.zeros((len(vector), len(vector)), dtype=complex)
            matrix[:, source] = vector
            matrices.append(matrix)
        return matrices

    class Campaign(unittest.TestCase):
        def analyze_case(self, name, kraus, povm, reference=None, subset=(0,), config=prototype.Config(), usable=True):
            result = prototype.analyze(kraus, povm, subset, config)
            self.assertEqual(result["claim"], "numeric_only_no_global_certificate")
            self.assertTrue(result["analysis_margin_is_not_a_rigorous_error_bound"])
            self.assertTrue(result["input_validation"]["numeric_admission_only"])
            self.assertTrue(result["threshold_label"].startswith("numerical_"))
            self.assertEqual(len(result["model_config_runtime_sha256"]), 64)
            json.dumps(result, allow_nan=False)
            if usable:
                self.assertTrue(result["usable_extrema_estimates"], (name, result))
                for direction in ("minimum", "maximum"):
                    evidence = result[direction]
                    self.assertEqual(evidence["status"], "optimal")
                    self.assertTrue(evidence["usable_estimate"])
                    self.assertTrue(all(evidence["scs"][item] is not None for item in ("res_pri", "res_dual", "gap")))
                if reference is not None:
                    self.assertAlmostEqual(result["minimum"]["estimate"], reference[0], delta=2e-5, msg=name)
                    self.assertAlmostEqual(result["maximum"]["estimate"], reference[1], delta=2e-5, msg=name)
            _record({"name": name, "analytic_reference": reference, "result": result})
            return result

        def reject_case(self, name, kraus, povm, subset=(0,), config=prototype.Config(), expected=None):
            with self.assertRaises(prototype.ResearchError) as caught:
                prototype.analyze(kraus, povm, subset, config)
            if expected is not None:
                self.assertEqual(caught.exception.category, expected, name)
            _record({"name": name, "rejected": caught.exception.category, "reason": str(caught.exception)})

        def test_analytic_identity_reset_dephasing_and_block_families(self):
            for dimension in (2, 3, 4):
                eye = np.eye(dimension)
                readout = effect(dimension)
                povm = [readout, eye - readout]
                eigenvalues = np.linalg.eigvalsh(readout)
                self.analyze_case(f"identity-d{dimension}", [eye], povm, (float(eigenvalues[0]), float(eigenvalues[-1])))
                reset, sigma = mixed_reset(dimension)
                expected = float(np.trace(readout @ sigma).real)
                self.analyze_case(f"mixed-reset-d{dimension}", reset, povm, (expected, expected))
                vector = np.exp(2j * np.pi * np.arange(dimension) / dimension) / np.sqrt(dimension)
                expected = float(np.vdot(vector, readout @ vector).real)
                self.analyze_case(f"complex-pure-reset-d{dimension}", pure_reset(vector), povm, (expected, expected))
                damping = 0.3
                damped = [np.diag([1] + [np.sqrt(1 - damping)] * (dimension - 1)).astype(complex)]
                for source in range(1, dimension):
                    jump = np.zeros((dimension, dimension), dtype=complex)
                    jump[0, source] = np.sqrt(damping)
                    damped.append(jump)
                self.analyze_case(f"amplitude-damping-d{dimension}", damped, povm, (0.2, 0.2))
                diagonal = np.diag(readout).real
                self.analyze_case(f"dephasing-d{dimension}", dephasing(dimension), povm, (float(diagonal.min()), float(diagonal.max())))
                split = dimension // 2
                first = np.diag([1] * split + [0] * (dimension - split))
                blocks = [first, eye - first]
                spectra = np.concatenate((np.linalg.eigvalsh(readout[:split, :split]), np.linalg.eigvalsh(readout[split:, split:])))
                result = self.analyze_case(f"nonunique-block-d{dimension}", blocks, povm, (float(spectra.min()), float(spectra.max())))
                # Two explicit fixed pure classes disagree on acceptance.
                self.assertLess(readout[0, 0].real, 1 / 3)
                self.assertGreater(readout[-1, -1].real, 2 / 3)
                self.assertEqual(result["threshold_label"], "numerical_ambiguous_estimate")

        def test_general_three_outcome_and_subset_povms(self):
            for dimension in (2, 3, 4):
                eye = np.eye(dimension)
                readout = effect(dimension)
                povm = [0.3 * readout, 0.4 * (eye - readout), 0.6 * eye + 0.1 * readout]
                for subset in ((0, 2), (1,)):
                    combined = sum((povm[index] for index in subset), np.zeros_like(readout))
                    eigenvalues = np.linalg.eigvalsh(combined)
                    result = self.analyze_case(f"three-outcome-d{dimension}-subset{subset}", [eye], povm, (float(eigenvalues[0]), float(eigenvalues[-1])), subset=subset)
                    self.assertEqual(result["povm_outcome_count"], 3)
            self.analyze_case("one-outcome-identity", [np.eye(2)], [np.eye(2)], (1.0, 1.0))
            self.analyze_case("eight-outcomes-selected-five", [np.eye(2)], [np.eye(2) / 8] * 8, (5 / 8, 5 / 8), subset=(0, 1, 2, 3, 4))
            self.analyze_case("empty-selected-subset", [np.eye(2)], [np.eye(2)], (0.0, 0.0), subset=())

        def test_degeneracy_and_ill_conditioned_fixed_constraints(self):
            for dimension in (2, 3, 4):
                eye = np.eye(dimension)
                self.analyze_case(f"degenerate-constant-readout-d{dimension}", [eye], [0.5 * eye, 0.5 * eye], (0.5, 0.5))
            # For the ideal CPTP family Phi=(1-weight)Id+weightReset, each
            # positive weight gives only the fixed density |0><0|. Its IEEE
            # Kraus realization approximates the ideal square-root coefficients;
            # tolerance-based admission is not an exact TP assertion about them.
            for weight in (1e-9, 1e-12):
                eye = np.eye(3)
                ground = np.array([1.0, 0.0, 0.0])
                matrices = [np.sqrt(1 - weight) * eye] + [np.sqrt(weight) * matrix for matrix in pure_reset(ground)]
                readout = effect(3)
                result = self.analyze_case(f"ill-conditioned-reset-mixture-{weight}", matrices, [readout, eye - readout], (0.2, 0.2), usable=False)
                self.assertTrue(result["conditioning"]["ill_conditioned"])
                self.assertFalse(result["usable_extrema_estimates"])
                self.assertEqual(result["threshold_label"], "numerical_uncertain")
                if weight == 1e-12:
                    # In this pinned campaign small feasible residuals coexist
                    # with a large error against the exact analytic fixed set.
                    self.assertGreater(abs(result["maximum"]["estimate"] - 0.2), 0.1)
                    self.assertLess(result["maximum"]["matrix_evidence"]["fixed_residual_fro"], 1e-8)
            unitary = np.diag(np.exp(1j * np.arange(3) * 1e-10))
            readout = effect(3)
            result = self.analyze_case("near-degenerate-unitary", [unitary], [readout, np.eye(3) - readout], (0.2, 0.8), usable=False)
            self.assertTrue(result["conditioning"]["ill_conditioned"])
            self.assertEqual(result["threshold_label"], "numerical_uncertain")

        def test_adjacent_thresholds_and_solver_exhaustion_stay_uncertain(self):
            readout = np.diag([0.2, 0.8])
            reset = pure_reset(np.array([1.0, 0.0]))
            for side in (-np.inf, np.inf):
                near = float(np.nextafter(0.2, side))
                config = replace(prototype.Config(), accept_threshold=near, reject_threshold=0.1)
                result = self.analyze_case(f"adjacent-accept-{side}", reset, [readout, np.eye(2) - readout], (0.2, 0.2), config=config)
                self.assertEqual(result["threshold_label"], "numerical_uncertain")
                config = replace(prototype.Config(), accept_threshold=0.9, reject_threshold=near)
                result = self.analyze_case(f"adjacent-reject-{side}", reset, [readout, np.eye(2) - readout], (0.2, 0.2), config=config)
                self.assertEqual(result["threshold_label"], "numerical_uncertain")
            result = self.analyze_case("one-iteration-exhaustion", [np.eye(4)], [effect(4), np.eye(4) - effect(4)], config=replace(prototype.Config(), max_iters=1), usable=False)
            self.assertFalse(result["usable_extrema_estimates"])
            self.assertEqual(result["threshold_label"], "numerical_uncertain")
            self.assertTrue(any(result[direction]["status"] == "optimal_inaccurate" for direction in ("minimum", "maximum")))

        def test_invalid_and_unsupported_inputs_reject_before_solve(self):
            eye = np.eye(2)
            normal = [eye / 2, eye / 2]
            for name, bad in (("NaN", np.nan), ("infinity", np.inf), ("overflow-scale", 1e308 + 1e308j)):
                matrix = eye.astype(complex)
                matrix[0, 0] = bad
                self.reject_case(name, [matrix], normal, expected="invalid_input")
            self.reject_case("non-trace-preserving", [0.8 * eye], normal, expected="invalid_input")
            self.reject_case("non-CPTP-transpose-map-outside-Kraus-representation", {"transpose": True}, normal, expected="unsupported")
            self.reject_case("bad-POVM-normalization", [eye], [0.4 * eye, 0.4 * eye], expected="invalid_input")
            self.reject_case("negative-POVM-effect", [eye], [-0.1 * eye, 1.1 * eye], expected="invalid_input")
            nonhermitian = np.array([[0.5, 0.1j], [0, 0.5]])
            self.reject_case("nonhermitian-POVM", [eye], [nonhermitian, eye - nonhermitian], expected="invalid_input")
            self.reject_case("five-dimensions", [np.eye(5)], [np.eye(5)], expected="unsupported")
            self.reject_case("seventeen-Kraus-operators", [eye / np.sqrt(17)] * 17, normal, expected="unsupported")
            self.reject_case("nine-POVM-outcomes", [eye], [eye / 9] * 9, expected="unsupported")
            self.reject_case("no-Kraus-operators", [], normal, expected="unsupported")
            self.reject_case("no-POVM-outcomes", [eye], [], expected="unsupported")
            for name, subset in (("duplicate-outcome", (0, 0)), ("negative-outcome", (-1,)), ("missing-outcome", (2,)), ("boolean-outcome", (True,))):
                self.reject_case(name, [eye], normal, subset=subset, expected="invalid_input")
            for name, config, expected in (
                ("NaN-tolerance", replace(prototype.Config(), solver_eps=np.nan), "invalid_input"),
                ("inverted-thresholds", replace(prototype.Config(), reject_threshold=0.9), "invalid_input"),
                ("future-version", replace(prototype.Config(), version=2), "unsupported"),
                ("unbounded-time", replace(prototype.Config(), time_limit_secs=0), "invalid_input"),
                ("excessive-time", replace(prototype.Config(), time_limit_secs=4), "invalid_input"),
                ("excessive-iterations", replace(prototype.Config(), max_iters=20_001), "unsupported"),
            ):
                self.reject_case(name, [eye], normal, config=config, expected=expected)
            for field in ("time_limit_secs", "solver_eps", "input_tolerance", "evidence_tolerance",
                          "conditioning_floor", "analysis_margin", "accept_threshold", "reject_threshold"):
                self.reject_case(f"huge-json-integer-{field}", [eye], normal,
                                 config=replace(prototype.Config(), **{field: 10**400}), expected="invalid_input")

        def test_report_binding_and_capability_separation(self):
            eye = np.eye(2)
            povm = [np.diag([0.9, 0.1]), np.diag([0.1, 0.9])]
            reset = pure_reset(np.array([1.0, 0.0]))
            first = self.analyze_case("binding-original-and-accept-label", reset, povm, (0.9, 0.9))
            self.assertEqual(first["threshold_label"], "numerical_accept_estimate")
            changed = self.analyze_case("binding-altered-outcome-and-reject-label", reset, povm, (0.1, 0.1), subset=(1,))
            self.assertEqual(changed["threshold_label"], "numerical_reject_estimate")
            altered = self.analyze_case("binding-altered-config", reset, povm, (0.9, 0.9), config=replace(prototype.Config(), analysis_margin=2e-5))
            changed_channel = self.analyze_case("binding-altered-channel", [eye], povm, (0.1, 0.9))
            self.assertEqual(len({value["model_config_runtime_sha256"] for value in (first, changed, altered, changed_channel)}), 4)
            repeat = prototype.analyze(reset, povm)
            self.assertEqual(first["model_config_runtime_sha256"], repeat["model_config_runtime_sha256"])
            self.assertIn("maximum_entropy_selection", first["unsupported"])
            self.assertIn("exact_amplitude_certification", first["unsupported"])
            # A slightly imperfect TP relation can be admitted within the
            # numeric tolerance, but never yields an exact CPTP/proof claim.
            self.analyze_case("numeric-admission-not-exact-TP", [np.sqrt(1 + 1e-12) * eye], [eye / 2, eye / 2], usable=False)

        def test_bounded_json_and_symbolic_amplitudes(self):
            with tempfile.TemporaryDirectory(prefix="ouro-quantum-fixture-") as directory:
                path = Path(directory) / "problem.json"
                path.write_bytes(b" " * (prototype.MAX_INPUT_BYTES + 1))
                with self.assertRaises(prototype.ResearchError):
                    prototype.load_input(path)
                _record({"name": "JSON-byte-cap", "rejected": "unsupported"})
                matrices = [[[[1, 0], [0, 0]], [[0, 0], [1, 0]]]]
                data = {"kraus": matrices, "povm": matrices, "selected_outcomes": [0], "config": asdict(prototype.Config())}
                path.write_text(json.dumps(data))
                operators, effects, selected, config = prototype.load_input(path)
                result = self.analyze_case("bounded-JSON-valid-identity", operators, effects, (1.0, 1.0), subset=selected, config=config)
                self.assertEqual(result["dimension"], 2)
                data["kraus"][0][0][0] = ["sqrt(1/2)", 0]
                path.write_text(json.dumps(data))
                with self.assertRaises(prototype.ResearchError):
                    prototype.load_input(path)
                _record({"name": "symbolic-exact-amplitudes", "rejected": "invalid_input"})
                path.write_text("{\"kraus\":NaN}")
                with self.assertRaises(prototype.ResearchError):
                    prototype.load_input(path)
                _record({"name": "nonfinite-JSON-constant", "rejected": "invalid_input"})
                for name, malformed in (
                    ("excessive-JSON-nesting", "[" * 2000 + "0" + "]" * 2000),
                    ("duplicate-JSON-fields", '{"kraus":[],"kraus":[],"povm":[]}'),
                ):
                    path.write_text(malformed)
                    with self.assertRaises(prototype.ResearchError):
                        prototype.load_input(path)
                    _record({"name": name, "rejected": "invalid_input"})

    started = time.monotonic()
    outcome = unittest.TextTestRunner(stream=sys.stderr, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Campaign))
    numeric = [case for case in REPORT if "result" in case]
    errors = [abs(case["result"][direction]["estimate"] - case["analytic_reference"][index]) for case in numeric if case["analytic_reference"] is not None and case["result"]["usable_extrema_estimates"] for index, direction in enumerate(("minimum", "maximum"))]
    summary = {
        "schema_version": 1,
        "claim": "finite_numeric_research_only",
        "analytic_reference_interpretation": (
            "ideal mathematical families, not exact arithmetic certification "
            "of serialized IEEE Kraus coefficients"
        ),
        "tests": outcome.testsRun,
        "failures": len(outcome.failures),
        "errors": len(outcome.errors),
        "passed": outcome.wasSuccessful(),
        "case_cap": MAX_CASES,
        "recorded_cases": len(REPORT),
        "numeric_cases": len(numeric),
        "rejected_cases": len(REPORT) - len(numeric),
        "numeric_sdp_solves": 2 * len(numeric) + 2,
        "numeric_sdp_solve_cap": 2 * MAX_CASES + 2,
        "maximum_usable_analytic_error": max(errors, default=None),
        "elapsed_seconds": time.monotonic() - started,
        "process_limits": {
            "wall_seconds": WALL_SECONDS,
            "cpu_seconds": 90,
            "address_space_bytes": 2 * 1024**3,
            "output_file_bytes": 16 * 1024**2,
            "open_files": 64,
        },
        "runtime": prototype.runtime_manifest(),
        "campaign_source_sha256": __import__("hashlib").sha256(Path(__file__).read_bytes()).hexdigest(),
        "cases": REPORT,
    }
    print(json.dumps(summary, sort_keys=True, allow_nan=False))
    return 0 if outcome.wasSuccessful() else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--report", type=Path)
    arguments = parser.parse_args()
    if arguments.worker:
        return run_worker()
    environment = os.environ.copy()
    for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        environment[variable] = "1"
    try:
        completed = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker"], cwd=ROOT, env=environment, capture_output=True, text=True, timeout=WALL_SECONDS, check=False)
    except subprocess.TimeoutExpired:
        print(json.dumps({"claim": "no_certificate", "status": "campaign_wall_cutoff", "wall_seconds": WALL_SECONDS}))
        return 2
    print(completed.stderr, file=sys.stderr, end="")
    print(completed.stdout, end="")
    if arguments.report is not None and completed.returncode == 0:
        # The requested local report is an experiment record, not a proof.
        json.loads(completed.stdout)
        arguments.report.parent.mkdir(parents=True, exist_ok=True)
        arguments.report.write_text(completed.stdout)
    return completed.returncode if completed.returncode >= 0 else 2


if __name__ == "__main__":
    sys.exit(main())
