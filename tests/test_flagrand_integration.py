from __future__ import annotations

import sys
import types
import unittest
import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


try:
    import torch  # type: ignore
except ModuleNotFoundError:
    torch = types.ModuleType("torch")
    torch.int32 = object()
    torch.int64 = object()
    torch.float32 = object()
    torch.float64 = object()
    torch.dtype = object
    torch.device = object
    torch.Tensor = object
    sys.modules["torch"] = torch


from contract_benchmark import flagrand_adapter  # noqa: E402
from contract_benchmark.flagrand_provenance import (  # noqa: E402
    REPO_ROOT as PROVENANCE_REPO_ROOT,
    flagrand_tree_sha256,
    load_flagrand_vendor_manifest,
)


HOSTAPI_SPEC = importlib.util.spec_from_file_location(
    "hostapi_only_benchmark_for_test",
    REPO_ROOT / "scripts" / "hostapi_only_benchmark.py",
)
assert HOSTAPI_SPEC is not None and HOSTAPI_SPEC.loader is not None
hostapi_only_benchmark = importlib.util.module_from_spec(HOSTAPI_SPEC)
sys.modules[HOSTAPI_SPEC.name] = hostapi_only_benchmark
HOSTAPI_SPEC.loader.exec_module(hostapi_only_benchmark)


class _FakeCurand:
    CURAND_RNG_PSEUDO_PHILOX4_32_10 = "pseudo_philox"
    CURAND_RNG_PSEUDO_XORWOW = "pseudo_xorwow"
    CURAND_RNG_PSEUDO_MRG32K3A = "pseudo_mrg32k3a"
    CURAND_RNG_PSEUDO_MTGP32 = "pseudo_mtgp32"
    CURAND_RNG_PSEUDO_MT19937 = "pseudo_mt19937"
    CURAND_RNG_QUASI_SOBOL32 = "quasi_sobol32"
    CURAND_RNG_QUASI_SCRAMBLED_SOBOL32 = "quasi_scrambled_sobol32"
    CURAND_RNG_QUASI_SOBOL64 = "quasi_sobol64"
    CURAND_RNG_QUASI_SCRAMBLED_SOBOL64 = "quasi_scrambled_sobol64"

    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []

    def create_generator(self, rng_type: str, **kwargs: object) -> object:
        self.calls.append(("create_generator", rng_type, kwargs))
        return {"rng_type": rng_type, **kwargs}

    def __getattr__(self, name: str):
        if not name.startswith("generate"):
            raise AttributeError(name)

        def generate(*args: object, **kwargs: object) -> tuple[str, tuple[object, ...], dict[str, object]]:
            self.calls.append((name, args, kwargs))
            return name, args, kwargs

        return generate


class _Output:
    def __init__(self, dtype: object) -> None:
        self.dtype = dtype


class FlagRandIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.curand = _FakeCurand()
        flagrand_adapter._FLAGRAND_CACHE = {"curand": self.curand}

    def tearDown(self) -> None:
        flagrand_adapter._FLAGRAND_CACHE = None

    def test_vendor_manifest_matches_checked_in_tree(self) -> None:
        manifest = load_flagrand_vendor_manifest()
        source_root = PROVENANCE_REPO_ROOT / str(manifest["source_path"])
        file_count, digest = flagrand_tree_sha256(source_root)
        self.assertEqual(file_count, manifest["tree_file_count"])
        self.assertEqual(digest, manifest["tree_sha256"])
        self.assertEqual(
            manifest["commit"],
            "bbb660e5cfda4530033c6cfb6115de58c1278654",
        )

    def test_generator_creation_uses_explicit_refactored_facade_mapping(self) -> None:
        generator = flagrand_adapter.make_flagrand_generator(
            "philox4x32_10",
            seed=17,
            offset=8,
            dimensions=3,
        )
        self.assertEqual(generator["rng_type"], "pseudo_philox")
        self.assertEqual(generator["seed"], 17)
        self.assertEqual(generator["offset"], 8)
        self.assertEqual(generator["dimensions"], 3)

    def test_distribution_dispatch_uses_dtype_specific_facade_calls(self) -> None:
        generator = object()
        cases = (
            (flagrand_adapter.flagrand_generate_raw, torch.int32, "generate", {}),
            (flagrand_adapter.flagrand_generate_raw, torch.int64, "generate_long_long", {}),
            (flagrand_adapter.flagrand_generate_uniform, torch.float32, "generate_uniform", {}),
            (flagrand_adapter.flagrand_generate_uniform, torch.float64, "generate_uniform_double", {}),
            (flagrand_adapter.flagrand_generate_normal, torch.float32, "generate_normal", {"mean": 1.0, "stddev": 2.0}),
            (flagrand_adapter.flagrand_generate_normal, torch.float64, "generate_normal_double", {"mean": 1.0, "stddev": 2.0}),
            (flagrand_adapter.flagrand_generate_lognormal, torch.float32, "generate_lognormal", {"mean": 1.0, "stddev": 2.0}),
            (flagrand_adapter.flagrand_generate_lognormal, torch.float64, "generate_lognormal_double", {"mean": 1.0, "stddev": 2.0}),
        )
        for function, dtype, expected, kwargs in cases:
            with self.subTest(expected=expected):
                result = function(_Output(dtype), generator, **kwargs)
                self.assertEqual(result[0], expected)

        result = flagrand_adapter.flagrand_generate_poisson(
            _Output(torch.int32),
            generator,
            lambda_val=4.0,
        )
        self.assertEqual(result[0], "generate_poisson")
        self.assertEqual(result[2]["lambda_val"], 4.0)

    def test_hostapi_matrix_covers_refactored_philox_f64(self) -> None:
        philox = hostapi_only_benchmark.GENERATOR_INFOS["philox4x32_10"]
        distributions = hostapi_only_benchmark._expand_distributions(
            philox,
            ["raw", "uniform", "normal", "lognormal", "poisson"],
        )
        self.assertIn("raw32", distributions)
        self.assertNotIn("raw64", distributions)
        for name in (
            "uniform_f32",
            "uniform_f64",
            "normal_f32",
            "normal_f64",
            "lognormal_f32",
            "lognormal_f64",
            "poisson_u32",
        ):
            self.assertIn(name, distributions)

    def test_rtx4060_gate_matrix_exactly_matches_h20_matrix(self) -> None:
        args = types.SimpleNamespace(
            generators="all",
            distributions="all",
            sizes="profile",
            poisson_lambdas="profile",
            qrng_dimensions=1,
            offset=0,
        )
        rtx_cases = hostapi_only_benchmark.build_cases(
            args,
            hostapi_only_benchmark.HOSTAPI_PROFILES["rtx4060_gate"],
        )
        h20_cases = hostapi_only_benchmark.build_cases(
            args,
            hostapi_only_benchmark.HOSTAPI_PROFILES["h20"],
        )
        self.assertEqual(len(rtx_cases), 588)
        self.assertEqual(
            [case.to_record() for case in rtx_cases],
            [case.to_record() for case in h20_cases],
        )

    def test_batch_call_count_caps_work_for_large_cases(self) -> None:
        profile = hostapi_only_benchmark.HOSTAPI_PROFILES["h20"]
        args = types.SimpleNamespace(
            batch_calls=None,
            batch_target_items=None,
            batch_repeats=None,
        )
        small = types.SimpleNamespace(n=4096)
        large = types.SimpleNamespace(n=8388608)
        self.assertEqual(hostapi_only_benchmark._batch_calls(small, args, profile), 32)
        self.assertEqual(hostapi_only_benchmark._batch_calls(large, args, profile), 2)

    def test_hostapi_speedups_separate_dispatch_batch_and_residual_views(self) -> None:
        records = [
            {
                "case_id": "case",
                "backend": "curand_host_api",
                "status": "ok",
                "median_gpu_us": 10.0,
                "median_wall_sync_us": 14.0,
                "median_cpu_enqueue_us": 6.0,
                "diagnostic_median_gpu_minus_enqueue_us": 4.0,
                "batch_median_gpu_us_per_call": 8.0,
                "batch_median_wall_sync_us_per_call": 9.0,
                "batch_median_cpu_enqueue_us_per_call": 5.0,
            },
            {
                "case_id": "case",
                "backend": "flagrand_public_api",
                "status": "ok",
                "median_gpu_us": 30.0,
                "median_wall_sync_us": 35.0,
                "median_cpu_enqueue_us": 27.0,
                "diagnostic_median_gpu_minus_enqueue_us": 3.0,
                "batch_median_gpu_us_per_call": 12.0,
                "batch_median_wall_sync_us_per_call": 13.0,
                "batch_median_cpu_enqueue_us_per_call": 10.0,
            },
        ]
        hostapi_only_benchmark._add_speedups(records)
        candidate = records[1]
        self.assertAlmostEqual(candidate["speedup_gpu_vs_curand_host"], 1.0 / 3.0)
        self.assertAlmostEqual(candidate["speedup_cpu_enqueue_vs_curand_host"], 6.0 / 27.0)
        self.assertAlmostEqual(candidate["speedup_batch_gpu_vs_curand_host"], 2.0 / 3.0)
        self.assertAlmostEqual(candidate["diagnostic_residual_speedup_vs_curand_host"], 4.0 / 3.0)

    def test_hostapi_matrix_uses_only_f64_distributions_for_sobol64(self) -> None:
        sobol64 = hostapi_only_benchmark.GENERATOR_INFOS["sobol64"]
        distributions = hostapi_only_benchmark._expand_distributions(
            sobol64,
            ["raw", "uniform", "normal", "lognormal", "poisson"],
        )
        self.assertEqual(
            distributions,
            ["raw64", "uniform_f64", "normal_f64", "lognormal_f64"],
        )

    def test_refactored_sobol_metadata_reports_direct_table_path(self) -> None:
        case = hostapi_only_benchmark.HostApiCase(
            case_index=0,
            case_id="sobol32:uniform_f32:n=4096:dim=4",
            generator="sobol32",
            distribution="uniform_f32",
            n=4096,
            dtype_name="float32",
            parameters={},
            dimensions=4,
            notes=[],
        )
        path_kind = hostapi_only_benchmark._flagrand_path_kind(case)
        self.assertEqual(path_kind, "direct_quasi_table_distribution")
        self.assertEqual(
            hostapi_only_benchmark._flagrand_launch_estimate(case, path_kind),
            1,
        )

    def test_hostapi_non_ok_records_fail_run_health(self) -> None:
        records = [
            {
                "case_id": "philox4x32_10:uniform_f32:n=4096",
                "backend": "flagrand_public_api",
                "status": "error",
            }
        ]
        summary = hostapi_only_benchmark.summarize_records(
            records,
            [],
            {"profile": "local_smoke"},
            failures=[],
        )
        self.assertEqual(summary["run_health"]["status"], "needs_attention")
        self.assertEqual(summary["run_health"]["error_record_count"], 1)

    def test_qrng_case_size_satisfies_dimensions_and_distribution_alignment(self) -> None:
        n = hostapi_only_benchmark._adjust_n(
            1024,
            "sobol32",
            "normal_f32",
            {},
            dimensions=5,
        )
        self.assertEqual(n % 5, 0)
        self.assertEqual(n % 2, 0)
        with self.assertRaises(ValueError):
            hostapi_only_benchmark._adjust_n(
                1024,
                "sobol32",
                "uniform_f32",
                {},
                dimensions=0,
            )

    def test_hostapi_rejects_invalid_qrng_dimensions_and_incomparable_offsets(self) -> None:
        values = {
            "generators": "sobol32",
            "distributions": "raw",
            "sizes": "1024",
            "poisson_lambdas": "1",
            "qrng_dimensions": hostapi_only_benchmark.MAX_QRNG_DIMENSIONS + 1,
            "offset": 0,
        }
        with self.assertRaises(SystemExit):
            hostapi_only_benchmark.build_cases(
                types.SimpleNamespace(**values),
                hostapi_only_benchmark.HOSTAPI_PROFILES["local_smoke"],
            )

        values.update(
            {
                "generators": "mtgp32",
                "qrng_dimensions": 1,
                "offset": 1,
            }
        )
        with self.assertRaises(SystemExit):
            hostapi_only_benchmark.build_cases(
                types.SimpleNamespace(**values),
                hostapi_only_benchmark.HOSTAPI_PROFILES["local_smoke"],
            )


if __name__ == "__main__":
    unittest.main()
