from __future__ import annotations

from typing import Any

from contract_benchmark.profiles import BenchmarkContext
from contract_benchmark.spec import TaskSpec
from contract_benchmark.tasks.common import adjust_n
from contract_benchmark.tasks.distribution_diagnostics.cases import GENERATOR, diagnostic_cases, diagnostic_sizes
from contract_benchmark.tasks.distribution_diagnostics.records import case_records


def run_distribution_decomposition(ctx: BenchmarkContext, spec: TaskSpec) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for case in diagnostic_cases(ctx):
        for n0 in diagnostic_sizes(ctx):
            n = adjust_n(n0, GENERATOR, case.distribution)
            records.extend(case_records(ctx, spec, case, n))
    return records
