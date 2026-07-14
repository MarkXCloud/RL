# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import sys
import threading
import types

from nemo_rl.models.generation.vllm import vllm_generation
from nemo_rl.models.generation.vllm import vllm_worker_async
from nemo_rl.models.generation.vllm.vllm_generation import VllmGeneration
from nemo_rl.models.generation.vllm.vllm_worker_async import (
    VllmAsyncGenerationWorkerImpl,
    _compute_vllm_metric_deltas,
    _read_vllm_cumulative_metrics,
)

_INFERENCE_TIME = "vllm:request_inference_time_seconds"
_HISTOGRAM_NAMES = {
    "vllm:time_to_first_token_seconds",
    "vllm:inter_token_latency_seconds",
    "vllm:request_time_per_output_token_seconds",
    "vllm:e2e_request_latency_seconds",
    "vllm:request_queue_time_seconds",
    _INFERENCE_TIME,
    "vllm:request_prefill_time_seconds",
    "vllm:request_decode_time_seconds",
}


def _series(metric_name: str, engine: str) -> tuple[str, tuple[tuple[str, str], ...]]:
    return metric_name, (("engine", engine),)


def test_vllm_histogram_deltas_use_baseline_and_preserve_labels() -> None:
    baseline = {
        "histograms": {
            _series(_INFERENCE_TIME, "0"): {"sum": 20.0, "count": 10},
            _series(_INFERENCE_TIME, "1"): {"sum": 7.0, "count": 3},
        }
    }
    current = {
        "histograms": {
            _series(_INFERENCE_TIME, "0"): {"sum": 32.0, "count": 14},
            _series(_INFERENCE_TIME, "1"): {"sum": 10.0, "count": 5},
        }
    }

    assert _compute_vllm_metric_deltas(baseline, current) == {
        _INFERENCE_TIME: {"sum": 15.0, "count": 6}
    }


def test_vllm_histogram_delta_invalidates_after_counter_reset(caplog) -> None:
    baseline = {
        "histograms": {_series(_INFERENCE_TIME, "0"): {"sum": 20.0, "count": 10}}
    }
    current = {"histograms": {_series(_INFERENCE_TIME, "0"): {"sum": 3.0, "count": 2}}}

    with caplog.at_level(logging.WARNING):
        result = _compute_vllm_metric_deltas(baseline, current)

    assert result is None
    assert "Skipping incomplete vLLM histogram window" in caplog.text


def test_read_vllm_cumulative_metrics_uses_native_histograms(monkeypatch) -> None:
    class FakeHistogram:
        def __init__(
            self,
            *,
            name: str,
            labels: dict[str, str],
            count: int,
            sum: float,
            buckets: dict[str, int],
        ) -> None:
            self.name = name
            self.labels = labels
            self.count = count
            self.sum = sum
            self.buckets = buckets

    for module_name in ("vllm", "vllm.v1", "vllm.v1.metrics"):
        monkeypatch.setitem(sys.modules, module_name, types.ModuleType(module_name))
    reader = types.ModuleType("vllm.v1.metrics.reader")
    reader.Histogram = FakeHistogram
    reader.get_metrics_snapshot = lambda: [
        *[
            FakeHistogram(
                name=name,
                labels={"model_name": "m", "engine": "0"},
                count=4,
                sum=9.0,
                buckets={"+Inf": 4},
            )
            for name in _HISTOGRAM_NAMES
        ],
        FakeHistogram(
            name="vllm:unrelated_histogram",
            labels={},
            count=100,
            sum=200.0,
            buckets={"+Inf": 100},
        ),
    ]
    monkeypatch.setitem(sys.modules, "vllm.v1.metrics.reader", reader)

    result = _read_vllm_cumulative_metrics()

    assert {series_key[0] for series_key in result["histograms"]} == _HISTOGRAM_NAMES
    assert all(
        series_key[1] == (("engine", "0"), ("model_name", "m"))
        for series_key in result["histograms"]
    )
    assert all(
        total == {"sum": 9.0, "count": 4} for total in result["histograms"].values()
    )


def test_vllm_worker_get_is_idempotent_until_atomic_clear(monkeypatch) -> None:
    first_baseline = {
        "histograms": {_series(_INFERENCE_TIME, "0"): {"sum": 10.0, "count": 5}}
    }
    first_current = {
        "histograms": {_series(_INFERENCE_TIME, "0"): {"sum": 16.0, "count": 7}}
    }
    second_current = {
        "histograms": {_series(_INFERENCE_TIME, "0"): {"sum": 19.0, "count": 8}}
    }
    snapshots = iter([first_current, first_current, first_current, second_current])
    monkeypatch.setattr(
        vllm_worker_async,
        "_try_read_vllm_cumulative_metrics",
        lambda: next(snapshots),
    )

    worker = VllmAsyncGenerationWorkerImpl.__new__(VllmAsyncGenerationWorkerImpl)
    worker.cfg = {"vllm_cfg": {"enable_vllm_metrics_logger": True}}
    worker._vllm_metrics_lock = threading.Lock()
    worker._vllm_cumulative_metrics_baseline = first_baseline
    worker.inflight_batch_sizes = [1]
    worker.num_pending_samples = []
    worker.kv_cache_usage_perc = []
    worker.generation_tokens = []

    expected_first = {_INFERENCE_TIME: {"sum": 6.0, "count": 2}}
    assert worker.get_vllm_logger_metrics()["histogram_deltas"] == expected_first
    assert worker.get_vllm_logger_metrics()["histogram_deltas"] == expected_first

    assert worker.get_and_clear_vllm_logger_metrics()["histogram_deltas"] == (
        expected_first
    )
    assert worker.inflight_batch_sizes == []
    assert worker.get_vllm_logger_metrics()["histogram_deltas"] == {
        _INFERENCE_TIME: {"sum": 3.0, "count": 1}
    }


def test_vllm_worker_invalidates_baseline_when_boundary_snapshot_fails(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        vllm_worker_async,
        "_try_read_vllm_cumulative_metrics",
        lambda: None,
    )
    worker = VllmAsyncGenerationWorkerImpl.__new__(VllmAsyncGenerationWorkerImpl)
    worker.cfg = {"vllm_cfg": {"enable_vllm_metrics_logger": True}}
    worker._vllm_metrics_lock = threading.Lock()
    worker._vllm_cumulative_metrics_baseline = {
        "histograms": {_series(_INFERENCE_TIME, "0"): {"sum": 10.0, "count": 5}}
    }
    worker.inflight_batch_sizes = []
    worker.num_pending_samples = []
    worker.kv_cache_usage_perc = []
    worker.generation_tokens = []

    assert "histogram_deltas" not in worker.get_and_clear_vllm_logger_metrics()
    assert worker._vllm_cumulative_metrics_baseline is None


def test_vllm_generation_collects_histogram_deltas_per_dp(monkeypatch) -> None:
    worker_methods: list[str] = []

    class FakeWorkerGroup:
        dp_size = 2

        @staticmethod
        def get_dp_leader_worker_idx(dp_idx: int) -> int:
            return dp_idx

        @staticmethod
        def run_single_worker_single_data(method_name: str, worker_idx: int) -> int:
            worker_methods.append(method_name)
            return worker_idx

    generation = VllmGeneration.__new__(VllmGeneration)
    generation.cfg = {
        "vllm_cfg": {
            "enable_vllm_metrics_logger": True,
            "async_engine": True,
        }
    }
    generation.worker_group = FakeWorkerGroup()
    monkeypatch.setattr(
        vllm_generation.ray,
        "get",
        lambda _: [
            {
                "inflight_batch_sizes": [1],
                "histogram_deltas": {_INFERENCE_TIME: {"sum": 8.0, "count": 2}},
            },
            {
                "inflight_batch_sizes": [2],
                "histogram_deltas": {_INFERENCE_TIME: {"sum": 6.0, "count": 6}},
            },
        ],
    )

    metrics = generation.get_vllm_logger_metrics()

    assert metrics["inflight_batch_sizes"] == {0: [1], 1: [2]}
    assert metrics["histogram_deltas"] == {
        _INFERENCE_TIME: {
            0: {"sum": 8.0, "count": 2},
            1: {"sum": 6.0, "count": 6},
        }
    }

    generation.get_and_clear_vllm_logger_metrics()
    assert worker_methods == [
        "get_vllm_logger_metrics",
        "get_vllm_logger_metrics",
        "get_and_clear_vllm_logger_metrics",
        "get_and_clear_vllm_logger_metrics",
    ]

    monkeypatch.setattr(
        vllm_generation.ray,
        "get",
        lambda _: [
            {"histogram_deltas": {}},
            {"inflight_batch_sizes": [2]},
        ],
    )
    assert "histogram_deltas" not in generation.get_vllm_logger_metrics()
