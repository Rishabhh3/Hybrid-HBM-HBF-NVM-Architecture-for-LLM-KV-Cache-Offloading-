from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np

from TokenSim.config.psla_config import PSLAConfig
from TokenSim.errors import WorkloadValidationError
from TokenSim.moe.routing import normalize_expert_histogram
from TokenSim.utils import get_decode_lens, get_prefill_lens
from TokenSim.workload.user_request import UserRequest, UserRequestList


def load_workload(args: Any, model_config: PSLAConfig) -> UserRequestList:
    workload_type = args.workload_type
    if args.dataset_path and workload_type == "synthetic":
        workload_type = "json_pairs"

    if workload_type == "synthetic":
        user_requests = load_synthetic_workload(args, model_config)
    elif workload_type == "json_pairs":
        if not args.dataset_path:
            raise WorkloadValidationError(
                "--dataset_path is required for json_pairs workloads"
            )
        user_requests = load_json_pairs_workload(
            dataset_path=args.dataset_path,
            request_count=args.request_count,
            dataset_skip_count=args.dataset_skip_count,
            num_experts=model_config.moe_config.num_experts,
        )
    elif workload_type == "qwen_jsonl":
        if not args.dataset_path:
            raise WorkloadValidationError(
                "--dataset_path is required for qwen_jsonl workloads"
            )
        user_requests = load_qwen_jsonl_workload(
            dataset_path=args.dataset_path,
            request_count=args.request_count,
            dataset_skip_count=args.dataset_skip_count,
            num_experts=model_config.moe_config.num_experts,
        )
    else:
        raise WorkloadValidationError(f"unknown workload_type {workload_type!r}")

    jitter_ms = getattr(args, "arrival_jitter_ms", 0.0)
    jitter_seed = getattr(args, "jitter_seed", 0)
    jitter_mode = getattr(args, "arrival_jitter_mode", "request")
    if jitter_mode not in ARRIVAL_JITTER_MODES:
        raise WorkloadValidationError(
            f"unknown arrival_jitter_mode {jitter_mode!r}; expected one of "
            + f"{sorted(ARRIVAL_JITTER_MODES)}"
        )
    if jitter_mode == "request":
        user_requests = apply_arrival_jitter(
            user_requests, jitter_ms=jitter_ms, jitter_seed=jitter_seed
        )
    user_requests = apply_trace_timestamp_scaling(
        user_requests,
        trace_timestamp_scale=getattr(args, "trace_timestamp_scale", None),
        trace_target_qps=getattr(args, "trace_target_qps", None),
    )
    if jitter_mode == "burst":
        user_requests = apply_burst_jitter(
            user_requests, jitter_ms=jitter_ms, jitter_seed=jitter_seed
        )
    return user_requests


def load_synthetic_workload(args: Any, model_config: PSLAConfig) -> UserRequestList:
    random.seed(args.random_seed)
    prefill_lens = get_prefill_lens(
        len_mean=model_config.prefill_mean_len,
        len_range=model_config.prefill_range_len,
        request_count=args.request_count,
    )
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    decode_lens = get_decode_lens(
        distribution=model_config.decode_len_distribution,
        len_mean=model_config.decode_mean_len,
        len_range=model_config.decode_range_len,
        request_count=args.request_count,
    )
    return UserRequestList(
        UserRequest(
            request_id=request_id,
            prefill_len=prefill_len,
            decode_len=decode_len,
        )
        for request_id, (prefill_len, decode_len) in enumerate(
            zip(prefill_lens, decode_lens)
        )
    )


def load_json_pairs_workload(
    dataset_path: str,
    request_count: int | None,
    dataset_skip_count: int = 0,
    num_experts: int = 0,
) -> UserRequestList:
    path = _validate_dataset_path(dataset_path)
    with path.open("r") as file:
        try:
            data = json.load(file)
        except json.JSONDecodeError as exc:
            raise WorkloadValidationError(
                f"invalid JSON in json_pairs dataset {path}: {exc}"
            ) from exc

    if not isinstance(data, list):
        raise WorkloadValidationError(f"json_pairs workload {path} must be a JSON list")

    data = _repeat_and_slice(data, request_count, dataset_skip_count)
    user_requests = UserRequestList()
    for request_id, item in enumerate(data):
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            raise WorkloadValidationError(
                f"json_pairs record {request_id} in {path} must be "
                + "[prefill_len, decode_len]"
            )
        extra = item[2] if len(item) > 2 else {}
        if extra is None:
            extra = {}
        if not isinstance(extra, dict):
            raise WorkloadValidationError(
                f"json_pairs record {request_id} in {path} third field must be an object"
            )
        user_requests.append(
            UserRequest(
                request_id=request_id,
                prefill_len=item[0],
                decode_len=item[1],
                arrival_time=extra.get("arrival_time"),
                inter_arrival_time=extra.get("inter_arrival_time"),
                chat_id=_optional_str(extra.get("chat_id")),
                parent_chat_id=_optional_str(extra.get("parent_chat_id")),
                turn=extra.get("turn"),
                hash_ids=extra.get("hash_ids"),
                output_hash_ids=extra.get("output_hash_ids"),
                expert_histogram=_histogram_or_none(
                    extra.get("expert_histogram"),
                    num_experts=num_experts,
                    request_id=request_id,
                ),
                prefill_expert_histogram=_histogram_or_none(
                    extra.get("prefill_expert_histogram"),
                    num_experts=num_experts,
                    request_id=request_id,
                ),
                decode_expert_histogram=_histogram_or_none(
                    extra.get("decode_expert_histogram"),
                    num_experts=num_experts,
                    request_id=request_id,
                ),
                cache_salt=_optional_str(extra.get("cache_salt")),
                reuse_group=_optional_str(extra.get("reuse_group")),
            )
        )

    _print_dataset_summary(path, user_requests)
    return user_requests


def load_qwen_jsonl_workload(
    dataset_path: str,
    request_count: int | None,
    dataset_skip_count: int = 0,
    num_experts: int = 0,
) -> UserRequestList:
    path = _validate_dataset_path(dataset_path)
    rows: list[dict[str, Any]] = []
    with path.open("r") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise WorkloadValidationError(
                    f"invalid JSONL record at {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise WorkloadValidationError(
                    f"qwen_jsonl record at {path}:{line_number} must be a JSON object"
                )
            rows.append(row)

    rows = _repeat_and_slice(rows, request_count, dataset_skip_count)
    timestamps = [row.get("timestamp") for row in rows]
    first_timestamp = None
    if any(timestamp is not None for timestamp in timestamps):
        first_timestamp = min(float(timestamp) for timestamp in timestamps if timestamp is not None)

    user_requests = UserRequestList()
    seen_request_ids: set[int] = set()
    for request_id, row in enumerate(rows):
        timestamp = row.get("timestamp")
        arrival_time = None
        if timestamp is not None and first_timestamp is not None:
            arrival_time = float(timestamp) - first_timestamp
        trace_request_id = row.get("request_id")
        unique_request_id = request_id
        if trace_request_id is not None:
            unique_request_id = int(trace_request_id)
            if unique_request_id in seen_request_ids:
                unique_request_id = request_id
        seen_request_ids.add(unique_request_id)
        try:
            prefill_len = row["input_length"]
            decode_len = row["output_length"]
        except KeyError as exc:
            raise WorkloadValidationError(
                f"qwen_jsonl record {request_id} in {path} is missing {exc.args[0]!r}"
            ) from exc
        user_requests.append(
            UserRequest(
                request_id=unique_request_id,
                prefill_len=prefill_len,
                decode_len=decode_len,
                arrival_time=arrival_time,
                chat_id=_optional_str(row.get("chat_id")),
                parent_chat_id=_optional_str(row.get("parent_chat_id")),
                turn=row.get("turn"),
                hash_ids=row.get("hash_ids"),
                output_hash_ids=row.get("output_hash_ids"),
                expert_histogram=_histogram_or_none(
                    row.get("expert_histogram"),
                    num_experts=num_experts,
                    request_id=unique_request_id,
                ),
                prefill_expert_histogram=_histogram_or_none(
                    row.get("prefill_expert_histogram"),
                    num_experts=num_experts,
                    request_id=unique_request_id,
                ),
                decode_expert_histogram=_histogram_or_none(
                    row.get("decode_expert_histogram"),
                    num_experts=num_experts,
                    request_id=unique_request_id,
                ),
                cache_salt=_optional_str(row.get("cache_salt")),
                reuse_group=_optional_str(row.get("reuse_group")),
            )
        )

    _print_dataset_summary(path, user_requests)
    return user_requests


def _validate_dataset_path(dataset_path: str) -> Path:
    path = Path(dataset_path)
    if not path.exists() or not path.is_file():
        raise WorkloadValidationError(f"dataset file not found: {dataset_path}")
    return path


ARRIVAL_JITTER_MODES = {"request", "burst"}


def _validated_jitter_ms(jitter_ms: float) -> float:
    jitter_ms = float(jitter_ms or 0.0)
    if not math.isfinite(jitter_ms) or jitter_ms < 0:
        raise WorkloadValidationError("arrival_jitter_ms must be finite and non-negative")
    return jitter_ms


def burst_gaps_s(user_requests: UserRequestList) -> list[float]:
    """Gaps between consecutive distinct arrival times, in the list's clock.

    A burst is the set of requests sharing one arrival time exactly.
    """
    times = sorted({r.arrival_time for r in user_requests if r.arrival_time is not None})
    return [b - a for a, b in zip(times, times[1:])]


def apply_burst_jitter(
    user_requests: UserRequestList,
    *,
    jitter_ms: float = 0.0,
    jitter_seed: int = 0,
) -> UserRequestList:
    """Offset each burst -- all requests sharing an arrival time -- by one draw.

    Applied on the replay clock, after timestamp scaling. One uniform draw in
    [-j, +j] ms per burst, from a private ``random.Random(jitter_seed)``, in
    order of first appearance in the file; times below 0 are clamped to 0.
    Requests within a burst keep identical times, so the stable arrival sort
    keeps their file order. Afterwards the arrival order must equal file order
    exactly, otherwise WorkloadValidationError: pick j below half the minimum
    inter-burst gap (``burst_gaps_s``) and this cannot fire on a trace whose
    file order is already time order. ``jitter_ms == 0`` returns the list
    untouched and draws nothing.
    """
    jitter_ms = _validated_jitter_ms(jitter_ms)
    if jitter_ms == 0.0:
        return user_requests
    rng = random.Random(jitter_seed)
    jitter_s = jitter_ms / 1000.0
    offsets: dict[float, float] = {}
    for request in user_requests:
        if request.arrival_time is None:
            continue
        if request.arrival_time not in offsets:
            offsets[request.arrival_time] = rng.uniform(-jitter_s, jitter_s)
    for request in user_requests:
        if request.arrival_time is not None:
            request.arrival_time = max(0.0, request.arrival_time + offsets[request.arrival_time])
    timed = [r for r in user_requests if r.arrival_time is not None]
    replay_order = sorted(range(len(timed)), key=lambda i: timed[i].arrival_time)
    if replay_order != list(range(len(timed))):
        raise WorkloadValidationError(
            "burst jitter changed arrival order: arrival order no longer equals "
            + f"file order (arrival_jitter_ms={jitter_ms}); reduce the jitter "
            + "below half the minimum inter-burst gap"
        )
    return user_requests


def apply_arrival_jitter(
    user_requests: UserRequestList,
    *,
    jitter_ms: float = 0.0,
    jitter_seed: int = 0,
) -> UserRequestList:
    """arrival_jitter_mode "request": offset each arrival by its own draw.

    Uniform in [-j, +j] ms per timestamped request. A workload variant, not
    noise: on traces whose timestamps are coarser than j it only permutes
    requests that share a timestamp (PHASE 14 STEP 2b).

    Applied on the trace clock, before trace_timestamp_scale/trace_target_qps,
    so the simulated offset is j times the scaling factor. Offsets come from a
    private ``random.Random(jitter_seed)`` -- never the global generators -- and
    are drawn in request order, one per timestamped request; a jittered time
    below 0 is clamped to 0. ``jitter_ms == 0``
    (the default) returns the list untouched and draws nothing.
    """
    jitter_ms = _validated_jitter_ms(jitter_ms)
    if jitter_ms == 0.0:
        return user_requests
    rng = random.Random(jitter_seed)
    jitter_s = jitter_ms / 1000.0
    for request in user_requests:
        if request.arrival_time is not None:
            # Clamped at 0 so a request at the trace origin cannot be moved
            # before it; only arrivals within j of t = 0 are affected.
            request.arrival_time = max(
                0.0, request.arrival_time + rng.uniform(-jitter_s, jitter_s)
            )
    return user_requests


def apply_trace_timestamp_scaling(
    user_requests: UserRequestList,
    *,
    trace_timestamp_scale: float | None = None,
    trace_target_qps: float | None = None,
) -> UserRequestList:
    if trace_timestamp_scale is None and trace_target_qps is None:
        return user_requests
    if trace_timestamp_scale is not None and trace_target_qps is not None:
        raise WorkloadValidationError(
            "trace_timestamp_scale and trace_target_qps are mutually exclusive"
        )

    target_qps = None
    if trace_timestamp_scale is not None:
        scale = _positive_finite_float(
            trace_timestamp_scale,
            "trace_timestamp_scale",
        )
    else:
        target_qps = _positive_finite_float(trace_target_qps, "trace_target_qps")

    arrival_times = [
        request.arrival_time
        for request in user_requests
        if request.arrival_time is not None
    ]
    if not arrival_times:
        raise WorkloadValidationError(
            "trace timestamp scaling requires dataset records with arrival_time or timestamp"
        )

    if target_qps is not None:
        if len(arrival_times) < 2:
            raise WorkloadValidationError(
                "trace_target_qps requires at least two timestamped arrivals"
            )
        last_ts = max(arrival_times)
        if last_ts <= 0:
            raise WorkloadValidationError(
                "trace_target_qps requires a positive maximum arrival timestamp"
            )
        original_trace_qps = (len(arrival_times) - 1) / last_ts
        scale = original_trace_qps / target_qps

    for request in user_requests:
        if request.arrival_time is not None:
            request.arrival_time *= scale
    return user_requests


def _positive_finite_float(value: Any, field_name: str) -> float:
    numeric_value = float(value)
    if not math.isfinite(numeric_value) or numeric_value <= 0:
        raise WorkloadValidationError(
            f"{field_name} must be a positive finite number, got {value!r}"
        )
    return numeric_value


def _repeat_and_slice(
    data: list[Any], request_count: int | None, dataset_skip_count: int
) -> list[Any]:
    if dataset_skip_count < 0:
        raise WorkloadValidationError("dataset_skip_count must be non-negative")
    data = data[dataset_skip_count:]
    if not data:
        raise WorkloadValidationError(
            "dataset is empty after applying dataset_skip_count"
        )
    if request_count is not None and request_count > len(data):
        data = data * (request_count // len(data) + 1)
    if request_count is not None:
        data = data[:request_count]
    return data


def _print_dataset_summary(path: Path, user_requests: UserRequestList) -> None:
    prefill_lens = user_requests.prefill_lens()
    decode_lens = user_requests.decode_lens()
    print(
        f'[workload] Read {len(user_requests)} reqs from dataset "{path}", '
        + f"prefill average {sum(prefill_lens) / len(prefill_lens)}, "
        + f"decode average {sum(decode_lens) / len(decode_lens)}"
    )


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _histogram_or_none(
    value: Any,
    *,
    num_experts: int,
    request_id: int,
) -> dict[int, int] | None:
    if value is None:
        return None
    if num_experts <= 0:
        raise WorkloadValidationError(
            f"expert histogram for request {request_id} requires a MoE model config"
        )
    return normalize_expert_histogram(
        value,
        num_experts=num_experts,
        request_id=request_id,
    )
