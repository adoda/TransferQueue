# Copyright 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2025 The TransferQueue Team
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

"""Tests for storage-unit request retry and the timeout diagnosis that classifies a failure."""

import logging
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest
import torch
import zmq
from tensordict import TensorDict

from transfer_queue.metadata import BatchMeta
from transfer_queue.storage.managers import simple_storage_manager as ssm
from transfer_queue.storage.managers.simple_storage_manager import (
    AsyncSimpleStorageManager,
    StorageUnitTimeout,
)
from transfer_queue.utils import common
from transfer_queue.utils.enum_utils import Role
from transfer_queue.utils.zmq_utils import ZMQServerInfo


def _manager(**units: ZMQServerInfo) -> AsyncSimpleStorageManager:
    """Build a manager carrying only the state the retry and diagnosis helpers read."""
    manager = AsyncSimpleStorageManager.__new__(AsyncSimpleStorageManager)
    manager.storage_manager_id = "TQ_STORAGE_test"
    manager.storage_unit_infos = dict(units)
    manager.close = lambda: None  # __init__ is skipped, so there is no socket or thread to close
    return manager


def _server_info(unit_id: str, ip: str, port: int) -> ZMQServerInfo:
    return ZMQServerInfo(role=Role.STORAGE, id=unit_id, ip=ip, ports={"put_get_socket": port})


def _single_sample_batch() -> tuple[TensorDict, BatchMeta]:
    """One sample routed to one unit, the smallest input put_data accepts."""
    metadata = BatchMeta(
        global_indexes=[0],
        partition_ids=["0"],
        field_schema={"input_ids": {"dtype": torch.int64, "shape": (2,), "is_nested": False, "is_non_tensor": False}},
        production_status=np.ones(1, dtype=np.int8),
    )
    return TensorDict({"input_ids": torch.zeros(1, 2, dtype=torch.int64)}, batch_size=1), metadata


async def _failing_put_attempts(data_parser) -> int:
    """Count the attempts put_data makes when the unit never answers."""
    manager = _manager(unit_a=_server_info("unit_a", "10.0.0.7", 5555))
    manager.notify_data_update = AsyncMock()
    manager._put_to_single_storage_unit = AsyncMock(side_effect=StorageUnitTimeout("no answer"))
    data, metadata = _single_sample_batch()

    with (
        patch.object(manager, "_diagnose_storage_unit", return_value="diagnosis"),
        pytest.raises(StorageUnitTimeout),
    ):
        await manager.put_data(data, metadata, data_parser=data_parser)

    manager.notify_data_update.assert_not_awaited()
    return manager._put_to_single_storage_unit.await_count


@pytest.mark.asyncio
async def test_parser_backed_put_is_not_replayed():
    """A parser re-runs on the unit, and the public API does not constrain its side effects."""
    assert await _failing_put_attempts(data_parser=lambda field_data: field_data) == 1


@pytest.mark.asyncio
async def test_put_without_a_parser_is_still_retried():
    """The retry must stay in force for the ordinary put, which is a plain overwrite."""
    assert await _failing_put_attempts(data_parser=None) == ssm.TQ_SIMPLE_STORAGE_MAX_ATTEMPTS


@pytest.mark.asyncio
async def test_lost_request_recovers_on_retry():
    """A request lost in flight must be recovered by a second attempt, not kill the job."""
    manager = _manager(unit_a=_server_info("unit_a", "10.0.0.7", 5555))
    attempts = []

    async def flaky():
        attempts.append(1)
        if len(attempts) == 1:
            raise StorageUnitTimeout("no answer")
        return "payload"

    with patch.object(manager, "_diagnose_storage_unit") as diagnose:
        result = await manager._request_with_retry("get", "unit_a", "samples=4", flaky)

    assert result == "payload"
    assert len(attempts) == 2, "the retry must issue a second attempt on a new connection"
    assert diagnose.call_count == 0, "a recovered request must not pay for a diagnosis"


@pytest.mark.asyncio
async def test_retry_gives_up_after_configured_attempts():
    """Attempts are bounded, and the final failure still raises for the caller to handle."""
    manager = _manager(unit_a=_server_info("unit_a", "10.0.0.7", 5555))
    attempts = []

    async def always_timeout():
        attempts.append(1)
        raise StorageUnitTimeout("no answer")

    with (
        patch.object(ssm, "TQ_SIMPLE_STORAGE_MAX_ATTEMPTS", 3),
        patch.object(manager, "_diagnose_storage_unit", return_value="diagnosis") as diagnose,
        pytest.raises(StorageUnitTimeout),
    ):
        await manager._request_with_retry("get", "unit_a", "samples=4", always_timeout)

    assert len(attempts) == 3
    diagnose.assert_called_once()


@pytest.mark.asyncio
async def test_a_nonpositive_attempt_count_still_issues_one_request():
    """Skipping the request would report success, and let put_data publish absent data."""
    manager = _manager(unit_a=_server_info("unit_a", "10.0.0.7", 5555))
    attempts = []

    async def succeed():
        attempts.append(1)
        return "payload"

    with patch.object(ssm, "TQ_SIMPLE_STORAGE_MAX_ATTEMPTS", 0):
        result = await manager._request_with_retry("get", "unit_a", "samples=4", succeed)

    assert result == "payload"
    assert len(attempts) == 1


@pytest.mark.asyncio
async def test_errors_reported_by_the_unit_are_not_retried():
    """Only a missing answer is worth another connection; a real error must surface at once."""
    manager = _manager(unit_a=_server_info("unit_a", "10.0.0.7", 5555))
    attempts = []

    async def hard_failure():
        attempts.append(1)
        raise RuntimeError("storage unit rejected the request")

    with pytest.raises(RuntimeError, match="rejected"):
        await manager._request_with_retry("get", "unit_a", "samples=4", hard_failure)

    assert len(attempts) == 1


@pytest.mark.asyncio
async def test_retry_log_adds_request_shape_without_echoing_the_failure(caplog):
    """The failure already names the unit and endpoint; the retry line adds shape, not an echo."""
    manager = _manager(unit_a=_server_info("unit_a", "10.0.0.7", 5555))
    attempts = []

    async def flaky():
        attempts.append(1)
        if len(attempts) == 1:
            raise StorageUnitTimeout("no answer in 200s during get from storage unit unit_a at 10.0.0.7:5555")
        return "payload"

    with caplog.at_level(logging.WARNING):
        await manager._request_with_retry("get", "unit_a", "samples=4 fields=['input_ids']", flaky)

    warning = next(r for r in caplog.records if "retry" in r.message)
    assert "samples=4 fields=['input_ids']" in warning.message
    assert warning.message.count("10.0.0.7:5555") == 1
    assert warning.message.count("unit_a") == 1


@pytest.mark.asyncio
async def test_retry_logs_timeout_detail(caplog):
    """The retry warning must carry the size reported by the failed attempt."""
    manager = _manager(unit_a=_server_info("unit_a", "10.0.0.7", 5555))
    attempts = []

    async def flaky():
        attempts.append(1)
        if len(attempts) == 1:
            raise StorageUnitTimeout("serialized_mb=12.5")
        return "payload"

    with caplog.at_level(logging.WARNING):
        await manager._request_with_retry("put", "unit_a", "samples=4", flaky)

    warning = next(r for r in caplog.records if "retry" in r.message)
    assert "serialized_mb=12.5" in warning.message


def test_log_heavy_operation_reports_the_measured_wire_size(caplog):
    """Both call sites hand over serialized bytes, so the message reports them as measured."""
    with (
        patch.object(common, "TQ_STORAGE_LARGE_PAYLOAD_MB", 0.0),
        patch.object(common, "TQ_STORAGE_SLOW_REQUEST_SECONDS", 1e9),
        caplog.at_level(logging.WARNING),
    ):
        common.log_heavy_operation("TQ_STORAGE_test", "put", 0.25, 512 * 2**20, "to unit_a samples=4")

    warning = next(r for r in caplog.records if "heavy put" in r.message)
    assert "serialized_mb=512.0" in warning.message
    assert "to unit_a samples=4" in warning.message


def test_log_heavy_operation_stays_silent_below_both_thresholds(caplog):
    """Only requests past a threshold are worth a line; the normal path must not log."""
    with (
        patch.object(common, "TQ_STORAGE_LARGE_PAYLOAD_MB", 256.0),
        patch.object(common, "TQ_STORAGE_SLOW_REQUEST_SECONDS", 5.0),
        caplog.at_level(logging.WARNING),
    ):
        common.log_heavy_operation("TQ_STORAGE_test", "get", 0.01, 2**20, "samples=4")

    assert not caplog.records


@pytest.mark.asyncio
async def test_diagnosis_blames_the_link_when_the_unit_still_answers():
    """A unit that answers a fresh probe was not the one that stalled."""
    manager = _manager(unit_a=_server_info("unit_a", "10.0.0.7", 5555))
    probe_body = {
        "active_keys": 4,
        "process_rss_bytes": 1_330_159_616,
        "op_stats": {"GET_DATA": {"request_count": 1025}},
    }

    async def reachable(*args, **kwargs):
        return None, _Writer()

    with (
        patch.object(ssm.asyncio, "open_connection", reachable),
        patch.object(manager, "_probe_storage_unit", return_value=probe_body),
    ):
        diagnosis = await manager._diagnose_storage_unit("unit_a")

    assert "tcp=up" in diagnosis
    assert "verdict=request_lost_in_flight" in diagnosis
    assert "1025" in diagnosis


@pytest.mark.asyncio
async def test_diagnosis_blames_the_unit_when_the_probe_times_out():
    """A probe timeout means the worker thread itself stopped serving."""
    manager = _manager(unit_a=_server_info("unit_a", "10.0.0.7", 5555))

    async def reachable(*args, **kwargs):
        return None, _Writer()

    async def probe_timeout(**kwargs):
        raise zmq.error.Again()

    with (
        patch.object(ssm.asyncio, "open_connection", reachable),
        patch.object(manager, "_probe_storage_unit", probe_timeout),
    ):
        diagnosis = await manager._diagnose_storage_unit("unit_a")

    assert "verdict=unit_not_serving" in diagnosis


@pytest.mark.asyncio
async def test_diagnosis_reports_an_unreachable_endpoint():
    """When the port itself is gone the diagnosis must say so rather than blame the unit."""
    manager = _manager(unit_a=_server_info("unit_a", "10.0.0.7", 5555))

    async def refused(*args, **kwargs):
        raise ConnectionRefusedError()

    async def probe_timeout(**kwargs):
        raise zmq.error.Again()

    with (
        patch.object(ssm.asyncio, "open_connection", refused),
        patch.object(manager, "_probe_storage_unit", probe_timeout),
    ):
        diagnosis = await manager._diagnose_storage_unit("unit_a")

    assert "tcp=down(ConnectionRefusedError)" in diagnosis


@pytest.mark.asyncio
async def test_diagnosis_never_raises_on_an_unknown_unit():
    """Diagnosis runs while another failure is being reported and must not mask it."""
    manager = _manager()

    assert "unit_not_registered" in await manager._diagnose_storage_unit("missing_unit")


class _Writer:
    """Minimal stand-in for the writer half returned by ``asyncio.open_connection``."""

    def close(self):
        pass
