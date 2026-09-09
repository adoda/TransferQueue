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
from unittest.mock import patch

import pytest
import torch
import zmq

from transfer_queue.storage.managers import simple_storage_manager as ssm
from transfer_queue.storage.managers.simple_storage_manager import (
    AsyncSimpleStorageManager,
    StorageUnitTimeout,
)
from transfer_queue.utils.common import estimate_payload_bytes
from transfer_queue.utils.enum_utils import Role
from transfer_queue.utils.zmq_utils import ZMQServerInfo


def _manager(**units: ZMQServerInfo) -> AsyncSimpleStorageManager:
    """Build a manager carrying only the state the retry and diagnosis helpers read."""
    manager = AsyncSimpleStorageManager.__new__(AsyncSimpleStorageManager)
    manager.storage_manager_id = "TQ_STORAGE_test"
    manager.storage_unit_infos = dict(units)
    return manager


def _server_info(unit_id: str, ip: str, port: int) -> ZMQServerInfo:
    return ZMQServerInfo(role=Role.STORAGE, id=unit_id, ip=ip, ports={"put_get_socket": port})


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
async def test_retry_logs_endpoint_and_request_shape(caplog):
    """The retry warning must name the endpoint and the request, to correlate both ends."""
    manager = _manager(unit_a=_server_info("unit_a", "10.0.0.7", 5555))
    attempts = []

    async def flaky():
        attempts.append(1)
        if len(attempts) == 1:
            raise StorageUnitTimeout("no answer")
        return "payload"

    with caplog.at_level(logging.WARNING):
        await manager._request_with_retry("get", "unit_a", "samples=4 fields=['input_ids']", flaky)

    warning = next(r for r in caplog.records if "retry" in r.message)
    assert "10.0.0.7:5555" in warning.message
    assert "samples=4" in warning.message


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


def test_estimate_payload_bytes_handles_put_and_get_shapes():
    """Both the batched put shape and the per-sample get shape must be measurable."""
    batched = {"input_ids": torch.zeros(4, 8, dtype=torch.int64)}
    per_sample = {"input_ids": [torch.zeros(8, dtype=torch.int64) for _ in range(4)]}

    assert estimate_payload_bytes(batched) == 4 * 8 * 8
    assert estimate_payload_bytes(per_sample) == 4 * 8 * 8


def test_estimate_payload_bytes_degrades_to_zero_on_unmeasurable_input():
    """Size estimation is diagnostic only and must never break the path it reports on."""
    assert estimate_payload_bytes(object()) == 0
    assert estimate_payload_bytes({"meta": "not a tensor"}) == 0
