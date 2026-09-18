"""Integration test of the vLLM wrapper against the real vLLM 0.24.0 interface.

Skipped in the CPU dev environment; runs on the cluster (``pytest -m needs_vllm``) before the
smoke test so an interface change in vLLM is caught without launching an engine.
"""

from __future__ import annotations

import pytest
import torch

from tests.conftest import needs_vllm

pytestmark = [needs_vllm, pytest.mark.needs_vllm]


def test_subclasses_vllm_abc_and_handles_a_batch(tmp_path):
    from vllm import SamplingParams
    from vllm.v1.sample.logits_processor import BatchUpdate, LogitsProcessor, MoveDirectionality

    from cuts.config import CutsParams
    from cuts.vllm_logits_processor import CutsLogitsProcessor

    assert issubclass(CutsLogitsProcessor, LogitsProcessor)
    lp = CutsLogitsProcessor(vllm_config=None, device=torch.device("cpu"), is_pin_memory=False)
    assert lp.is_argmax_invariant() is False

    cuts_params = CutsParams(k=3, delta=0.0, t_warm=1, stats_dir=str(tmp_path), step=4, uid="p", session_id=1)
    cuts_sp = SamplingParams(temperature=1.0, extra_args=cuts_params.to_extra_args())
    std_sp = SamplingParams(temperature=1.0)
    CutsLogitsProcessor.validate_params(cuts_sp)
    CutsLogitsProcessor.validate_params(std_sp)
    with pytest.raises(ValueError):
        CutsLogitsProcessor.validate_params(SamplingParams(extra_args={"cuts": {"k": 0}}))

    out_ids: list[int] = []
    lp.update_state(
        BatchUpdate(
            batch_size=2, removed=[], added=[(0, cuts_sp, [1, 2], out_ids), (1, std_sp, [3], [])], moved=[]
        )
    )
    assert lp.num_tracked_requests == 1
    logits = torch.randn(2, 50)
    same = lp.apply(logits.clone())
    assert torch.equal(same, logits), "before T_warm the processor is a no-op"
    out_ids.append(7)
    out = lp.apply(logits.clone())
    assert int(torch.isfinite(out[0]).sum()) == 3 and torch.all(out[0][torch.isfinite(out[0])] == 0)
    assert torch.equal(out[1], logits[1])

    lp.update_state(BatchUpdate(batch_size=2, removed=[], added=[], moved=[(0, 1, MoveDirectionality.SWAP)]))
    out = lp.apply(logits.clone())
    assert torch.equal(out[0], logits[0]) and int(torch.isfinite(out[1]).sum()) == 3

    lp.update_state(BatchUpdate(batch_size=1, removed=[1], added=[], moved=[]))
    assert lp.num_tracked_requests == 0
    lp.close()
    files = list(tmp_path.glob("cuts_stats_*.jsonl"))
    assert len(files) == 1 and '"step": 4' in files[0].read_text()
