"""The SDPA patch must switch transformers to repeated KV heads (memory-efficient kernel on sm_75)."""

from __future__ import annotations

import pytest

transformers = pytest.importorskip("transformers")


def test_install_disables_gqa_and_is_idempotent():
    import transformers.integrations.sdpa_attention as sdpa

    from mixed_cuts import sdpa_patch

    sdpa_patch.install()
    assert sdpa.use_gqa_in_sdpa(None, object()) is False
    fn = sdpa.use_gqa_in_sdpa
    sdpa_patch.install()  # second call is a no-op
    assert sdpa.use_gqa_in_sdpa is fn
