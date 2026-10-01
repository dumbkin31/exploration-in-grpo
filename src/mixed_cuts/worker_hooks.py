"""Ray ``worker_process_setup_hook`` for every worker of the training job (configs/train/base_grpo.yaml).

Ray takes a single callable, so this runs each patch in turn:
  * ``sdpa_patch``: transformers' SDPA repeats KV heads on sm_75 (memory-efficient kernel, decision 012)
  * ``lora_sync_patch``: the merged-LoRA weight sync unmerges instead of cloning the base model to CPU
  * ``zmq_socket_patch``: the weight-sync socket carries the run name (two runs on one machine)
"""

from __future__ import annotations


def install() -> None:
    from mixed_cuts import lora_sync_patch, sdpa_patch, zmq_socket_patch

    sdpa_patch.install()
    lora_sync_patch.install()
    zmq_socket_patch.install()
