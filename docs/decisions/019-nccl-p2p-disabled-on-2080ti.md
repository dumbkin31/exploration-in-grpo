# 019: Disable NCCL CUDA P2P on the 4-GPU 2080 Ti layout

**Measured on Ada, 2026-10-28.** The RTX 2080 Ti allocation reported `False` for every
`torch.cuda.can_device_access_peer(i, j)` pair. The node topology showed PHB links within
each CPU socket and SYS links across NUMA nodes, with no NVLink. The 4-GPU FSDP2 startup
then failed while broadcasting the model state:

```text
Cuda failure 217 'peer access is not supported between these two devices'
```

## Decision

Set `NCCL_P2P_DISABLE=1`, `NCCL_SHM_DISABLE=1`, and `NCCL_IB_DISABLE=1` in the Ray
runtime environment for `layout=nlp_4gpu`. The first prevents direct CUDA peer-to-peer
transport; the second prevents shared-memory CUDA IPC, which also failed with CUDA
error 217 on the measured nodes; the third disables an unavailable multi-node transport.
NCCL can then use its host-mediated socket fallback. The settings are deliberately not
global: the H100 and single-GPU layouts should retain their normal NCCL transport choices.

The layout validation requires these settings so a future edit cannot silently reintroduce
the startup failure.
