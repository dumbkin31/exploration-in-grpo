# 019: Disable NCCL CUDA P2P on the 4-GPU 2080 Ti layout

**Measured on Ada, 2026-10-28.** The RTX 2080 Ti allocation reported `False` for every
`torch.cuda.can_device_access_peer(i, j)` pair. The node topology showed PHB links within
each CPU socket and SYS links across NUMA nodes, with no NVLink. The 4-GPU FSDP2 startup
then failed while broadcasting the model state:

```text
Cuda failure 217 'peer access is not supported between these two devices'
```

## Decision

Set `NCCL_P2P_DISABLE=1` in the Ray runtime environment for `layout=nlp_4gpu`. This
prevents NCCL from selecting CUDA peer-to-peer transport and allows it to use its
host-mediated fallback paths. The setting is deliberately not global: the H100 and
single-GPU layouts should retain their normal NCCL transport choices.

The layout validation requires this setting so a future edit cannot silently reintroduce
the startup failure.
