# NVIDIA driver/library mismatch recovery

## Symptom

```text
Failed to initialize NVML: Driver/library version mismatch
```

This normally means the userspace NVIDIA libraries were upgraded while the kernel still has an older NVIDIA module loaded. Ollama may still discover one or more GPUs through its own CUDA runtime discovery while NVML monitoring fails; that does not prove every card or driver layer is healthy.

## Read-only diagnosis

```bash
scripts/diagnose.sh

cat /proc/driver/nvidia/version
modinfo -F version nvidia
readlink -f /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1
lspci -nn | grep -i nvidia
```

The loaded version from `/proc/driver/nvidia/version` should match the installed module reported by `modinfo` and the NVML library filename.

## Safest repair

Finish or stop GPU workloads, then reboot the machine. On a healthy installation, rebooting normally loads the already-installed matching kernel module and is safer than attempting to unload NVIDIA modules beneath running display, CUDA, or container workloads. If the mismatch remains after reboot, inspect the selected kernel and driver packages rather than repeatedly unloading live modules.

After reboot:

```bash
nvidia-smi
scripts/diagnose.sh
curl -fsS http://127.0.0.1:8008/api/system/status | python3 -m json.tool
```

Confirm both GPUs, approximately 24 GB VRAM each, and matching driver versions before benchmarking Q8 or multi-GPU models.

## Why LocalLLM does not reload modules automatically

Removing `nvidia`, `nvidia_uvm`, `nvidia_drm`, or `nvidia_modeset` can interrupt displays, inference, containers, and other users. The setup scripts deliberately avoid that destructive system-wide action. They also avoid installing or changing kernel drivers.
