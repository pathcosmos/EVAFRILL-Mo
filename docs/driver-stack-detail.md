# Driver & System Stack Detail

**Machine**: DCTN-0306095349
**Date**: 2026-03-06

---

## 1. NVIDIA Driver

| Item | Value |
|------|-------|
| Driver Version | 580.95.05 |
| Release Type | NVIDIA UNIX Open Kernel Module |
| Build Date | Tue Sep 23 09:55:41 UTC 2025 |
| GCC Version | 13.3.0 (Ubuntu 13.3.0-6ubuntu2~24.04) |
| GSP Firmware | 580.95.05 |
| Persistence Mode | On (all GPUs) |

### Kernel Modules

| Module | Size | Used By |
|--------|------|---------|
| nvidia | 14,381,056 | nvidia_uvm, nvidia_peermem, gdrdrv, nvidia_modeset |
| nvidia_uvm | 2,166,784 | 8 references |
| nvidia_drm | 139,264 | 2 references |
| nvidia_modeset | 1,814,528 | nvidia_drm |
| nvidia_peermem | 16,384 | GPUDirect RDMA peer memory |
| ib_uverbs | 200,704 | nvidia_peermem, rdma_ucm, mlx5_ib |

---

## 2. GPU Hardware (7x NVIDIA B200)

### Per-GPU Details

| GPU | Bus-ID | VRAM | Power (Idle/Max) | Temp | Clock (GFX/Mem) | CC | ECC Errors |
|-----|--------|------|-------------------|------|------------------|-----|------------|
| 0 | 04:00.0 | 183,359 MiB (179 GB) | 142W / 1,000W | 23°C | 120 / 3,996 MHz | 10.0 | 0 |
| 1 | 14:00.0 | 183,359 MiB (179 GB) | 140W / 1,000W | 22°C | 120 / 3,996 MHz | 10.0 | 0 |
| 2 | 64:00.0 | 183,359 MiB (179 GB) | 139W / 1,000W | 23°C | 120 / 3,996 MHz | 10.0 | 0 |
| 3 | 77:00.0 | 183,359 MiB (179 GB) | 144W / 1,000W | 24°C | 120 / 3,996 MHz | 10.0 | 0 |
| 4 | 84:00.0 | 183,359 MiB (179 GB) | 141W / 1,000W | 23°C | 120 / 3,996 MHz | 10.0 | 0 |
| 5 | 94:00.0 | 183,359 MiB (179 GB) | 142W / 1,000W | 23°C | 120 / 3,996 MHz | 10.0 | 0 |
| 6 | E4:00.0 | 183,359 MiB (179 GB) | 142W / 1,000W | 23°C | 120 / 3,996 MHz | 10.0 | 0 |

- **Total VRAM**: 1,283,513 MiB (~1,248 GB)
- **Compute Capability**: 10.0 (sm_100, Blackwell architecture)
- **VBIOS**: 97.00.C5.00.2F
- **Board Part**: 692-2G525-0225-001
- **GPU Part**: 2901-886-A1
- **MIG Mode**: Disabled (all GPUs)
- **Display**: Off (headless server)

### 8번째 GPU 참고

- `lspci`는 8개 NVIDIA 장치 감지 (Bus-ID: 04, 14, 64, 77, 84, 94, E4, F6)
- `nvidia-smi`는 7개만 표시 (F6:00.0 미인식)
- 가능 원인: 드라이버 이슈, BIOS 설정, 하드웨어 결함

### NVLink Topology

모든 GPU 간 **NV18 (18-way NVLink)** 직접 연결:

```
GPU간 연결 매트릭스:
     GPU0  GPU1  GPU2  GPU3  GPU4  GPU5  GPU6
GPU0  -    NV18  NV18  NV18  NV18  NV18  NV18
GPU1 NV18   -    NV18  NV18  NV18  NV18  NV18
GPU2 NV18  NV18   -    NV18  NV18  NV18  NV18
GPU3 NV18  NV18  NV18   -    NV18  NV18  NV18
GPU4 NV18  NV18  NV18  NV18   -    NV18  NV18
GPU5 NV18  NV18  NV18  NV18  NV18   -    NV18
GPU6 NV18  NV18  NV18  NV18  NV18  NV18   -
```

- 링크당 대역폭: 53.125 GB/s
- GPU당 총 NVLink 대역폭: 18 x 53.125 = **956.25 GB/s**
- 5세대 NVLink (NVLink 5.0, Blackwell)

### NUMA Affinity

| NUMA Node | CPUs | RAM | GPUs | NICs |
|-----------|------|-----|------|------|
| Node 0 | 0-35 (36 cores) | 1,160 GB | GPU 0, 1, 2, 3 | mlx5_0, mlx5_1, mlx5_2 |
| Node 1 | 36-71 (36 cores) | 1,157 GB | GPU 4, 5, 6 | mlx5_3~mlx5_9 |

NUMA distance: Node 0↔1 = 32 (inter-socket)

---

## 3. CUDA Toolkit

| Item | Value |
|------|-------|
| CUDA Version (nvidia-smi) | 13.0 |
| CUDA Toolkit (nvcc) | 13.1, V13.1.80 |
| Install Path | /usr/local/cuda-13.1 |
| Symlink | /usr/local/cuda → /usr/local/cuda-13 |

### CUDA Packages (System-installed via dpkg)

| Package | Version | Description |
|---------|---------|-------------|
| cuda-nvcc-13-1 | 13.1.80-1 | CUDA C/C++ compiler |
| cuda-cudart-13-1 | 13.1.80-1 | CUDA runtime library |
| cuda-cudart-dev-13-1 | 13.1.80-1 | CUDA runtime dev headers |
| cuda-cupti-13-1 | 13.1.75-1 | CUDA profiling tools |
| cuda-cupti-dev-13-1 | 13.1.75-1 | CUPTI dev headers |
| cuda-nvrtc-13-1 | 13.1.80-1 | CUDA runtime compilation |
| cuda-nvrtc-dev-13-1 | 13.1.80-1 | NVRTC dev headers |
| cuda-nvtx-13-1 | 13.1.68-1 | NVIDIA Tools Extension |
| cuda-nvml-dev-13-1 | 13.1.68-1 | NVML dev headers |
| cuda-gdb-13-1 | 13.1.68-1 | CUDA debugger |
| cuda-sanitizer-13-1 | 13.1.75-1 | CUDA sanitizer |
| cuda-cccl-13-1 | 13.1.78-1 | CUDA C++ Core Libraries |
| cuda-crt-13-1 | 13.1.80-1 | CUDA runtime tools |
| cuda-compat-13-1 | 590.44.01-0ubuntu1 | CUDA forward compat |
| cuda-driver-dev-13-1 | 13.1.80-1 | CUDA driver dev |
| cuda-profiler-api-13-1 | 13.1.80-1 | CUPTI profiler API |
| cuda-cuobjdump-13-1 | 13.1.80-1 | CUDA object dump |
| cuda-nvdisasm-13-1 | 13.1.80-1 | CUDA disassembler |
| cuda-nvprune-13-1 | 13.1.80-1 | CUDA ELF pruner |
| cuda-culibos-13-1 | 13.1.68-1 | CUDA lib OS |
| cuda-culibos-dev-13-1 | 13.1.68-1 | CUDA lib OS dev |
| libnvvm-13-1 | 13.1.80-1 | NVVM compiler library |
| libnvjitlink-13-1 | 13.1.80-1 | NV JIT Linker |
| libnvjitlink-dev-13-1 | 13.1.80-1 | NV JIT Linker dev |
| libnvptxcompiler-13-1 | 13.1.80-1 | PTX compiler |

### CUDA Toolkit Libraries (in /usr/local/cuda/lib64/)

34 shared libraries total:

| Library | Description |
|---------|-------------|
| libcublas.so | Basic Linear Algebra Subroutines |
| libcublasLt.so | cuBLAS Lightweight (Tensor Core ops) |
| libcudart.so | CUDA Runtime |
| libcufft.so | Fast Fourier Transform |
| libcufftw.so | FFTW-compatible wrapper |
| libcufile.so | GPUDirect Storage |
| libcufile_rdma.so | GPUDirect Storage RDMA |
| libcupti.so | Profiling Tools Interface |
| libcurand.so | Random number generation |
| libcusolver.so | Dense/sparse solvers |
| libcusolverMg.so | Multi-GPU solver |
| libcusparse.so | Sparse matrix operations |
| libcusparseLt.so | Structured sparsity |
| libnvJitLink.so | JIT Linker |
| libnvblas.so | Multi-GPU BLAS |
| libnvjpeg.so | JPEG encode/decode |
| libnvrtc.so | Runtime compilation |
| libnvrtc-builtins.so | NVRTC built-in functions |
| libnvtx3interop.so | NVTX interop |
| libnppc/nppial/nppicc/... | NVIDIA Performance Primitives (image/signal) |
| libcheckpoint.so | CUDA checkpoint/restore |

### CUDA Toolkit CLI Tools (in /usr/local/cuda/bin/)

| Tool | Description |
|------|-------------|
| nvcc | CUDA C/C++ compiler |
| ptxas | PTX assembler |
| nvlink | CUDA linker |
| cuobjdump | CUDA object file dump |
| nvdisasm | CUDA disassembler |
| nvprune | CUDA ELF pruner |
| cuda-gdb | CUDA debugger |
| compute-sanitizer | Memory/race checker |
| nsys | Nsight Systems profiler |
| fatbinary | Fat binary packager |
| bin2c | Binary to C header |
| __nvcc_device_query | Device capability query |

---

## 4. cuDNN (CUDA Deep Neural Network Library)

| Item | Value |
|------|-------|
| Version | 9.17.0.29 |
| CUDA Target | CUDA 13 |
| PyTorch 인식 버전 | 91002 (9.10.02) |

### System Packages

| Package | Version |
|---------|---------|
| libcudnn9-cuda-13 | 9.17.0.29-1 (runtime) |
| libcudnn9-dev-cuda-13 | 9.17.0.29-1 (development) |
| libcudnn9-headers-cuda-13 | 9.17.0.29-1 (headers) |

### Library Files

```
/usr/lib/x86_64-linux-gnu/libcudnn.so
/usr/lib/x86_64-linux-gnu/libcudnn.so.9
/usr/lib/x86_64-linux-gnu/libcudnn.so.9.17.0
/usr/lib/x86_64-linux-gnu/libcudnn_adv.so         # Advanced operations
/usr/lib/x86_64-linux-gnu/libcudnn_adv.so.9
/usr/lib/x86_64-linux-gnu/libcudnn_cnn.so          # CNN operations
/usr/lib/x86_64-linux-gnu/libcudnn_cnn.so.9
/usr/lib/x86_64-linux-gnu/libcudnn_engines_precompiled.so  # Pre-compiled engines
```

---

## 5. NCCL (NVIDIA Collective Communication Library)

| Item | Value |
|------|-------|
| Version | 2.28.9 |
| CUDA Target | CUDA 13.0 |

### System Packages

| Package | Version |
|---------|---------|
| libnccl2 | 2.28.9-1+cuda13.0 (runtime) |
| libnccl-dev | 2.28.9-1+cuda13.0 (development) |

### Library Files & Plugins

```
/usr/lib/x86_64-linux-gnu/libnccl.so
/usr/lib/x86_64-linux-gnu/libnccl.so.2
/usr/lib/x86_64-linux-gnu/libnccl.so.2.28.9
```

**HPC-X / RDMA 플러그인**:

| Plugin | Path |
|--------|------|
| NCCL wrapper | /opt/hpcx/clusterkit/lib/libnccl_wrapper.so |
| NCCL RDMA SHARP | /opt/hpcx/nccl_rdma_sharp_plugin/lib/libnccl-net.so |
| NCCL Spectrum-X | /opt/hpcx/nccl_spectrum-x_plugin/lib/libnccl-net.so |
| NCCL MRC | /opt/hpcx/nccl_mrc_plugin/lib/libnccl-net-mrc.so |
| AWS OFI NCCL | /opt/amazon/aws-ofi-nccl/lib/libnccl-net-ofi.so |
| NCCL tuner OFI | /opt/amazon/aws-ofi-nccl/lib/libnccl-tuner-ofi.so |
| NCCL profiler | /opt/hpcx/nccl_spectrum-x_plugin/lib/libnccl-profiler-inspector.so |

---

## 6. TensorRT

| Item | Value |
|------|-------|
| Version | 10.14.1.48 |
| CUDA Target | CUDA 13.0 |

### System Packages

| Package | Version |
|---------|---------|
| libnvinfer10 | 10.14.1.48-1+cuda13.0 (runtime) |
| libnvinfer-dev | 10.14.1.48-1+cuda13.0 (dev) |
| libnvinfer-lean10 | 10.14.1.48-1+cuda13.0 (lean runtime) |
| libnvinfer-dispatch10 | 10.14.1.48-1+cuda13.0 (dispatch) |
| libnvinfer-plugin10 | 10.14.1.48-1+cuda13.0 (plugins) |
| libnvinfer-vc-plugin10 | 10.14.1.48-1+cuda13.0 (VC plugins) |
| libnvinfer-bin | 10.14.1.48-1+cuda13.0 (tools) |
| libnvinfer-headers-dev | 10.14.1.48-1+cuda13.0 (headers) |
| libnvinfer-headers-plugin-dev | 10.14.1.48-1+cuda13.0 (plugin headers) |
| libnvinfer-headers-python-plugin-dev | 10.14.1.48-1+cuda13.0 (Python plugin) |
| libnvonnxparsers10 | 10.14.1.48-1+cuda13.0 (ONNX parser) |
| libnvonnxparsers-dev | 10.14.1.48-1+cuda13.0 (ONNX parser dev) |
| tensorrt-dev | 10.14.1.48-1+cuda13.0 (dev meta) |

---

## 7. Additional CUDA Libraries

| Package | Version | Description |
|---------|---------|-------------|
| libcublas-13-1 | 13.2.0.9-1 | cuBLAS |
| libcufft-13-1 | 12.1.0.31-1 | cuFFT |
| libcufile-13-1 | 1.16.0.49-1 | cuFile (GPUDirect Storage) |
| libcusolver-13-1 | 12.0.7.41-1 | cuSOLVER |
| libcusparse-13-1 | 12.7.2.19-1 | cuSPARSE |
| libcusparselt0-cuda-13 | 0.8.1.1-1 | cuSPARSELt (structured sparsity) |
| cublasmp-cuda-13 | 0.7.0.125-1 | cuBLASMp (multi-process BLAS) |
| libnvshmem3-cuda-13 | 3.4.5-1 | NVSHMEM (symmetric memory) |

---

## 8. Networking & InfiniBand/RDMA

| Package | Version |
|---------|---------|
| ibverbs-providers | 56.0-1 |
| ibverbs-utils | 56.0-1 |
| libibverbs1 | 56.0-1 |
| libibverbs-dev | 56.0-1 |
| librdmacm1 | 56.0-1 |
| librdmacm-dev | 56.0-1 |
| doca-sdk-rdma | 3.1.0105-1 |
| libfabric1-aws | 2.1.0amzn5.0 |
| libfabric-aws-dev | 2.1.0amzn5.0 |

**NIC**: 10x Mellanox ConnectX (mlx5_0 ~ mlx5_9)

---

## 9. Profiling & Monitoring Tools

| Tool | Version | Description |
|------|---------|-------------|
| nsight-compute | 2025.4.0.12-1 | NVIDIA Nsight Compute (kernel profiler) |
| nsys | (in cuda/bin) | Nsight Systems (system profiler) |
| nvtop | 3.0.2-1 | GPU process monitor (htop-like) |
| compute-sanitizer | (in cuda/bin) | Memory/race condition checker |

---

## 10. System Limits

| Parameter | Value |
|-----------|-------|
| Open files (ulimit -n) | 650,000 |
| Locked memory (ulimit -l) | unlimited |
| Max threads | 18,540,347 |
| VM overcommit | 0 (conservative) |
| CPU cores | 72 |
| Total RAM | 2.2 TiB |
| Free RAM | ~2.0 TiB |
