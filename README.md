# GPUSelect

GPU backend selection for [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl).
Detects hardware via driver libraries (no CLI tools required), installs the right backend once, and
gives every script a one-liner to load it.

Supports **CUDA**, **AMDGPU**, **Metal** (Apple Silicon), **oneAPI** (Intel), and **Lava** (Vulkan).

> **Intended for scripts, benchmarks, and applications, not libraries.**
>
> Libraries should accept a backend as a parameter and let the caller decide, using
> [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl) or
> [GPUArrays.jl](https://github.com/JuliaGPU/GPUArrays.jl) abstractions directly.
> GPUSelect *could* be appropriate for library defaults though, but still needs some more discussion if that's a good idea:
>
> ```julia
> function my_kernel_runner(data; backend = GPUSelect.Backend())
>     …
> end
> ```

## Quick start

```julia
# 1. Add GPUSelect to your project once
using Pkg; Pkg.add("GPUSelect")

# 2. Detect hardware and install the matching backend (once per environment)
using GPUSelect
GPUSelect.auto_install!()          # e.g. installs AMDGPU on an AMD machine

# 3. In every script
using GPUSelect
# These directly return the types from the backend
backend = GPUSelect.Backend()      # ROCBackend() / CUDABackend() / CPU() / …
AT      = GPUSelect.Storage()      # ROCArray     / CuArray       / Array  / …

x = AT{Float32}(undef, 1024)      # allocate on the right device
y = adapt(backend, rand(4, 4))    # backend/storage work with adapt
```

## API

### Setup (once per environment)

| Function | Description |
|---|---|
| `auto_install!()` | Detect GPU hardware and `Pkg.add` the matching backend. Pass `force=true` to reinstall. |
| `set_backend!(name)` | Pin a specific backend (`"CUDA"`, `"AMDGPU"`, `"Lava"`, …) in `LocalPreferences.toml`. |
| `clear_backend!()` | Remove the pinned preference; `@load_backend()` falls back to manifest scanning. |
| `set_install_preference!(true)` | Make `Backend()` / `Storage()` auto-install on first call instead of erroring. |

### Per-script

| Call | Returns |
|---|---|
| `Backend()` | `KernelAbstractions.Backend` instance for the detected GPU. |
| `Backend(:CPU)` | `KernelAbstractions.CPU()`, always works, no package needed. |
| `Backend(:Lava)` | `LavaBackend()`. Vulkan backend; falls back to lavapipe when no discrete GPU. |
| `Storage()` | The GPU array type (`ROCArray`, `CuArray`, …). |
| `Storage(:CPU)` | `Array`. |
| `devices()` | `Vector{DeviceInfo}` describing the GPUs of the selected backend (see [Device information](#device-information)). |
| `devices(:CPU)` | A one-element `Vector{DeviceInfo}` describing the host CPU. |
| `known_types()` | Float and integer element types the selected backend supports, from a lookup table; loads nothing (see [Supported element types](#supported-element-types)). |
| `test_types()` | The same, found by running a small kernel per type on the device. |

`Backend`, `Storage`, `devices` and `test_types` accept:

- `fallback=true`: silently return `CPU()` / `Array` and emit a warning when the backend is unavailable (default: `true`).
- `fallback=false`: error instead of falling back.
- `install=true`: **EXPERIMENTAL**, call `auto_install!()` automatically if no backend is found. Still needs some investigation if this is a good idea!

> **`Backend()` / `Storage()` do not auto-install by default.** The `install` keyword
> defaults to `get_install_preference()`, which is `false` unless you opt in once with
> `GPUSelect.set_install_preference!(true)`. Until then, missing backends produce an
> error (or a fallback to CPU when `fallback=true`), never an implicit `Pkg.add`.

### Device information

`GPUSelect.devices()` lists the devices of the backend that `Backend()` selects, through the same
unified interface on every vendor:

```julia
julia> GPUSelect.devices()
1-element Vector{GPUSelect.DeviceInfo}:
 DeviceInfo("Metal", 1, "Apple M5 Pro")

julia> first(GPUSelect.devices())
Metal device 1: Apple M5 Pro
  total_memory: 51.840 GiB
  max_threads_per_block: 1024
  compute_units: 20
  uuid: 1000007ae
```

Each `DeviceInfo` has the fields below; a field is `nothing` when the backend cannot report it.

| Field | Meaning |
|---|---|
| `backend` | `"CUDA"`, `"AMDGPU"`, `"Metal"`, `"oneAPI"` or `"CPU"` |
| `index` | 1-based position in `devices()` |
| `name` | Device name, e.g. `"NVIDIA A100-SXM4-40GB"` |
| `total_memory` | Device memory in bytes (Metal: recommended maximum working set; CPU: system memory) |
| `max_threads_per_block` | Maximum threads per block / workgroup / threadgroup |
| `compute_units` | Streaming multiprocessors (CUDA), compute units (AMDGPU), execution units (oneAPI), GPU cores (Metal) or threads (CPU) |
| `capability` | CUDA compute capability as a `VersionNumber`, or AMD architecture such as `"gfx90a"` |
| `uuid` | Device UUID as a string (Metal: registry ID) |
| `pci_bus_id` | PCI address, e.g. `"0000:3b:00.0"` (CUDA and AMDGPU) |

### Supported element types

Which element types work in a kernel differs per vendor, per device and per version of the
backend package: Metal has no `Float64`, Intel Arc GPUs have no `Float64` either, and the CPU
backend lost 128-bit integers when KernelAbstractions 0.10 moved it to POCL. GPUSelect answers
this in two ways.

`GPUSelect.known_types()` looks the backend up in a table. It loads, compiles and runs nothing,
so it is instant and works before the backend is even installed. It uses the backend version in
your manifest, and lists types that depend on the device, or that the table has no entry for,
separately:

```julia
julia> GPUSelect.known_types("Metal")
Metal supports: Float16, Float32, Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, UInt64
  Float64 not supported: Apple GPUs have no Float64
  Int128 not supported: Metal has no 128-bit integers
  UInt128 not supported: Metal has no 128-bit integers
  BFloat16 depends on device: needs Metal 3.1 (macOS 14) or newer

julia> Float64 in GPUSelect.known_types("CUDA")
true
```

`GPUSelect.test_types()` finds out for certain on your device: for each type it compiles and runs
a small KernelAbstractions kernel that stores, multiplies, adds and reads back values. This loads
the backend and takes a few seconds, since each type compiles its own kernel.

```julia
julia> GPUSelect.test_types()
Metal v1.12.1 device 1 supports: Float16, BFloat16, Float32, Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, UInt64
  Float64 not supported: backend reports no Float64 support
  Int128 not supported: Metal does not support Int128 values, try using Int64 instead
  UInt128 not supported: Metal does not support UInt128 values, try using UInt64 instead

julia> GPUSelect.test_types(GPUSelect.devices()[2]; types=[Float16, Float64])  # a specific device and types
```

The test suite checks the table against `test_types` on whatever hardware it runs on, so running
`Pkg.test("GPUSelect")` on a GPU also verifies the table for that backend.

The queries live in package extensions that load together with the backend package, so they add
no dependencies. `devices(:Lava)` is not supported yet.

### Hardware detection

```julia
GPUSelect.find_gpu_pkgid()    # -> Base.PkgId or nothing
GPUSelect.installed_backends() # -> Vector{String} from the active manifest
```

Detection probes driver shared libraries (`libcuda`, `libamdhip64`, `libze_loader`, `libvulkan`, …)
via `Libdl`, with no `nvidia-smi` / `rocm-smi` required, and works on Windows, Linux, and macOS.

### Verification

```julia
GPUSelect.test_backends()
```

Prints a live report from the calling environment:

```
────────────────────────────────────────────────────────────
  GPUSelect: Backend Test
  Linux / x86_64
────────────────────────────────────────────────────────────
  Detected hardware:  AMDGPU
  Installed backends: AMDGPU, Lava

  :CPU   Backend=KernelAbstractions.CPU  Storage=Array
  :GPU   Backend=ROCBackend              Storage=ROCArray
  :Lava  Backend=LavaBackend             Storage=LavaArray

  GPU device 1: AMD Instinct MI250X  (64.000 GiB)
────────────────────────────────────────────────────────────
```

Run this from the REPL in your project after `auto_install!()` to confirm everything wired up correctly.
`Pkg.test("GPUSelect")` covers the isolation-safe unit tests (constants, preferences, CPU, error paths).

## Multiple backends / explicit choice

When multiple backends are installed, `Backend()` prefers the one `find_gpu_pkgid()` detects from driver libraries, falling back to the first alphabetically. In the rare case that's wrong, uninstall the unwanted backend or keep only the one you need in your project manifest.

## Backends

| Target | Package | Array type | Notes |
|---|---|---|---|
| `:GPU` (CUDA) | [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) | `CuArray` | NVIDIA |
| `:GPU` (AMDGPU) | [AMDGPU.jl](https://github.com/JuliaGPU/AMDGPU.jl) | `ROCArray` | AMD ROCm |
| `:GPU` (Metal) | [Metal.jl](https://github.com/JuliaGPU/Metal.jl) | `MtlArray` | Apple Silicon |
| `:GPU` (oneAPI) | [oneAPI.jl](https://github.com/JuliaGPU/oneAPI.jl) | `oneArray` | Intel |
| `:Lava` | [Lava.jl](https://github.com/JuliaGPU/Lava.jl) | `LavaArray` | Vulkan; falls back to lavapipe |
| `:CPU` | KernelAbstractions | `Array` | Always available |
