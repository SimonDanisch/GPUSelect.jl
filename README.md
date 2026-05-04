# GPUSelect

GPU backend selection for [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl).
Detects hardware via driver libraries (no CLI tools required), installs the right backend once, and
gives every script a one-liner to load it.

Supports **CUDA**, **AMDGPU**, **Metal** (Apple Silicon), **oneAPI** (Intel), and **Lava** (Vulkan).

> **Intended for scripts, benchmarks, and applications — not libraries.**
>
> Libraries should accept a backend as a parameter and let the caller decide, using
> [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl) or
> [GPUArrays.jl](https://github.com/JuliaGPU/GPUArrays.jl) abstractions directly.
> GPUSelect can be appropriate for library *defaults* though:
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
backend = GPUSelect.Backend()      # ROCBackend() / CUDABackend() / CPU() / …
AT      = GPUSelect.Storage()      # ROCArray     / CuArray       / Array  / …

x = AT{Float32}(undef, 1024)      # allocate on the right device
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
| `Backend(:CPU)` | `KernelAbstractions.CPU()` — always works, no package needed. |
| `Backend(:Lava)` | `LavaBackend()` — Vulkan backend; falls back to lavapipe when no discrete GPU. |
| `Storage()` | The GPU array type (`ROCArray`, `CuArray`, …). |
| `Storage(:CPU)` | `Array`. |

Both `Backend` and `Storage` accept:

- `fallback=true` — silently return `CPU()` / `Array` and emit a warning when the backend is unavailable (default: `true`).
- `fallback=false` — error instead of falling back.
- `install=true` — call `auto_install!()` automatically if no backend is found (default: value of `get_install_preference()`).

### Hardware detection

```julia
GPUSelect.find_gpu_pkgid()    # -> Base.PkgId or nothing
GPUSelect.installed_backends() # -> Vector{String} from the active manifest
```

Detection probes driver shared libraries (`libcuda`, `libamdhip64`, `libze_loader`, `libvulkan`, …)
via `Libdl` — no `nvidia-smi` / `rocm-smi` required, works on Windows, Linux, and macOS.

### Verification

```julia
GPUSelect.test_backends()
```

Prints a live report from the calling environment:

```
────────────────────────────────────────────────────────────
  GPUSelect — Backend Test
  Linux / x86_64
────────────────────────────────────────────────────────────
  Detected hardware:  AMDGPU
  Installed backends: AMDGPU, Lava

  :CPU   Backend=KernelAbstractions.CPU  Storage=Array
  :GPU   Backend=ROCBackend              Storage=ROCArray
  :Lava  Backend=LavaBackend             Storage=LavaArray
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
