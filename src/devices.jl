"""
    DeviceInfo

Details about one device, as returned by [`devices`](@ref). Fields that the backend cannot
report are `nothing`.

| field                   | meaning                                                                    |
|:------------------------|:---------------------------------------------------------------------------|
| `backend`               | backend package name: `"CUDA"`, `"AMDGPU"`, `"Metal"`, `"oneAPI"` or `"CPU"` |
| `index`                 | 1-based position in `devices()`                                            |
| `name`                  | device name, e.g. `"NVIDIA A100-SXM4-40GB"`                                |
| `total_memory`          | device memory in bytes (Metal: recommended max working set; CPU: system RAM) |
| `max_threads_per_block` | maximum threads per block / workgroup / threadgroup                        |
| `compute_units`         | streaming multiprocessors, compute units, execution units, GPU cores or CPU threads |
| `capability`            | CUDA compute capability (`VersionNumber`) or AMD architecture (`"gfx90a"`) |
| `uuid`                  | device UUID (Metal: registry ID), as a string                              |
| `pci_bus_id`            | PCI address, e.g. `"0000:3b:00.0"`                                         |
"""
Base.@kwdef struct DeviceInfo
    backend::String
    index::Int
    name::String
    total_memory::Union{Int,Nothing} = nothing
    max_threads_per_block::Union{Int,Nothing} = nothing
    compute_units::Union{Int,Nothing} = nothing
    capability::Union{VersionNumber,String,Nothing} = nothing
    uuid::Union{String,Nothing} = nothing
    pci_bus_id::Union{String,Nothing} = nothing
end

Base.show(io::IO, d::DeviceInfo) = print(io, "DeviceInfo(\"", d.backend, "\", ", d.index, ", \"", d.name, "\")")

function Base.show(io::IO, ::MIME"text/plain", d::DeviceInfo)
    print(io, d.backend, " device ", d.index, ": ", d.name)
    for f in (:total_memory, :max_threads_per_block, :compute_units, :capability, :uuid, :pci_bus_id)
        v = getfield(d, f)
        v === nothing && continue
        print(io, "\n  ", f, ": ", f === :total_memory ? Base.format_bytes(v) : v)
    end
end

"""
    device_info(::Val{name}) -> Vector{DeviceInfo}

Query the devices of backend package `name`. Methods are added by the package extensions
in `ext/`, which load together with the backend package.
"""
function device_info end

# Helpers for the extensions: best-effort property queries.
function _try(f)
    try
        return f()
    catch
        return nothing
    end
end
_int(x) = x === nothing ? nothing : Int(x)

function _format_pci(domain, bus, device)
    any(isnothing, (domain, bus, device)) && return nothing
    hex(x, n) = string(x; base=16, pad=n)
    return "$(hex(domain, 4)):$(hex(bus, 2)):$(hex(device, 2)).0"
end

function cpu_devices()
    info = Sys.cpu_info()
    name = isempty(info) ? string(Sys.ARCH) : strip(first(info).model)
    return [DeviceInfo(; backend="CPU", index=1, name, total_memory=Int(Sys.total_memory()),
                       compute_units=Sys.CPU_THREADS)]
end

"""
    devices(target::Symbol = :GPU; fallback=true, install=get_install_preference()) -> Vector{DeviceInfo}

List the devices of the backend that [`Backend`](@ref) would select for `target`, with their
name, memory, maximum threads per block, compute units, capability, UUID and PCI bus ID.

- `:GPU`: loads the native GPU backend (CUDA, AMDGPU, Metal, or oneAPI) and queries it.
- `:CPU`: describes the host CPU, no package needed.

With `fallback=true`, a GPU backend that is missing or not functional produces a warning and
the CPU description instead, matching `Backend()` and `Storage()`. Lava is not supported yet.

```julia
julia> GPUSelect.devices()
1-element Vector{GPUSelect.DeviceInfo}:
 DeviceInfo("CUDA", 1, "NVIDIA A100-SXM4-40GB")

julia> first(GPUSelect.devices()).total_memory
42298834944
```
"""
function devices(target::Symbol = :GPU; fallback::Bool = true, install::Bool = get_install_preference())
    target === :Lava && error("GPUSelect.devices does not support Lava yet.")
    mod = backend_module(target, install, fallback)
    mod === KernelAbstractions && return cpu_devices()
    name = nameof(mod)
    try
        return Base.invokelatest(device_info, Val(name))::Vector{DeviceInfo}
    catch e
        fallback || rethrow()
        @warn "GPUSelect: could not query $name devices, falling back to CPU." exception=e
        return cpu_devices()
    end
end
