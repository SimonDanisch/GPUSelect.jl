using KernelAbstractions: @kernel, @index, @Const

# Element types that GPUSelect reports on, by name so that BFloat16 needs no package.
const TYPE_NAMES = (:Float16, :BFloat16, :Float32, :Float64,
                    :Int8, :Int16, :Int32, :Int64, :Int128,
                    :UInt8, :UInt16, :UInt32, :UInt64, :UInt128)

"""
    TypeSupport

Which element types a backend can store and compute with in a KernelAbstractions kernel.
[`known_types`](@ref) fills it from GPUSelect's lookup table, [`test_types`](@ref) by running
kernels on a device. Types are given by name (`:Float64`, `:BFloat16`, …). Use `T in ts` or
`:T in ts` to check one type.

| field              | meaning                                                                   |
|:-------------------|:--------------------------------------------------------------------------|
| `backend`          | `"CUDA"`, `"AMDGPU"`, `"Metal"`, `"oneAPI"` or `"CPU"`                     |
| `version`          | backend package version it applies to (KernelAbstractions for the CPU), or `nothing` if unknown |
| `device`           | 1-based index of the tested device; `nothing` for the lookup table         |
| `supported`        | types that work                                                           |
| `unsupported`      | `type => reason` for types that do not work                               |
| `device_dependent` | `type => condition` for types that work on some devices only (table only) |
| `unknown`          | types the table has no entry for (table only); use `test_types` for those |
"""
struct TypeSupport
    backend::String
    version::Union{VersionNumber,Nothing}
    device::Union{Int,Nothing}
    supported::Vector{Symbol}
    unsupported::Vector{Pair{Symbol,String}}
    device_dependent::Vector{Pair{Symbol,String}}
    unknown::Vector{Symbol}
end

Base.in(T::Symbol, ts::TypeSupport) = T in ts.supported
Base.in(T::Type, ts::TypeSupport) = nameof(T) in ts.supported

function Base.show(io::IO, ::MIME"text/plain", ts::TypeSupport)
    print(io, ts.backend, ts.version === nothing ? "" : " v$(ts.version)")
    ts.device === nothing || print(io, " device ", ts.device)
    print(io, " supports: ", isempty(ts.supported) ? "none" : join(ts.supported, ", "))
    for (label, list) in (("not supported", ts.unsupported), ("depends on device", ts.device_dependent))
        for (T, why) in list
            print(io, "\n  ", T, " ", label, ": ", why)
        end
    end
    isempty(ts.unknown) || print(io, "\n  unknown (run test_types to find out): ", join(ts.unknown, ", "))
end

# ---------------------------------------------------------------------------------------
# Lookup table
# ---------------------------------------------------------------------------------------

struct TypeRule
    backend::String
    types::Vector{Symbol}
    status::Symbol                          # :supported, :unsupported or :device
    from::Union{VersionNumber,Nothing}      # first backend version the rule applies to
    until::Union{VersionNumber,Nothing}     # first version it no longer applies to
    note::String
end
TypeRule(backend, types, status, note=""; from=nothing, until=nothing) =
    TypeRule(backend, collect(Symbol, types), status, from, until, note)

const INTS_8_64 = (:Int8, :Int16, :Int32, :Int64, :UInt8, :UInt16, :UInt32, :UInt64)
const INTS_128 = (:Int128, :UInt128)

# The first rule that matches a backend, type and version wins. Only add entries that are
# documented by the backend or were checked with `test_types`; anything else stays unknown.
const TYPE_RULES = TypeRule[
    TypeRule("CUDA", (:Float16, :Float32, :Float64, INTS_8_64...), :supported),
    TypeRule("CUDA", (:BFloat16,), :device, "native arithmetic needs compute capability 8.0 (Ampere) or newer"),

    TypeRule("AMDGPU", (:Float16, :Float32, :Float64, INTS_8_64...), :supported),

    TypeRule("Metal", (:Float16, :Float32, INTS_8_64...), :supported),
    TypeRule("Metal", (:Float64,), :unsupported, "Apple GPUs have no Float64"),
    TypeRule("Metal", INTS_128, :unsupported, "Metal has no 128-bit integers"),
    TypeRule("Metal", (:BFloat16,), :device, "needs Metal 3.1 (macOS 14) or newer"),

    TypeRule("oneAPI", (:Float32, INTS_8_64...), :supported),
    TypeRule("oneAPI", (:Float16,), :device, "the device must report FP16 support"),
    TypeRule("oneAPI", (:Float64,), :device, "not on Intel Arc (Alchemist) and most integrated GPUs; available on Data Center GPU Max"),
    TypeRule("oneAPI", (:BFloat16,), :device, "the device must support the bfloat16 conversions extension"),

    # KernelAbstractions 0.9 runs CPU kernels as native Julia code; 0.10 compiles them to SPIR-V for POCL.
    TypeRule("CPU", (:Float16, :Float32, :Float64, INTS_8_64...), :supported),
    TypeRule("CPU", (:BFloat16,), :supported, "with BFloat16s.jl"),
    TypeRule("CPU", INTS_128, :supported; until=v"0.10"),
    TypeRule("CPU", INTS_128, :unsupported, "the SPIR-V back-end has no 128-bit integers"; from=v"0.10"),
]

function rule_applies(r::TypeRule, backend, version)
    r.backend == backend || return false
    if version === nothing   # version unknown: assume the latest release
        return r.until === nothing
    end
    return (r.from === nothing || version >= r.from) && (r.until === nothing || version < r.until)
end

# Version of package `name` in the active project's manifest, without loading it.
function manifest_version(name::AbstractString)
    proj = Base.active_project()
    proj === nothing && return nothing
    manifest = joinpath(dirname(proj), "Manifest.toml")
    isfile(manifest) || return nothing
    data = TOML.parsefile(manifest)
    deps = get(data, "deps", data)
    entries = get(deps, name, nothing)
    entries isa Vector && !isempty(entries) || return nothing
    v = get(first(entries), "version", nothing)
    return v === nothing ? nothing : VersionNumber(v)
end

# The GPU backend `Backend()` would use, without loading anything: the installed one that
# matches the hardware, else the first installed one, else the one the hardware needs.
function selected_gpu_backend()
    found = filter(n -> n ∈ GPU_BACKENDS, installed_backends())
    detected = find_gpu_pkgid()
    if !isempty(found)
        return detected !== nothing && detected.name ∈ found ? detected.name : first(found)
    end
    return detected !== nothing && detected.name ∈ GPU_BACKENDS ? detected.name : nothing
end

"""
    known_types(backend = <selected GPU backend>; version = <installed version>) -> TypeSupport

Look up which element types `backend` (`"CUDA"`, `"AMDGPU"`, `"Metal"`, `"oneAPI"` or
`"CPU"`) supports in kernels, from a table in GPUSelect. Nothing is loaded, compiled or run,
so this is instant and works before the backend package is installed.

Without arguments it uses the GPU backend that [`Backend`](@ref) would select, or the one
the detected hardware needs when none is installed. `version` defaults to the version in
the active manifest (KernelAbstractions for `"CPU"`); with `version=nothing` the table
assumes the latest release.

Some types depend on the device, such as `Float64` on Intel GPUs; those are listed in
`device_dependent` with the condition. Types the table has no entry for are listed in
`unknown`. Use [`test_types`](@ref) to find out for certain on a device.

```julia
julia> GPUSelect.known_types("Metal")
Metal supports: Float16, Float32, Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, UInt64
  Float64 not supported: Apple GPUs have no Float64
  Int128 not supported: Metal has no 128-bit integers
  UInt128 not supported: Metal has no 128-bit integers
  BFloat16 depends on device: needs Metal 3.1 (macOS 14) or newer

julia> :Float64 in GPUSelect.known_types("CUDA")
true
```
"""
function known_types(backend::Union{AbstractString,Nothing} = selected_gpu_backend();
                     version = nothing_or_installed(backend))
    backend === nothing && error("No GPU detected and no GPU backend installed; pass a backend name, e.g. known_types(\"CUDA\").")
    backend = String(backend)
    backend ∈ GPU_BACKENDS || backend == "CPU" ||
        error("Unknown backend \"$backend\". Choose from: $(join(GPU_BACKENDS, ", ")), CPU")
    version = version === nothing ? nothing : VersionNumber(version)
    supported, unsupported, device_dependent, unknown = Symbol[], Pair{Symbol,String}[], Pair{Symbol,String}[], Symbol[]
    for T in TYPE_NAMES
        i = findfirst(r -> T in r.types && rule_applies(r, backend, version), TYPE_RULES)
        if i === nothing
            push!(unknown, T)
        else
            r = TYPE_RULES[i]
            r.status === :supported ? push!(supported, T) :
            r.status === :unsupported ? push!(unsupported, T => r.note) : push!(device_dependent, T => r.note)
        end
    end
    return TypeSupport(backend, version, nothing, supported, unsupported, device_dependent, unknown)
end

function nothing_or_installed(backend)
    backend === nothing && return nothing
    backend == "CPU" && return pkgversion(KernelAbstractions)
    return manifest_version(backend)
end

# ---------------------------------------------------------------------------------------
# Testing on a device
# ---------------------------------------------------------------------------------------

const BFLOAT16S_UUID = Base.UUID("ab4f0b2a-ad5b-11e8-123f-65d77653426b")

# BFloat16 arithmetic comes from BFloat16s.jl. AMDGPU, Metal and CUDA load it, so test
# BFloat16 whenever it is loaded, without making it a dependency of GPUSelect.
function bfloat16_type()
    m = get(Base.loaded_modules, Base.PkgId(BFLOAT16S_UUID, "BFloat16s"), nothing)
    return m === nothing ? nothing : getfield(m, :BFloat16)
end

"""
    candidate_types() -> Vector{DataType}

The element types [`test_types`](@ref) tests by default: `Float16`, `BFloat16` (when
BFloat16s.jl is loaded), `Float32`, `Float64`, and the signed and unsigned integers from 8
to 128 bits.
"""
function candidate_types()
    bf16 = bfloat16_type()
    return DataType[T === :BFloat16 ? bf16 : getfield(Base, T) for T in TYPE_NAMES
                    if T !== :BFloat16 || bf16 !== nothing]
end

@kernel function _type_probe!(y, @Const(x))
    i = @index(Global)
    @inbounds y[i] = x[i] * x[i] + x[i]
end

# Store, compute and read back a few values of type T on `backend`; throws when that fails.
function probe_type(backend, ::Type{T}) where {T}
    n = 32
    host = T[T(mod1(i, 7)) for i in 1:n]   # small integers, exact in every tested type
    x = KernelAbstractions.allocate(backend, T, n)
    y = KernelAbstractions.allocate(backend, T, n)
    KernelAbstractions.copyto!(backend, x, host)
    _type_probe!(backend, n)(y, x; ndrange=n)
    KernelAbstractions.synchronize(backend)
    Array(y) == host .* host .+ host || error("kernel returned wrong values")
    return nothing
end

function first_line(e)
    line = first(split(strip(sprint(showerror, e)), '\n'))
    return length(line) > 200 ? first(line, 200) * "…" : String(line)
end

function type_support(mod::Module, device::Union{Int,Nothing}, types)
    name = String(nameof(mod))
    backend = Base.invokelatest(getfield(mod, BACKEND_CONSTRUCTOR[name]))
    previous = nothing
    if device !== nothing
        isdefined(KernelAbstractions, :device!) ||
            error("Selecting a device needs a newer KernelAbstractions version.")
        previous = Base.invokelatest(KernelAbstractions.device, backend)
        Base.invokelatest(KernelAbstractions.device!, backend, device)
    end
    try
        index = something(device, isdefined(KernelAbstractions, :device) ?
                                   Base.invokelatest(KernelAbstractions.device, backend) : 1)
        supported, unsupported = Symbol[], Pair{Symbol,String}[]
        for T in types
            if T === Float64 && !Base.invokelatest(KernelAbstractions.supports_float64, backend)
                push!(unsupported, :Float64 => "backend reports no Float64 support")
                continue
            end
            try
                Base.invokelatest(probe_type, backend, T)
                push!(supported, nameof(T))
            catch e
                push!(unsupported, nameof(T) => first_line(e))
            end
        end
        label = mod === KernelAbstractions ? "CPU" : name
        return TypeSupport(label, pkgversion(mod), index, supported, unsupported, Pair{Symbol,String}[], Symbol[])
    finally
        previous === nothing || Base.invokelatest(KernelAbstractions.device!, backend, previous)
    end
end

"""
    test_types(target::Symbol = :GPU; device=nothing, types=candidate_types(),
               fallback=true, install=get_install_preference()) -> TypeSupport
    test_types(d::DeviceInfo; types=candidate_types()) -> TypeSupport

Find out which element types a device supports by trying them. For every type in `types`
this compiles and runs a small KernelAbstractions kernel that stores, multiplies, adds and
reads back values. A type is supported when that works and gives the right result.

The answer reflects the installed version of the backend package, the driver and the
device, which [`known_types`](@ref) can only approximate. It loads the backend and takes a
few seconds, since each type compiles its own kernel, so use `known_types` when the table
answers your question.

`device` is the 1-based device index (see [`devices`](@ref)); by default the current device
is tested. `fallback` and `install` work as in `Backend()`.

```julia
julia> GPUSelect.test_types()
Metal v1.12.1 device 1 supports: Float16, BFloat16, Float32, Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, UInt64
  Float64 not supported: backend reports no Float64 support
  Int128 not supported: Metal does not support Int128 values, try using Int64 instead
  UInt128 not supported: Metal does not support UInt128 values, try using UInt64 instead
```
"""
function test_types(target::Symbol = :GPU; device::Union{Int,Nothing} = nothing,
                    types = candidate_types(), fallback::Bool = true,
                    install::Bool = get_install_preference())
    target === :Lava && error("GPUSelect.test_types does not support Lava yet.")
    mod = backend_module(target, install, fallback)
    return type_support(mod, mod === KernelAbstractions ? nothing : device, types)
end

function test_types(d::DeviceInfo; types = candidate_types())
    d.backend == "CPU" && return type_support(KernelAbstractions, nothing, types)
    haskey(BACKENDS, d.backend) || error("Unknown backend \"$(d.backend)\".")
    mod = Base.require(Base.PkgId(BACKENDS[d.backend], d.backend))
    return type_support(mod, d.index, types)
end
