module GPUSelect

using Adapt
import KernelAbstractions
import Libdl
using Pkg
using Preferences
using TOML

const BACKENDS = Dict{String,Base.UUID}(
    "CUDA"   => Base.UUID("052768ef-5323-5732-b1bb-66c8b64840ba"),
    "AMDGPU" => Base.UUID("21141c5a-9bdb-4563-92ae-f87d6854732e"),
    "Lava"   => Base.UUID("3a680b1f-cb25-4bee-9cf7-bc880b76dc8c"),
    "Metal"  => Base.UUID("dde4c033-4e86-420c-a63e-0dd931031962"),
    "oneAPI" => Base.UUID("8f75cd03-7ff8-4ecb-9b8f-daf728133b1b"),
)

const GPU_BACKENDS = ("CUDA", "AMDGPU", "Metal", "oneAPI")

const BACKEND_CONSTRUCTOR = Dict{String,Symbol}(
    "KernelAbstractions" => :CPU,
    "CUDA"               => :CUDABackend,
    "AMDGPU"             => :ROCBackend,
    "Metal"              => :MetalBackend,
    "oneAPI"             => :oneAPIBackend,
    "Lava"               => :LavaBackend,
)

const STORAGE_TYPE = Dict{String,Symbol}(
    "KernelAbstractions" => :Array,
    "CUDA"               => :CuArray,
    "AMDGPU"             => :ROCArray,
    "Metal"              => :MtlArray,
    "oneAPI"             => :oneArray,
    "Lava"               => :LavaArray,
)

"""
    find_gpu_pkgid() -> Base.PkgId or nothing

Probe driver libraries to find what GPU hardware is present and return its
backend's `PkgId`, or `nothing` if nothing is detected.
"""
function find_gpu_pkgid()::Union{Base.PkgId,Nothing}
    Sys.isapple() && Sys.ARCH === :aarch64 && return Base.PkgId(BACKENDS["Metal"], "Metal")
    cuda_lib = Sys.iswindows() ?
        Libdl.find_library(["nvcuda"]) :
        Libdl.find_library(["libcuda", "libcuda.so.1", "libcuda.so"])
    !isempty(cuda_lib) && return Base.PkgId(BACKENDS["CUDA"], "CUDA")
    amd_lib = Sys.iswindows() ?
        Libdl.find_library(["amdhip64"]) :
        Libdl.find_library(["libamdhip64", "libhip_hcc"])
    (!isempty(amd_lib) || (Sys.islinux() && isfile("/dev/kfd"))) &&
        return Base.PkgId(BACKENDS["AMDGPU"], "AMDGPU")
    ze_lib = Libdl.find_library(["libze_loader", "ze_loader"])
    !isempty(ze_lib) && return Base.PkgId(BACKENDS["oneAPI"], "oneAPI")
    vk_lib = Sys.iswindows() ?
        Libdl.find_library(["vulkan-1"]) :
        Libdl.find_library(["libvulkan", "libvulkan.so.1", "libMoltenVK"])
    !isempty(vk_lib) && return Base.PkgId(BACKENDS["Lava"], "Lava")
    return nothing
end

"""
    current_backend_module(target::Symbol) -> Module

Load and return the module for the given backend target.

- `:GPU`: loads the first available native GPU backend (CUDA, AMDGPU, Metal, oneAPI).
          When multiple are installed, prefers the one `find_gpu_pkgid()` recommends.
- `:Lava`: loads Lava.

Errors if the requested backend is not installed in the active project.
"""
function current_backend_module(target::Symbol)::Module
    target === :CPU && return KernelAbstractions
    if target === :GPU
        found = filter(n -> n ∈ GPU_BACKENDS, installed_backends())
        isempty(found) && error(
            "No GPU backend installed. " *
            "Run GPUSelect.auto_install!() or Pkg.add(\"CUDA\") / Pkg.add(\"AMDGPU\") etc.")
        detected = find_gpu_pkgid()
        name = detected !== nothing && detected.name ∈ found ? detected.name : first(found)
        return Base.require(Base.PkgId(BACKENDS[name], name))
    elseif target === :Lava
        "Lava" ∈ installed_backends() ||
            error("Lava is not installed. Run Pkg.add(\"Lava\") first.")
        return Base.require(Base.PkgId(BACKENDS["Lava"], "Lava"))
    end
    error("Unknown target :$target. Use :GPU, :CPU, or :Lava.")
end

function get_backend_func(target::Symbol, table::Dict{String,Symbol}, install::Bool, fallback::Bool)::Type
    target ∈ (:CPU, :GPU, :Lava) || error("Unknown target :$target. Use :GPU, :CPU, or :Lava.")
    if install && target !== :CPU
        needed = target === :Lava ? ("Lava",) : GPU_BACKENDS
        if !any(n -> n ∈ needed, installed_backends())
            auto_install!()
        end
    end
    mod = try
        current_backend_module(target)
    catch e
        fallback || rethrow()
        @warn "GPUSelect: $target unavailable, falling back to CPU." exception=e
        KernelAbstractions
    end
    name = String(nameof(mod))
    sym = get(table, name, nothing)
    sym === nothing && error("GPUSelect: no entry registered for module $name")
    return getfield(mod, sym)
end

function Backend end

"""
    Backend(target::Symbol = :GPU; fallback=true, install=get_install_preference()) -> KernelAbstractions.Backend

Return a KernelAbstractions backend for the given target.

- `:CPU`: always returns `KernelAbstractions.CPU()`, no package needed.
- `:GPU`: loads and returns the native GPU backend (CUDA, AMDGPU, Metal, or oneAPI).
          Falls back to `CPU()` with a warning when `fallback=true`.
- `:Lava`: loads and returns the Lava (Vulkan) backend. On systems without a hardware
           Vulkan GPU, Lava itself may fall back to lavapipe. Falls back to `CPU()` if
           Lava is unavailable.

**`Backend()` never auto-installs a package by default.** If no matching backend is
present in the active project, it errors (or falls back to CPU when `fallback=true`).
Pass `install=true` explicitly, or opt in once per environment with
`set_install_preference!(true)`, to have `auto_install!()` run automatically.
"""
function Backend(target::Symbol = :GPU; fallback::Bool = true, install::Bool = get_install_preference())
    T = get_backend_func(target, BACKEND_CONSTRUCTOR, install, fallback)
    return Base.invokelatest(T)
end

function Storage end

"""
    Storage(target::Symbol = :GPU; fallback=true, install=get_install_preference()) -> Type

Return the array type for the given target.

- `:CPU`: returns `Array`.
- `:GPU`: returns the native GPU array type (`CuArray`, `ROCArray`, `MtlArray`, or `oneArray`).
- `:Lava`: returns `LavaArray`.

Falls back to `Array` with a warning when `fallback=true` and the backend is unavailable.
Use with `adapt` or allocate directly: `Storage()(undef, 1024)`.

Like `Backend`, `Storage()` does not auto-install by default. Pass `install=true` or
set `set_install_preference!(true)` to opt in.
"""
function Storage(target::Symbol = :GPU; fallback::Bool = true, install::Bool = get_install_preference())
    get_backend_func(target, STORAGE_TYPE, install, fallback)
end

"""
    get_install_preference() -> Bool

Return the current auto-install preference (default `false`).
When `true`, `Backend()` will call `auto_install!()` automatically on the first
call if no matching backend is found in the active project.
"""
function get_install_preference()::Bool
    Preferences.load_preference(GPUSelect, "auto_install", false)
end

"""
    set_install_preference!(val::Bool)

Persist `val` to `LocalPreferences.toml`.  When `true`, `Backend()` auto-installs
the detected GPU backend on first use instead of erroring or falling back to CPU.
"""
function set_install_preference!(val::Bool)
    @set_preferences!("auto_install" => val)
    @info "GPUSelect: auto_install preference set to $val."
end

"""
    set_backend!(name)

Write `name` (e.g. `"CUDA"`, `"AMDGPU"`, `"Lava"`) to `LocalPreferences.toml`
so `@load_backend()` picks it up automatically. Call once per environment.
"""
function set_backend!(name::String)
    name ∈ keys(BACKENDS) ||
        error("Unknown backend \"$name\". Choose from: $(join(sort(collect(keys(BACKENDS))), ", "))")
    @set_preferences!("backend" => name)
    @info "GPUSelect: preference set to \"$name\". Restart Julia for it to take effect."
end

"""
    clear_backend!()

Remove the backend preference so `@load_backend()` falls back to manifest scanning.
"""
function clear_backend!()
    @delete_preferences!("backend")
    @info "GPUSelect: backend preference cleared."
end

"""
    installed_backends() -> Vector{String}

Return the names of known GPU backends present in the active project's manifest.
"""
function installed_backends()::Vector{String}
    proj = Base.active_project()
    proj === nothing && return String[]
    manifest = joinpath(dirname(proj), "Manifest.toml")
    isfile(manifest) || return String[]
    data = TOML.parsefile(manifest)
    deps = haskey(data, "deps") ? data["deps"] : data  # v2 vs v1 manifest format
    return sort([name for name in keys(BACKENDS) if haskey(deps, name)])
end

"""
    auto_install!(; force=false)

Detect the GPU on this machine and `Pkg.add` the matching backend into the active
environment. Errors if a backend is already installed (pass `force=true` to skip).
Call once per environment, not in every script.
"""
function auto_install!(; force::Bool=false)
    already = installed_backends()
    if !isempty(already) && !force
        error("Backend(s) already installed: $(join(already, ", ")). Pass force=true to reinstall.")
    end
    pkgid = find_gpu_pkgid()
    pkgid === nothing && error(
        "Could not detect a supported GPU. " *
        "Install a backend manually: Pkg.add(\"CUDA\"), Pkg.add(\"AMDGPU\"), etc.")
    @info "GPUSelect: detected $(pkgid.name). Running Pkg.add(\"$(pkgid.name)\")..."
    Pkg.add(pkgid.name)
    @info "GPUSelect: done. Call Backend(:GPU) or Backend(:Lava) in your scripts."
end

"""
    test_backends()

Live smoke-test of backend detection and loading in the calling environment.
Call this from the REPL (not via `Pkg.test`) in a project that has at least one
GPU backend installed to verify everything wires up correctly.

Prints a human-readable report and returns `true` if all attempted loads succeeded.
"""
function test_backends()::Bool
    bar = "─" ^ 60
    println("\n$bar")
    println("  GPUSelect: Backend Test")
    println("  $(Sys.iswindows() ? "Windows" : Sys.isapple() ? "macOS" : "Linux") / $(Sys.ARCH)")
    println(bar)

    pkgid  = find_gpu_pkgid()
    bs     = installed_backends()
    println("  Detected hardware:  $(pkgid === nothing ? "none" : pkgid.name)")
    println("  Installed backends: $(isempty(bs) ? "none" : join(bs, ", "))")
    println()

    results = Dict{String,Union{NamedTuple,Exception}}()

    for target in (:CPU, :GPU, :Lava)
        try
            b = Backend(target; fallback=false, install=false)
            s = Storage(target; fallback=false, install=false)
            results[string(target)] = (backend=string(typeof(b)), storage=string(s))
        catch e
            results[string(target)] = e
        end
    end

    ok = true
    for (label, r) in sort(collect(results); by=first)
        if r isa Exception
            println("  :$label  FAILED: $r")
            ok = false
        else
            println("  :$label  Backend=$(r.backend)  Storage=$(r.storage)")
        end
    end
    println(bar, "\n")
    return ok
end

"""
    @load_backend()

Scan the active project manifest (or read the preference set via `set_backend!`)
and expand to `using <BackendName>`. After this, `Backend(:GPU)` or `Backend(:Lava)`
return immediately without triggering another load.

Errors if zero or multiple backends are installed and no preference is set.
"""
macro load_backend()
    preferred = Preferences.load_preference(GPUSelect, "backend", nothing)
    name = if preferred !== nothing
        preferred ∈ keys(BACKENDS) ||
            error("Unknown backend in preferences: \"$preferred\". " *
                  "Fix with GPUSelect.set_backend!(\"CUDA\") etc.")
        preferred
    else
        found = installed_backends()
        isempty(found) &&
            error("No GPU backend found in the manifest. " *
                  "Run GPUSelect.auto_install!() or Pkg.add(\"CUDA\") / Pkg.add(\"AMDGPU\") etc.")
        length(found) > 1 &&
            error("Multiple GPU backends found: $(join(found, ", ")). " *
                  "Pick one with GPUSelect.set_backend!(\"CUDA\") or be explicit with `using CUDA`.")
        only(found)
    end
    return Expr(:using, Expr(:., Symbol(name)))
end

end
