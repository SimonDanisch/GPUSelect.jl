using Test
using GPUSelect
import KernelAbstractions as KA
using KernelAbstractions: @kernel, @index

@kernel function fill_kernel!(arr, val)
    i = @index(Global)
    arr[i] = val
end

function run_fill_test(backend, AT)
    n = 1024
    arr = AT{Float32}(undef, n)
    fill_kernel!(backend, 64)(arr, 2.0f0, ndrange=n)
    KA.synchronize(backend)
    return all(==(2.0f0), Array(arr))
end

@testset "GPUSelect" begin

    @testset "find_gpu_pkgid" begin
        pkgid = GPUSelect.find_gpu_pkgid()
        @test pkgid === nothing || pkgid isa Base.PkgId
        if pkgid !== nothing
            @test pkgid.name ∈ keys(GPUSelect.BACKENDS)
            @test pkgid.uuid == GPUSelect.BACKENDS[pkgid.name]
        end
        @info "find_gpu_pkgid → $(pkgid === nothing ? "nothing" : pkgid.name)"
    end

    @testset "installed_backends" begin
        bs = GPUSelect.installed_backends()
        @test bs isa Vector{String}
        @test issorted(bs)
        for name in bs
            @test name ∈ keys(GPUSelect.BACKENDS)
        end
        @info "installed_backends → $(isempty(bs) ? "none" : join(bs, ", "))"
    end

    @testset "Preferences" begin
        @test GPUSelect.get_install_preference() isa Bool
        GPUSelect.set_backend!("CUDA")
        GPUSelect.clear_backend!()
        @test_throws Exception GPUSelect.set_backend!("NotABackend")
        @test_throws Exception GPUSelect.set_install_preference!(1)  # wrong type
    end

    @testset "Backend(:CPU)" begin
        @test GPUSelect.Backend(:CPU) isa KA.CPU
        @test GPUSelect.Backend(:CPU; fallback=false) isa KA.CPU
        @test GPUSelect.Backend(:CPU; install=false) isa KA.CPU
    end

    @testset "Backend: unknown target always throws" begin
        @test_throws Exception GPUSelect.Backend(:Vulkan)
        @test_throws Exception GPUSelect.Backend(:CUDA)
    end

    @testset "Backend(:GPU)" begin
        bs = GPUSelect.installed_backends()
        gpu_installed = any(n -> n ∈ GPUSelect.GPU_BACKENDS, bs)
        b = GPUSelect.Backend(:GPU; fallback=true)
        @test b isa KA.Backend
        if gpu_installed
            @test !(b isa KA.CPU)
            @test GPUSelect.Backend(:GPU; fallback=false) isa KA.Backend
        else
            @test b isa KA.CPU
            @test_throws Exception GPUSelect.Backend(:GPU; fallback=false)
        end
        @info "Backend(:GPU) → $(typeof(b))  [GPU $(gpu_installed ? "installed" : "not installed")]"
    end

    @testset "Backend(:Lava)" begin
        bs = GPUSelect.installed_backends()
        lava_installed = "Lava" ∈ bs
        b = GPUSelect.Backend(:Lava; fallback=true)
        @test b isa KA.Backend
        if lava_installed
            @test !(b isa KA.CPU)
            @test GPUSelect.Backend(:Lava; fallback=false) isa KA.Backend
        else
            @test b isa KA.CPU
            @test_throws Exception GPUSelect.Backend(:Lava; fallback=false)
        end
        @info "Backend(:Lava) → $(typeof(b))  [Lava $(lava_installed ? "installed" : "not installed")]"
    end

    @testset "devices" begin
        cpu = only(GPUSelect.devices(:CPU))
        @test cpu isa GPUSelect.DeviceInfo
        @test cpu.backend == "CPU" && cpu.index == 1
        @test cpu.compute_units == Sys.CPU_THREADS
        @test cpu.total_memory > 0
        @test occursin("CPU device 1", sprint(show, MIME("text/plain"), cpu))
        @test_throws Exception GPUSelect.devices(:Lava)
        @test_throws Exception GPUSelect.devices(:Vulkan)

        bs = GPUSelect.installed_backends()
        gpu_installed = any(n -> n ∈ GPUSelect.GPU_BACKENDS, bs)
        devs = GPUSelect.devices(:GPU; fallback=true)
        @test devs isa Vector{GPUSelect.DeviceInfo}
        @test !isempty(devs)
        @test [d.index for d in devs] == 1:length(devs)
        if gpu_installed
            @test GPUSelect.devices(:GPU; fallback=false) == devs
            @test all(d -> d.backend ∈ GPUSelect.GPU_BACKENDS, devs)
            @test all(d -> d.max_threads_per_block !== nothing && d.max_threads_per_block > 0, devs)
        else
            @test only(devs).backend == "CPU"
            @test_throws Exception GPUSelect.devices(:GPU; fallback=false)
        end
        @info "devices() → $(join((d.name for d in devs), ", "))"
    end

    # Every type the table calls supported or unsupported must behave that way on the device.
    function check_table(table, tested)
        for T in table.supported
            haskey(Dict(tested.unsupported), T) && @error "known_types says $T works on $(table.backend), test_types disagrees" Dict(tested.unsupported)[T]
            @test !haskey(Dict(tested.unsupported), T)
        end
        for (T, _) in table.unsupported
            @test !(T in tested)
        end
    end

    @testset "known_types" begin
        for b in ("CUDA", "AMDGPU", "Metal", "oneAPI", "CPU")
            ts = GPUSelect.known_types(b; version=nothing)
            @test ts.backend == b && ts.device === nothing
            all = vcat(ts.supported, first.(ts.unsupported), first.(ts.device_dependent), ts.unknown)
            @test sort(all) == sort(collect(GPUSelect.TYPE_NAMES))   # each type listed exactly once
            @test :Float32 in ts && Int32 in ts
        end
        @test !(Float64 in GPUSelect.known_types("Metal"))
        @test Float64 in GPUSelect.known_types("CUDA")
        @test :Float64 in first.(GPUSelect.known_types("oneAPI").device_dependent)
        @test Int128 in GPUSelect.known_types("CPU"; version=v"0.9.41")      # native Julia CPU backend
        @test !(Int128 in GPUSelect.known_types("CPU"; version=v"0.10.9"))   # POCL CPU backend
        @test GPUSelect.known_types("CPU").version == pkgversion(KA)
        @test occursin("not supported", sprint(show, MIME("text/plain"), GPUSelect.known_types("Metal")))
        @test_throws Exception GPUSelect.known_types("Vulkan")
        bf16 = GPUSelect.bfloat16_type()   # BFloat16 is only returned when BFloat16s.jl is loaded
        @test GPUSelect.float_types(GPUSelect.known_types("Metal")) ==
              (bf16 === nothing ? [Float16, Float32] : [Float16, bf16, Float32])
        @test GPUSelect.float_types(GPUSelect.known_types("Metal"); certain=true) == [Float16, Float32]
        @test GPUSelect.float_types(GPUSelect.known_types("oneAPI"); certain=true) == [Float32]
        @test GPUSelect.float_types(GPUSelect.known_types("oneAPI")) == [Float16, Float32, Float64]
        @test GPUSelect.integer_types(GPUSelect.known_types("CUDA"; version=v"5.9.0"); certain=true) ==
              [Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, UInt64]
        @test Int128 in GPUSelect.integer_types(GPUSelect.known_types("CUDA"; version=v"5.9.0"))   # unknown, so possible
        @test GPUSelect.integer_types(GPUSelect.known_types("Metal")) ==
              [Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, UInt64]
        @test GPUSelect.integer_types(GPUSelect.known_types("CUDA"; version=nothing)) ==
              [Int8, Int16, Int32, Int64, Int128, UInt8, UInt16, UInt32, UInt64, UInt128]
        @test all(T -> T <: AbstractFloat, GPUSelect.float_types(GPUSelect.known_types("CPU")))
        @test GPUSelect.manifest_version("NotAPackage") === nothing
    end

    @testset "test_types" begin
        @test Float32 in GPUSelect.candidate_types() && Int64 in GPUSelect.candidate_types()
        ts = GPUSelect.test_types(:CPU)
        @test ts isa GPUSelect.TypeSupport
        @test ts.backend == "CPU" && ts.device == 1
        @test ts.version == pkgversion(KA)
        @test Float32 in ts && Int32 in ts && UInt8 in ts
        @test Float32 in GPUSelect.float_types(ts) && Int32 in GPUSelect.integer_types(ts)
        @test all(T -> T in ts, vcat(GPUSelect.float_types(ts), GPUSelect.integer_types(ts)))
        @test length(ts.supported) + length(ts.unsupported) == length(GPUSelect.candidate_types())
        if GPUSelect.bfloat16_type() === nothing
            @test ts.unknown == [:BFloat16]
            @test occursin("using BFloat16s", sprint(show, MIME("text/plain"), ts))
        end
        @test occursin("CPU", sprint(show, MIME("text/plain"), ts))
        @test GPUSelect.test_types(:CPU; types=[Float32]).supported == [:Float32]
        @test GPUSelect.test_types(only(GPUSelect.devices(:CPU)); types=[Int32]).supported == [:Int32]
        @test_throws Exception GPUSelect.test_types(:Lava)
        check_table(GPUSelect.known_types("CPU"), ts)
@test !occursin("not tested", Dict(ts.unsupported)[:Int128])   # fails with an error, so it is tested
        @test GPUSelect.crashing_types("oneAPI", v"2.10.0") |> keys |> collect |> sort == [:Int128, :UInt128]
        @test isempty(GPUSelect.crashing_types("Metal", nothing))
        @test GPUSelect.first_line(ErrorException("Some header\nReason: unsupported call\nmore")) ==
              "ErrorException: unsupported call"
        @test GPUSelect.first_line(ErrorException("plain message")) == "plain message"

        bs = GPUSelect.installed_backends()
        if any(n -> n ∈ GPUSelect.GPU_BACKENDS, bs)
            gts = GPUSelect.test_types(:GPU; fallback=false)
            @test gts.backend ∈ GPUSelect.GPU_BACKENDS
            @test Float32 in gts && Int32 in gts
            @test GPUSelect.test_types(first(GPUSelect.devices(:GPU; fallback=false))).supported == gts.supported
            check_table(GPUSelect.known_types(gts.backend; version=gts.version), gts)
            @info "test_types() → $(join(gts.supported, ", "))"
        end
    end

    @testset "Computation: Backend()" begin
        bs = GPUSelect.installed_backends()
        gpu_installed = any(n -> n ∈ GPUSelect.GPU_BACKENDS, bs)
        if gpu_installed
            backend = GPUSelect.Backend()
            AT      = GPUSelect.Storage()
            @test run_fill_test(backend, AT)
            @info "Backend() computation ✓  $(typeof(backend))"
        else
            @info "Backend() computation skipped, no GPU backend installed"
        end
    end

    @testset "Computation: Backend(:Lava)" begin
        bs = GPUSelect.installed_backends()
        if "Lava" ∈ bs
            backend = GPUSelect.Backend(:Lava)
            AT      = GPUSelect.Storage(:Lava)
            @test run_fill_test(backend, AT)
            @info "Backend(:Lava) computation ✓  $(typeof(backend))"
        else
            @info "Backend(:Lava) computation skipped, Lava not installed"
        end
    end

end
