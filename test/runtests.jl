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
