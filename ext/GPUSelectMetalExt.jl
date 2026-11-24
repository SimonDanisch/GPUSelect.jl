module GPUSelectMetalExt

using Metal
using GPUSelect: GPUSelect, DeviceInfo, _try, _int

function GPUSelect.device_info(::Val{:Metal})
    Metal.functional() || error("Metal.jl is not functional on this system")
    return [DeviceInfo(;
        backend="Metal", index=i, name=String(dev.name),
        total_memory=_int(_try(() -> dev.recommendedMaxWorkingSetSize)),
        max_threads_per_block=_int(_try(() -> dev.maxThreadsPerThreadgroup.width)),
        compute_units=_int(_try(() -> Metal.num_gpu_cores())),
        uuid=_try(() -> string(dev.registryID; base=16)),
    ) for (i, dev) in enumerate(Metal.devices())]
end

end
