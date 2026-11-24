module GPUSelectoneAPIExt

using oneAPI
using GPUSelect: GPUSelect, DeviceInfo, _try, _int

function GPUSelect.device_info(::Val{:oneAPI})
    oneAPI.functional() || error("oneAPI.jl is not functional on this system")
    return [begin
        props = _try(() -> oneAPI.oneL0.properties(dev))
        prop(f) = props === nothing ? nothing : _try(() -> getproperty(props, f))
        DeviceInfo(;
            backend="oneAPI", index=i, name=something(prop(:name), "Intel GPU $i"),
            total_memory=_int(_try(() -> sum(m -> m.totalSize, oneAPI.oneL0.memory_properties(dev)))),
            max_threads_per_block=_int(_try(() -> oneAPI.oneL0.compute_properties(dev).maxTotalGroupSize)),
            compute_units=_try(() -> Int(props.numSlices * props.numSubslicesPerSlice * props.numEUsPerSubslice)),
            uuid=_try(() -> string(props.uuid)),
        )
    end for (i, dev) in enumerate(oneAPI.devices(oneAPI.driver()))]
end

end
