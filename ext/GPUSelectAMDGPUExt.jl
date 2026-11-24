module GPUSelectAMDGPUExt

using AMDGPU
using GPUSelect: GPUSelect, DeviceInfo, _try, _int, _format_pci

function GPUSelect.device_info(::Val{:AMDGPU})
    AMDGPU.functional() || error("AMDGPU.jl is not functional on this system")
    return [begin
        props = _try(() -> AMDGPU.HIP.properties(dev))
        prop(f) = props === nothing ? nothing : _try(() -> getproperty(props, f))
        DeviceInfo(;
            backend="AMDGPU", index=i, name=AMDGPU.HIP.name(dev),
            total_memory=_int(prop(:totalGlobalMem)),
            max_threads_per_block=_int(prop(:maxThreadsPerBlock)),
            compute_units=_int(prop(:multiProcessorCount)),
            capability=_try(() -> AMDGPU.HIP.gcn_arch(dev)),
            pci_bus_id=_format_pci(prop(:pciDomainID), prop(:pciBusID), prop(:pciDeviceID)),
        )
    end for (i, dev) in enumerate(AMDGPU.devices())]
end

end
