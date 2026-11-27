module GPUSelectCUDAExt

using CUDA
using GPUSelect: GPUSelect, DeviceInfo, _try, _int, _format_pci

function GPUSelect.device_info(::Val{:CUDA})
    CUDA.functional() || error("CUDA.jl is not functional on this system")
    attr(dev, a) = _int(_try(() -> CUDA.attribute(dev, a)))
    return [DeviceInfo(;
        backend="CUDA", index=i, name=CUDA.name(dev),
        total_memory=_int(_try(() -> CUDA.totalmem(dev))),
        max_threads_per_block=attr(dev, CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK),
        compute_units=attr(dev, CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT),
        capability=_try(() -> CUDA.capability(dev)),
        uuid=_try(() -> string(CUDA.uuid(dev))),
        pci_bus_id=_format_pci(attr(dev, CUDA.DEVICE_ATTRIBUTE_PCI_DOMAIN_ID),
                               attr(dev, CUDA.DEVICE_ATTRIBUTE_PCI_BUS_ID),
                               attr(dev, CUDA.DEVICE_ATTRIBUTE_PCI_DEVICE_ID)),
    ) for (i, dev) in enumerate(CUDA.devices())]
end

end
