/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "kvCacheManagerV2Utils.h"
#include "tensorrt_llm/batch_manager/kvCacheManagerV2Utils.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/torchView.h"
#include <ATen/ATen.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>
#include <torch/extension.h>

namespace tr = tensorrt_llm::runtime;
namespace nb = nanobind;

using SizeType32 = tensorrt_llm::runtime::SizeType32;

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

std::optional<tensorrt_llm::runtime::ITensor::UniquePtr> from_torch(std::optional<at::Tensor> torchPtr)
{
    if (torchPtr)
    {
        return tr::TorchView::of(torchPtr.value());
    }
    return std::nullopt;
}

void KVCacheManagerV2UtilsBindings::initBindings(nb::module_& module)
{
    // Bind DiskAddress struct
    nb::class_<DiskAddress>(module, "DiskAddress")
        .def(nb::init<int, ssize_t>(), nb::arg("fd"), nb::arg("pos"))
        .def_rw("fd", &DiskAddress::fd)
        .def_rw("pos", &DiskAddress::pos);

    // Bind Task template instantiations
    nb::class_<Task<DiskAddress, DiskAddress>>(module, "DiskToDiskTask")
        .def(nb::init<DiskAddress, DiskAddress>(), nb::arg("dst"), nb::arg("src"))
        .def_rw("dst", &Task<DiskAddress, DiskAddress>::dst)
        .def_rw("src", &Task<DiskAddress, DiskAddress>::src);

    nb::class_<Task<MemAddress, DiskAddress>>(module, "DiskToHostTask")
        .def(nb::init<MemAddress, DiskAddress>(), nb::arg("dst"), nb::arg("src"))
        .def_rw("dst", &Task<MemAddress, DiskAddress>::dst)
        .def_rw("src", &Task<MemAddress, DiskAddress>::src);

    nb::class_<Task<DiskAddress, MemAddress>>(module, "HostToDiskTask")
        .def(nb::init<DiskAddress, MemAddress>(), nb::arg("dst"), nb::arg("src"))
        .def_rw("dst", &Task<DiskAddress, MemAddress>::dst)
        .def_rw("src", &Task<DiskAddress, MemAddress>::src);

    nb::class_<Task<MemAddress, MemAddress>>(module, "MemToMemTask")
        .def(nb::init<MemAddress, MemAddress>(), nb::arg("dst"), nb::arg("src"))
        .def_rw("dst", &Task<MemAddress, MemAddress>::dst)
        .def_rw("src", &Task<MemAddress, MemAddress>::src);

    nb::class_<BlockIndices>(module, "BlockIndices")
        .def(nb::init<MemAddress, SizeType32>(), nb::arg("addr"), nb::arg("length"))
        .def_rw("addr", &BlockIndices::addr)
        .def_rw("length", &BlockIndices::length);

    // Bind copy functions
    module.def(
        "copy_disk_to_disk",
        [](std::vector<Task<DiskAddress, DiskAddress>> const& tasks, ssize_t numBytes, uintptr_t stream) -> int
        { return copyDiskToDisk(tasks, numBytes, reinterpret_cast<CUstream>(stream)); },
        nb::arg("tasks"), nb::arg("num_bytes"), nb::arg("stream"), nb::call_guard<nb::gil_scoped_release>(),
        "Copy data from disk to disk using CUDA host function");

    module.def(
        "copy_disk_to_host",
        [](std::vector<Task<MemAddress, DiskAddress>> const& tasks, ssize_t numBytes, uintptr_t stream) -> int
        { return copyDiskToHost(tasks, numBytes, reinterpret_cast<CUstream>(stream)); },
        nb::arg("tasks"), nb::arg("num_bytes"), nb::arg("stream"), nb::call_guard<nb::gil_scoped_release>(),
        "Copy data from disk to host using CUDA host function");

    module.def(
        "copy_host_to_disk",
        [](std::vector<Task<DiskAddress, MemAddress>> const& tasks, ssize_t numBytes, uintptr_t stream) -> int
        { return copyHostToDisk(tasks, numBytes, reinterpret_cast<CUstream>(stream)); },
        nb::arg("tasks"), nb::arg("num_bytes"), nb::arg("stream"), nb::call_guard<nb::gil_scoped_release>(),
        "Copy data from host to disk using CUDA host function");

    module.def(
        "copy_host_to_host",
        [](std::vector<Task<MemAddress, MemAddress>> const& tasks, ssize_t numBytes, uintptr_t stream) -> int
        { return copyHostToHost(tasks, numBytes, reinterpret_cast<CUstream>(stream)); },
        nb::arg("tasks"), nb::arg("num_bytes"), nb::arg("stream"), nb::call_guard<nb::gil_scoped_release>(),
        "Copy data from host to host using CUDA host function");

    module.def(
        "copy_host_to_device",
        [](std::vector<Task<MemAddress, MemAddress>> const& tasks, ssize_t numBytes, uintptr_t stream) -> int
        { return copyHostToDevice(tasks, numBytes, reinterpret_cast<CUstream>(stream)); },
        nb::arg("tasks"), nb::arg("num_bytes"), nb::arg("stream"), nb::call_guard<nb::gil_scoped_release>(),
        "Copy data from host to device using CUDA kernels");

    module.def(
        "copy_device_to_host",
        [](std::vector<Task<MemAddress, MemAddress>> const& tasks, ssize_t numBytes, uintptr_t stream) -> int
        { return copyDeviceToHost(tasks, numBytes, reinterpret_cast<CUstream>(stream)); },
        nb::arg("tasks"), nb::arg("num_bytes"), nb::arg("stream"), nb::call_guard<nb::gil_scoped_release>(),
        "Copy data from device to host using CUDA kernels");

    module.def(
        "copy_device_to_device",
        [](std::vector<Task<MemAddress, MemAddress>> const& tasks, ssize_t numBytes, uintptr_t stream) -> int
        { return copyDeviceToDevice(tasks, numBytes, reinterpret_cast<CUstream>(stream)); },
        nb::arg("tasks"), nb::arg("num_bytes"), nb::arg("stream"), nb::call_guard<nb::gil_scoped_release>(),
        "Copy data from device to device using CUDA kernels");

    module.def(
        "copy_batch_block_offsets",
        [](at::Tensor output, SizeType32 batchSize, std::vector<BlockIndices> const& batchBlockIndices,
            SizeType32 numPools, SizeType32 offset)
        {
            auto _output = from_torch(output);
            TLLM_CHECK_WITH_INFO(_output.has_value(), "Invalid output tensor.");
            copyBatchBlockOffsets(*(_output.value()), batchSize, batchBlockIndices, numPools, offset);
        },
        nb::arg("output"), nb::arg("batch_size"), nb::arg("batch_block_indices"), nb::arg("num_pools"),
        nb::arg("offset"), nb::call_guard<nb::gil_scoped_release>(), "Copy batch block indices to output tensor");
}

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
