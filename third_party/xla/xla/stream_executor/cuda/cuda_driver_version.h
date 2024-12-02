/* Copyright 2024 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef XLA_STREAM_EXECUTOR_CUDA_CUDA_DRIVER_VERSION_H_
#define XLA_STREAM_EXECUTOR_CUDA_CUDA_DRIVER_VERSION_H_

#include <cstdint>

#include "absl/status/statusor.h"

namespace stream_executor::gpu {

// Returns the version of the CUDA driver.
absl::StatusOr<int32_t> CudaDriverVersion();

}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_CUDA_CUDA_DRIVER_VERSION_H_
