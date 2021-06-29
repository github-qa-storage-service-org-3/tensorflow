/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/cwise_ops_common.h"

namespace tensorflow {
REGISTER8(BinaryOp, CPU, "FloorMod", functor::safe_floor_mod, int8, int16,
          int32, int64, uint8, uint16, uint32, uint64);
REGISTER4(BinaryOp, CPU, "FloorMod", functor::floor_fmod, Eigen::half, bfloat16,
          float, double);

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
// A special GPU kernel for int32.
// TODO(b/25387198): Also enable int32 in device memory. This kernel
// registration requires all int32 inputs and outputs to be in host memory.

#define REGISTER_KERNELS(T)                             \
  REGISTER_KERNEL_BUILDER(                              \
      Name("FloorMod")                                  \
          .Device(DEVICE_GPU)                           \
          .HostMemory("x")                              \
          .HostMemory("y")                              \
          .HostMemory("z")                              \
          .TypeConstraint<T>("T"),                      \
      BinaryOp<CPUDevice, functor::safe_floor_mod<T>>)
TF_CALL_INTEGRAL_TYPES(REGISTER_KERNELS);
#undef REGISTER_KERNELS
#endif

REGISTER_KERNEL_BUILDER(Name("FloorMod")
                            .Device(DEVICE_DEFAULT)
                            .HostMemory("x")
                            .HostMemory("y")
                            .HostMemory("z")
                            .TypeConstraint<int32>("T"),
                        BinaryOp<CPUDevice, functor::safe_floor_mod<int32>>);

}  // namespace tensorflow
