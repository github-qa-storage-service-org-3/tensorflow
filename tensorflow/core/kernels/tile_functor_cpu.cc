/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#define EIGEN_USE_THREADS

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/tile_functor.h"
#include <iostream>

namespace tensorflow {

namespace internal {

template <typename Device, typename T>
void TileSimple(const Device& device, Tensor* out, const Tensor& in) {
  const int ndims = in.dims();
  const int64 nelem = out->NumElements();
  gtl::InlinedVector<int64, 8> in_strides(ndims);
  ComputeStride(in.shape(), in_strides.data());
  gtl::InlinedVector<int64, 8> out_strides(ndims);
  ComputeStride(out->shape(), out_strides.data());
  const T* p = reinterpret_cast<const T*>(in.tensor_data().data());
  T* q = reinterpret_cast<T*>(const_cast<char*>((out->tensor_data().data())));

  for (int64 o_idx = 0; o_idx < nelem; ++o_idx) {
    int64 i_idx = 0;
    int64 t = o_idx;
    for (int i = 0; i < ndims; ++i) {
      i_idx += t / out_strides[i] % in.dim_size(i) * in_strides[i];
      t %= out_strides[i];
    }
    q[o_idx] = p[i_idx];
  }
}

}  // end namespace internal

namespace functor {

typedef Eigen::ThreadPoolDevice CPUDevice;

// Register functors used for Tile functor.
#define DEFINE_TYPE(T) template struct Tile<CPUDevice, T>;

TF_CALL_bool(DEFINE_TYPE);
TF_CALL_float(DEFINE_TYPE);
TF_CALL_double(DEFINE_TYPE);
TF_CALL_uint8(DEFINE_TYPE);
TF_CALL_int32(DEFINE_TYPE);
TF_CALL_int16(DEFINE_TYPE);
TF_CALL_int64(DEFINE_TYPE);
TF_CALL_half(DEFINE_TYPE);
TF_CALL_complex64(DEFINE_TYPE);
TF_CALL_complex128(DEFINE_TYPE);
TF_CALL_string(DEFINE_TYPE);

#undef DEFINE_TYPE

#ifdef TENSORFLOW_USE_SYCL
typedef Eigen::SyclDevice SYCLDevice;

#define DEFINE_TYPE(T) template struct Tile<SYCLDevice, T>;

TF_CALL_bool(DEFINE_TYPE);
TF_CALL_float(DEFINE_TYPE);
TF_CALL_double(DEFINE_TYPE);
TF_CALL_uint8(DEFINE_TYPE);
TF_CALL_int32(DEFINE_TYPE);
TF_CALL_int16(DEFINE_TYPE);
TF_CALL_int64(DEFINE_TYPE);

#undef DEFINE_TYPE
#endif // TENSORFLOW_USE_SYCL

}  // end namespace functor
}  // end namespace tensorflow
