/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_TFRT_IFRT_IFRT_SERVING_CORE_SELECTOR_H_
#define TENSORFLOW_CORE_TFRT_IFRT_IFRT_SERVING_CORE_SELECTOR_H_

#include <cstdint>

#include "tsl/framework/serving_device_selector.h"
namespace tensorflow {
namespace ifrt_serving {

// A wrapper of a `tsl::ServingDeviceSelector` that will be responsible for the
// core selection during Ifrt TPU execution.
class IfrtServingCoreSelector {
 public:
  explicit IfrtServingCoreSelector(tsl::ServingDeviceSelector* device_selector);
  // Reserves a device for the given `program_id`. The `program_id` is used to
  // identify an IFRT executable and should be the key of
  // `tensorflow::ifrt_serving::ServingExecutableRegistry `.
  tsl::DeviceReservation ReserveDevice(int64_t program_id);

 private:
  tsl::ServingDeviceSelector* device_selector_;
};

}  // namespace ifrt_serving
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TFRT_IFRT_IFRT_SERVING_CORE_SELECTOR_H_
