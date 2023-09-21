/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#ifndef THIRD_PARTY_DUCC_GOOGLE_DUCC0_CUSTOM_LOWLEVEL_THREADING_H_
#define THIRD_PARTY_DUCC_GOOGLE_DUCC0_CUSTOM_LOWLEVEL_THREADING_H_

#include "tsl/platform/mutex.h"

namespace ducc0 {
namespace detail_threading {

using Mutex = tsl::mutex;
using UniqueLock = tsl::mutex_lock;
using LockGuard = tsl::mutex_lock;
using CondVar = tsl::condition_variable;

// Missing variable used by DUCC threading.cc.
static thread_local bool in_parallel_region = false;

}  // namespace detail_threading
}  // namespace ducc0

#endif  // THIRD_PARTY_DUCC_GOOGLE_DUCC0_CUSTOM_LOWLEVEL_THREADING_H_
