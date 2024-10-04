// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and

#include <gtest/gtest.h>
#include "tensorflow/lite/experimental/lrt/qnn_sdk/qnn_manager.h"
#include "tensorflow/lite/experimental/lrt/test_data/test_data_util.h"

namespace {

// NOTE: This tests that all of the dynamic loading works properly and
// the QNN SDK instance can be properly initialized and destroyed.
TEST(QnnSdkTest, SetupQnnManager) {
  qnn::QnnManager qnn;
  ASSERT_STATUS_OK(qnn::SetupAll(qnn));
}

}  // namespace
