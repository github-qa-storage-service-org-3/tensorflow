/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_COMPILER_MLIR_QUANTIZATION_TENSORFLOW_PYTHON_QUANTIZE_MODEL_WRAPPER_H_
#define TENSORFLOW_COMPILER_MLIR_QUANTIZATION_TENSORFLOW_PYTHON_QUANTIZE_MODEL_WRAPPER_H_

#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"

namespace tensorflow {
namespace quantization {

// Runs quantization on a model trained with quantization-aware training (QAT).
// Returns serialized ExportedModel.
std::string QuantizeQatModel(absl::string_view saved_model_path,
                             const std::vector<std::string>& signature_keys,
                             const std::unordered_set<std::string>& tags,
                             absl::string_view quant_opts_serialized);

// Runs dynamic range post-training quantization (PTQ). Returns serialized
// ExportedModel.
std::string QuantizePtqDynamicRange(
    absl::string_view saved_model_path,
    const std::vector<std::string>& signature_keys,
    const std::unordered_set<std::string>& tags,
    absl::string_view quant_opts_serialized);

// Runs the pre-calibration step of post-training quantization (PTQ). Returns
// serialized ExportedModel.
std::string QuantizePtqModelPreCalibration(
    absl::string_view saved_model_path,
    const std::vector<std::string>& signature_keys,
    const std::unordered_set<std::string>& tags,
    absl::string_view quant_opts_serialized);

// Runs the post-calibration step of post-training quantization (PTQ). Returns
// serialized ExportedModel.
std::string QuantizePtqModelPostCalibration(
    absl::string_view saved_model_path,
    const std::vector<std::string>& signature_keys,
    const std::unordered_set<std::string>& tags,
    absl::string_view quant_opts_serialized);

void ClearCollectedInformationFromCalibrator();

void ClearDataFromCalibrator(absl::string_view id);

float GetMinFromCalibrator(absl::string_view id);

float GetMaxFromCalibrator(absl::string_view id);

}  // namespace quantization
}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_QUANTIZATION_TENSORFLOW_PYTHON_QUANTIZE_MODEL_WRAPPER_H_
