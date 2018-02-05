/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/contrib/lite/interpreter.h"
#include <gtest/gtest.h>
#include "tensorflow/contrib/lite/error_reporter.h"
#include "tensorflow/contrib/lite/string_util.h"
#include "tensorflow/contrib/lite/testing/util.h"

namespace tflite {
namespace {

// Make an interpreter that has no tensors and no nodes
TEST(BasicInterpreter, ZeroInterpreter) {
  Interpreter interpreter;
  interpreter.SetInputs({});
  interpreter.SetOutputs({});
  ASSERT_EQ(interpreter.AllocateTensors(), kTfLiteOk);
  ASSERT_EQ(interpreter.Invoke(), kTfLiteOk);
}

// Test various error conditions.
TEST(BasicInterpreter, InvokeInvalidModel) {
  Interpreter interpreter;
  ASSERT_NE(interpreter.Invoke(), kTfLiteOk);
  ASSERT_EQ(interpreter.AllocateTensors(), kTfLiteOk);
  ASSERT_EQ(interpreter.Invoke(), kTfLiteOk);
}

// Test size accesser functions.
TEST(BasicInterpreter, TestSizeFunctions) {
  Interpreter interpreter;
  int base_index;
  ASSERT_EQ(interpreter.nodes_size(), 0);
  ASSERT_EQ(interpreter.tensors_size(), 0);
  ASSERT_EQ(interpreter.AddTensors(2, &base_index), kTfLiteOk);
  ASSERT_EQ(interpreter.tensors_size(), 2);
  ASSERT_EQ(base_index, 0);
  ASSERT_EQ(interpreter.AddTensors(3, &base_index), kTfLiteOk);
  ASSERT_EQ(interpreter.tensors_size(), 5);
  ASSERT_EQ(interpreter.AddTensors(1), kTfLiteOk);
  ASSERT_EQ(interpreter.tensors_size(), 6);
  ASSERT_EQ(base_index, 2);
}

// Test if invalid indices make a model inconsistent (and conversely if
// valid indices keep a model consistent).
TEST(BasicInterpreter, InconsistentModel) {
  // Invalid inputs
  {
    Interpreter interpreter;
    ASSERT_NE(interpreter.SetInputs({5}), kTfLiteOk);
    ASSERT_NE(interpreter.AllocateTensors(), kTfLiteOk);
    ASSERT_NE(interpreter.Invoke(), kTfLiteOk);
    ASSERT_EQ(interpreter.inputs(), std::vector<int>());
  }
  // Invalid outputs
  {
    Interpreter interpreter;
    ASSERT_NE(interpreter.SetOutputs({5}), kTfLiteOk);
    ASSERT_NE(interpreter.AllocateTensors(), kTfLiteOk);
    ASSERT_NE(interpreter.Invoke(), kTfLiteOk);
    ASSERT_EQ(interpreter.outputs(), std::vector<int>());
  }
  // Invalid node inputs
  {
    Interpreter interpreter;
    TfLiteRegistration registration = {nullptr, nullptr, nullptr, nullptr};
    ASSERT_NE(interpreter.AddNodeWithParameters({3}, {0}, nullptr, 0, nullptr,
                                                &registration),
              kTfLiteOk);
    ASSERT_NE(interpreter.AllocateTensors(), kTfLiteOk);
    ASSERT_NE(interpreter.Invoke(), kTfLiteOk);
  }
  // Valid inputs and outputs and a node with valid inputs and outputs
  {
    Interpreter interpreter;
    ASSERT_EQ(interpreter.AddTensors(2), kTfLiteOk);
    TfLiteRegistration registration = {nullptr, nullptr, nullptr, nullptr};
    ASSERT_EQ(interpreter.SetInputs({0}), kTfLiteOk);
    ASSERT_EQ(interpreter.SetOutputs({0}), kTfLiteOk);
    ASSERT_EQ(interpreter.AddNodeWithParameters({0}, {1}, nullptr, 0, nullptr,
                                                &registration),
              kTfLiteOk);
  }
}

// Make an interpreter that has one tensor but no ops
TEST(BasicInterpreter, CheckAllocate) {
  struct {
    TfLiteType type;
    size_t size;
  } cases[] = {
      {kTfLiteFloat32, sizeof(float)},
      {kTfLiteInt32, sizeof(int32_t)},
      {kTfLiteUInt8, sizeof(uint8_t)},
      {kTfLiteInt64, sizeof(int64_t)},
  };

  for (auto test : cases) {
    Interpreter interpreter;
    ASSERT_EQ(interpreter.AddTensors(2), kTfLiteOk);
    interpreter.SetInputs({0, 1});
    interpreter.SetOutputs({});
    TfLiteQuantizationParams quant;

    interpreter.SetTensorParametersReadWrite(0, test.type, "", {3}, quant);
    interpreter.SetTensorParametersReadWrite(1, test.type, "", {4}, quant);
    ASSERT_EQ(interpreter.AllocateTensors(), kTfLiteOk);
    ASSERT_EQ(interpreter.tensor(0)->bytes, 3 * test.size);
    ASSERT_NE(interpreter.tensor(0)->data.raw, nullptr);
    ASSERT_EQ(interpreter.tensor(1)->bytes, 4 * test.size);
    ASSERT_NE(interpreter.tensor(1)->data.raw, nullptr);
  }
}

TEST(BasicInterpreter, CheckResize) {
  const float floats[] = {-3., -4.};
  const int32_t int32s[] = {-3, -4};
  const uint8_t uint8s[] = {3, 4};
  const int64_t int64s[] = {6, -7};

  struct {
    TfLiteType type;
    size_t size;
    const char* array;
  } cases[] = {
      {kTfLiteFloat32, sizeof(float), reinterpret_cast<const char*>(floats)},
      {kTfLiteInt32, sizeof(int32_t), reinterpret_cast<const char*>(int32s)},
      {kTfLiteUInt8, sizeof(uint8_t), reinterpret_cast<const char*>(uint8s)},
      {kTfLiteInt64, sizeof(int64_t), reinterpret_cast<const char*>(int64s)},
  };

  for (auto test : cases) {
    Interpreter interpreter;

    ASSERT_EQ(interpreter.AddTensors(2), kTfLiteOk);
    interpreter.SetInputs({0, 1});
    interpreter.SetOutputs({});
    TfLiteQuantizationParams quant;

    ASSERT_EQ(
        interpreter.SetTensorParametersReadWrite(0, test.type, "", {3}, quant),
        kTfLiteOk);
    ASSERT_EQ(interpreter.SetTensorParametersReadOnly(
                  1, test.type, "", {2}, quant, test.array, 2 * test.size),
              kTfLiteOk);
    ASSERT_EQ(interpreter.AllocateTensors(), kTfLiteOk);
    ASSERT_EQ(interpreter.ResizeInputTensor(0, {1, 2}), kTfLiteOk);
    // Resizing a mmapped tensor is not allowed and should produce error.
    ASSERT_NE(interpreter.ResizeInputTensor(1, {3}), kTfLiteOk);
    // Set the tensor to be mmapped but with a buffer size that is insufficient
    // to match the dimensionality.
    ASSERT_NE(interpreter.SetTensorParametersReadOnly(
                  1, test.type, "", {2}, quant, test.array, 1 * test.size),
              kTfLiteOk);
    // Allocating should work since we should have our last correct array
    // values in place.
    ASSERT_EQ(interpreter.AllocateTensors(), kTfLiteOk);
  }
}

TEST(BasicInterpreter, CheckAlignment) {
  struct {
    TfLiteType type;
  } cases[] = {
      {kTfLiteFloat32},
      {kTfLiteInt32},
      {kTfLiteUInt8},
      {kTfLiteInt64},
  };

  for (auto test : cases) {
    Interpreter interpreter;

    ASSERT_EQ(interpreter.AddTensors(4), kTfLiteOk);

    for (int i = 0; i < 4; i++) {
      TfLiteQuantizationParams quant;
      interpreter.SetTensorParametersReadWrite(i, test.type, "", {2 * i + 1},
                                               quant);
    }
    interpreter.AllocateTensors();
    for (int i = 0; i < 4; i++) {
      const TfLiteTensor& tensor = *interpreter.tensor(i);
      ASSERT_EQ(reinterpret_cast<intptr_t>(tensor.data.raw) % 4, 0);
    }
  }
}

TEST(BasicInterpreter, CheckArenaAllocation) {
  Interpreter interpreter;
  ASSERT_EQ(interpreter.AddTensors(10), kTfLiteOk);

  TfLiteQuantizationParams quant;
  TfLiteRegistration reg = {nullptr, nullptr, nullptr, nullptr};

  std::vector<int> sizes{2048, 4096, 1023, 2047, 1021,
                         2047, 1023, 2046, 1021, 2048};
  for (int i = 0; i < sizes.size(); ++i) {
    interpreter.SetTensorParametersReadWrite(i, kTfLiteUInt8, "", {sizes[i]},
                                             quant);
  }
  interpreter.SetInputs({0, 1});
  interpreter.SetOutputs({9, 4});
  interpreter.AddNodeWithParameters({0, 1}, {2, 3}, nullptr, 0, nullptr, &reg);
  interpreter.AddNodeWithParameters({2, 1}, {4, 5}, nullptr, 0, nullptr, &reg);
  interpreter.AddNodeWithParameters({4, 3}, {6, 7}, nullptr, 0, nullptr, &reg);
  interpreter.AddNodeWithParameters({6, 5}, {8}, nullptr, 0, nullptr, &reg);
  interpreter.AddNodeWithParameters({8, 7}, {9}, nullptr, 0, nullptr, &reg);

  ASSERT_EQ(interpreter.AllocateTensors(), kTfLiteOk);

  ASSERT_EQ(interpreter.tensor(0)->data.raw, interpreter.tensor(4)->data.raw);
  ASSERT_EQ(interpreter.tensor(1)->data.raw, interpreter.tensor(7)->data.raw);

  ASSERT_LT(interpreter.tensor(4)->data.raw, interpreter.tensor(1)->data.raw);
  ASSERT_LT(interpreter.tensor(6)->data.raw, interpreter.tensor(1)->data.raw);
  ASSERT_LT(interpreter.tensor(0)->data.raw, interpreter.tensor(1)->data.raw);

  ASSERT_LT(interpreter.tensor(0)->data.raw, interpreter.tensor(3)->data.raw);
  ASSERT_LT(interpreter.tensor(1)->data.raw, interpreter.tensor(3)->data.raw);
  ASSERT_LT(interpreter.tensor(2)->data.raw, interpreter.tensor(3)->data.raw);
  ASSERT_LT(interpreter.tensor(4)->data.raw, interpreter.tensor(3)->data.raw);
  ASSERT_LT(interpreter.tensor(6)->data.raw, interpreter.tensor(3)->data.raw);
  ASSERT_LT(interpreter.tensor(7)->data.raw, interpreter.tensor(3)->data.raw);
  ASSERT_LT(interpreter.tensor(8)->data.raw, interpreter.tensor(3)->data.raw);
  ASSERT_LT(interpreter.tensor(9)->data.raw, interpreter.tensor(3)->data.raw);

  ASSERT_LT(interpreter.tensor(0)->data.raw, interpreter.tensor(5)->data.raw);
  ASSERT_LT(interpreter.tensor(1)->data.raw, interpreter.tensor(5)->data.raw);
  ASSERT_LT(interpreter.tensor(2)->data.raw, interpreter.tensor(5)->data.raw);
  ASSERT_LT(interpreter.tensor(3)->data.raw, interpreter.tensor(5)->data.raw);
  ASSERT_LT(interpreter.tensor(4)->data.raw, interpreter.tensor(5)->data.raw);
  ASSERT_LT(interpreter.tensor(6)->data.raw, interpreter.tensor(5)->data.raw);
  ASSERT_LT(interpreter.tensor(7)->data.raw, interpreter.tensor(5)->data.raw);
  ASSERT_LT(interpreter.tensor(8)->data.raw, interpreter.tensor(5)->data.raw);
  ASSERT_LT(interpreter.tensor(9)->data.raw, interpreter.tensor(5)->data.raw);
}

TEST(BasicInterpreter, BufferAccess) {
  Interpreter interpreter;
  ASSERT_EQ(interpreter.AddTensors(1), kTfLiteOk);
  ASSERT_EQ(interpreter.SetInputs({0}), kTfLiteOk);

  ASSERT_EQ(interpreter.SetTensorParametersReadWrite(
                0, kTfLiteFloat32, "", {3}, TfLiteQuantizationParams()),
            kTfLiteOk);
  ASSERT_EQ(interpreter.AllocateTensors(), kTfLiteOk);
  // Verify we get a valid pointer.r
  ASSERT_NE(interpreter.typed_tensor<float>(0), nullptr);
  // Verify incorrect pointer will not returned.
  ASSERT_EQ(interpreter.typed_tensor<int>(0), nullptr);
  // Verify that raw c interface ptr matches safe interface.
  ASSERT_EQ(interpreter.typed_tensor<float>(0), interpreter.tensor(0)->data.f);
}

TEST(BasicInterpreter, NoOpInterpreter) {
  Interpreter interpreter;
  ASSERT_EQ(interpreter.AddTensors(1), kTfLiteOk);
  ASSERT_EQ(interpreter.SetInputs({0}), kTfLiteOk);
  ASSERT_EQ(interpreter.SetOutputs({0}), kTfLiteOk);

  ASSERT_EQ(interpreter.SetTensorParametersReadWrite(
                0, kTfLiteFloat32, "", {3}, TfLiteQuantizationParams()),
            kTfLiteOk);

  ASSERT_EQ(interpreter.ResizeInputTensor(interpreter.inputs()[0], {1, 2, 3}),
            kTfLiteOk);
  ASSERT_EQ(interpreter.AllocateTensors(), kTfLiteOk);
  ASSERT_EQ(interpreter.Invoke(), kTfLiteOk);
}

TEST(BasicInterpreter, ResizingTensors) {
  Interpreter interpreter;
  ASSERT_EQ(interpreter.AddTensors(1), kTfLiteOk);
  ASSERT_EQ(interpreter.SetInputs({0}), kTfLiteOk);
  ASSERT_EQ(interpreter.SetOutputs({0}), kTfLiteOk);

  ASSERT_EQ(interpreter.SetTensorParametersReadWrite(
                0, kTfLiteFloat32, "", {3}, TfLiteQuantizationParams()),
            kTfLiteOk);

  int t = interpreter.inputs()[0];
  TfLiteTensor* tensor = interpreter.tensor(t);

  ASSERT_EQ(interpreter.ResizeInputTensor(t, {1, 2, 3}), kTfLiteOk);
  EXPECT_EQ(tensor->bytes, 6 * sizeof(float));
  ASSERT_EQ(interpreter.AllocateTensors(), kTfLiteOk);

  tensor->data.f[5] = 0.123f;

  // Changing from kTfLiteArenaRw to kTfLiteDynamic is quite complicate: we need
  // to unset data.raw, otherwise Realloc will try to free that memory.
  tensor->data.raw = nullptr;
  tensor->allocation_type = kTfLiteDynamic;

  ASSERT_EQ(interpreter.ResizeInputTensor(t, {1, 2, 4}), kTfLiteOk);
  EXPECT_EQ(tensor->bytes, 8 * sizeof(float));
  ASSERT_EQ(interpreter.AllocateTensors(), kTfLiteOk);

  // TODO(ahentz): We shouldn't have to force reallocation, but
  // ResizeInputTensor doesn't realloc dynamic tensors. Also note that
  // TfLiteTensorRealloc(tensor->bytes, tensor) is a no-op.
  TfLiteTensorRealloc(9 * sizeof(float), tensor);
  tensor->data.f[7] = 0.123f;

  ASSERT_EQ(interpreter.ResizeInputTensor(t, {2, 2, 4}), kTfLiteOk);
  EXPECT_EQ(tensor->bytes, 16 * sizeof(float));
  ASSERT_EQ(interpreter.AllocateTensors(), kTfLiteOk);

  // TODO(ahentz): We shouldn't have to force reallocation, but
  // ResizeInputTensor doesn't realloc dynamic tensors. Also note that
  // TfLiteTensorRealloc(tensor->bytes, tensor) is a no-op.
  TfLiteTensorRealloc(17 * sizeof(float), tensor);
  tensor->data.f[15] = 0.123f;
}

TEST(BasicInterpreter, OneOpInterpreter) {
  Interpreter interpreter;
  ASSERT_EQ(interpreter.AddTensors(2), kTfLiteOk);
  ASSERT_EQ(interpreter.SetInputs({0}), kTfLiteOk);
  ASSERT_EQ(interpreter.SetOutputs({1}), kTfLiteOk);

  TfLiteQuantizationParams quantized;
  ASSERT_EQ(interpreter.SetTensorParametersReadWrite(0, kTfLiteFloat32, "in1",
                                                     {3}, quantized),
            kTfLiteOk);
  ASSERT_EQ(interpreter.SetTensorParametersReadWrite(1, kTfLiteFloat32, "out0",
                                                     {3}, quantized),
            kTfLiteOk);

  ASSERT_EQ(interpreter.GetInputName(0), "in1");
  ASSERT_EQ(interpreter.GetOutputName(0), "out0");

  TfLiteRegistration reg = {nullptr, nullptr, nullptr, nullptr};
  reg.init = [](TfLiteContext* context, const char*, size_t) -> void* {
    auto* first_new_tensor = new int;
    context->AddTensors(context, 2, first_new_tensor);
    return first_new_tensor;
  };
  reg.free = [](TfLiteContext* context, void* buffer) {
    delete reinterpret_cast<int*>(buffer);
  };
  reg.prepare = [](TfLiteContext* context, TfLiteNode* node) {
    auto* first_new_tensor = reinterpret_cast<int*>(node->user_data);

    TfLiteTensor* tensor0 = &context->tensors[node->inputs->data[0]];
    TfLiteTensor* tensor1 = &context->tensors[node->outputs->data[0]];

    TfLiteIntArray* newSize = TfLiteIntArrayCopy(tensor0->dims);
    TF_LITE_ENSURE_STATUS(context->ResizeTensor(context, tensor1, newSize));

    TfLiteIntArrayFree(node->temporaries);
    node->temporaries = TfLiteIntArrayCreate(2);
    for (int i = 0; i < 2; ++i) {
      node->temporaries->data[i] = *(first_new_tensor) + i;
    }

    auto setup_temporary = [&](int id) {
      TfLiteTensor* tmp = &context->tensors[id];
      tmp->type = kTfLiteFloat32;
      tmp->allocation_type = kTfLiteArenaRw;
      return context->ResizeTensor(context, tmp,
                                   TfLiteIntArrayCopy(tensor0->dims));
    };
    TF_LITE_ENSURE_STATUS(setup_temporary(node->temporaries->data[0]));
    TF_LITE_ENSURE_STATUS(setup_temporary(node->temporaries->data[1]));

    return kTfLiteOk;
  };
  reg.invoke = [](TfLiteContext* context, TfLiteNode* node) {
    TfLiteTensor* a0 = &context->tensors[node->inputs->data[0]];

    auto populate = [&](int id) {
      TfLiteTensor* t = &context->tensors[id];
      int num = a0->dims->data[0];
      for (int i = 0; i < num; i++) {
        t->data.f[i] = a0->data.f[i];
      }
    };

    populate(node->outputs->data[0]);
    populate(node->temporaries->data[0]);
    populate(node->temporaries->data[1]);
    return kTfLiteOk;
  };
  ASSERT_EQ(
      interpreter.AddNodeWithParameters({0}, {1}, nullptr, 0, nullptr, &reg),
      kTfLiteOk);
  ASSERT_EQ(interpreter.ResizeInputTensor(0, {3}), kTfLiteOk);
  ASSERT_EQ(interpreter.AllocateTensors(), kTfLiteOk);

  ASSERT_EQ(interpreter.Invoke(), kTfLiteOk);
}

// Forcefully divides tensor allocation in three steps: one before invocation
// and two more at invocation time. This happens because we use string tensors
// and their sizes can't be determined until invocation time.
TEST(BasicInterpreter, ThreeStepAllocate) {
  Interpreter interpreter;
  ASSERT_EQ(interpreter.AddTensors(5), kTfLiteOk);
  ASSERT_EQ(interpreter.SetInputs({0}), kTfLiteOk);
  ASSERT_EQ(interpreter.SetOutputs({4}), kTfLiteOk);

  TfLiteQuantizationParams quantized;
  char data[] = {1, 0, 0, 0, 12, 0, 0, 0, 15, 0, 0, 0, 'A', 'B', 'C'};
  // Read only string tensor.
  ASSERT_EQ(interpreter.SetTensorParametersReadOnly(0, kTfLiteString, "", {1},
                                                    quantized, data, 15),
            kTfLiteOk);
  // Read-write string tensor.
  ASSERT_EQ(interpreter.SetTensorParametersReadWrite(1, kTfLiteString, "", {1},
                                                     quantized),
            kTfLiteOk);
  ASSERT_EQ(interpreter.SetTensorParametersReadWrite(2, kTfLiteInt32, "", {1},
                                                     quantized),
            kTfLiteOk);
  ASSERT_EQ(interpreter.SetTensorParametersReadWrite(3, kTfLiteString, "", {1},
                                                     quantized),
            kTfLiteOk);
  ASSERT_EQ(interpreter.SetTensorParametersReadWrite(4, kTfLiteInt32, "", {1},
                                                     quantized),
            kTfLiteOk);

  // String-in String-out node.
  TfLiteRegistration reg_copy = {nullptr, nullptr, nullptr, nullptr};
  reg_copy.invoke = [](TfLiteContext* context, TfLiteNode* node) {
    TfLiteTensor* a0 = &context->tensors[node->inputs->data[0]];
    TfLiteTensor* a1 = &context->tensors[node->outputs->data[0]];
    DynamicBuffer buf;
    StringRef str_ref = GetString(a0, 0);
    buf.AddString(str_ref);
    buf.WriteToTensor(a1);
    return kTfLiteOk;
  };

  // String-in Int-out node.
  TfLiteRegistration reg_len = {nullptr, nullptr, nullptr, nullptr};
  reg_len.prepare = [](TfLiteContext* context, TfLiteNode* node) {
    TfLiteTensor* output = &context->tensors[node->outputs->data[0]];
    TfLiteIntArray* outputSize = TfLiteIntArrayCreate(1);
    outputSize->data[0] = 1;
    return context->ResizeTensor(context, output, outputSize);
  };
  reg_len.invoke = [](TfLiteContext* context, TfLiteNode* node) {
    TfLiteTensor* a0 = &context->tensors[node->inputs->data[0]];
    TfLiteTensor* a1 = &context->tensors[node->outputs->data[0]];
    a1->data.i32[0] = a0->bytes;
    return kTfLiteOk;
  };

  ASSERT_EQ(interpreter.AddNodeWithParameters({0}, {1}, nullptr, 0, nullptr,
                                              &reg_copy),
            kTfLiteOk);
  ASSERT_EQ(interpreter.AddNodeWithParameters({1}, {2}, nullptr, 0, nullptr,
                                              &reg_len),
            kTfLiteOk);
  ASSERT_EQ(interpreter.AddNodeWithParameters({0}, {3}, nullptr, 0, nullptr,
                                              &reg_copy),
            kTfLiteOk);
  ASSERT_EQ(interpreter.AddNodeWithParameters({3}, {4}, nullptr, 0, nullptr,
                                              &reg_len),
            kTfLiteOk);

  ASSERT_EQ(interpreter.AllocateTensors(), kTfLiteOk);
  ASSERT_EQ(interpreter.Invoke(), kTfLiteOk);

  ASSERT_EQ(interpreter.tensor(0)->bytes, 15);
  ASSERT_NE(interpreter.tensor(0)->data.raw, nullptr);
  ASSERT_EQ(interpreter.tensor(1)->bytes, 15);
  ASSERT_NE(interpreter.tensor(1)->data.raw, nullptr);
  ASSERT_EQ(interpreter.tensor(3)->bytes, 15);
  ASSERT_NE(interpreter.tensor(4)->data.raw, nullptr);
  ASSERT_EQ(interpreter.tensor(2)->bytes, 4);
  ASSERT_EQ(interpreter.tensor(2)->data.i32[0], 15);
  ASSERT_EQ(interpreter.tensor(4)->bytes, 4);
  ASSERT_EQ(interpreter.tensor(4)->data.i32[0], 15);
}

TEST(BasicInterpreter, AllocateTwice) {
  Interpreter interpreter;
  ASSERT_EQ(interpreter.AddTensors(2), kTfLiteOk);
  ASSERT_EQ(interpreter.SetInputs({0}), kTfLiteOk);
  ASSERT_EQ(interpreter.SetOutputs({1}), kTfLiteOk);

  TfLiteQuantizationParams quantized;
  ASSERT_EQ(interpreter.SetTensorParametersReadWrite(0, kTfLiteFloat32, "", {3},
                                                     quantized),
            kTfLiteOk);
  ASSERT_EQ(interpreter.SetTensorParametersReadWrite(1, kTfLiteFloat32, "", {3},
                                                     quantized),
            kTfLiteOk);

  TfLiteRegistration reg = {nullptr, nullptr, nullptr, nullptr};
  reg.prepare = [](TfLiteContext* context, TfLiteNode* node) {
    TfLiteTensor* tensor0 = &context->tensors[node->inputs->data[0]];
    TfLiteTensor* tensor1 = &context->tensors[node->outputs->data[0]];
    TfLiteIntArray* newSize = TfLiteIntArrayCopy(tensor0->dims);
    return context->ResizeTensor(context, tensor1, newSize);
  };
  reg.invoke = [](TfLiteContext* context, TfLiteNode* node) {
    TfLiteTensor* a0 = &context->tensors[node->inputs->data[0]];
    TfLiteTensor* a1 = &context->tensors[node->outputs->data[0]];
    int num = a0->dims->data[0];
    for (int i = 0; i < num; i++) {
      a1->data.f[i] = a0->data.f[i];
    }
    return kTfLiteOk;
  };
  ASSERT_EQ(
      interpreter.AddNodeWithParameters({0}, {1}, nullptr, 0, nullptr, &reg),
      kTfLiteOk);
  ASSERT_EQ(interpreter.ResizeInputTensor(0, {3}), kTfLiteOk);
  ASSERT_EQ(interpreter.AllocateTensors(), kTfLiteOk);
  ASSERT_EQ(interpreter.Invoke(), kTfLiteOk);
  char* old_tensor0_ptr = interpreter.tensor(0)->data.raw;
  char* old_tensor1_ptr = interpreter.tensor(1)->data.raw;

  ASSERT_EQ(interpreter.AllocateTensors(), kTfLiteOk);
  ASSERT_EQ(interpreter.Invoke(), kTfLiteOk);
  ASSERT_EQ(old_tensor0_ptr, interpreter.tensor(0)->data.raw);
  ASSERT_EQ(old_tensor1_ptr, interpreter.tensor(1)->data.raw);
}

struct TestErrorReporter : public ErrorReporter {
  int Report(const char* format, va_list args) override {
    char buffer[1024];
    int size = vsnprintf(buffer, sizeof(buffer), format, args);
    all_reports += buffer;
    calls++;
    return size;
  }
  int calls = 0;
  std::string all_reports;
};

TEST(BasicInterpreter, TestNullErrorReporter) {
  TestErrorReporter reporter;
  Interpreter interpreter;
}

TEST(BasicInterpreter, TestCustomErrorReporter) {
  TestErrorReporter reporter;
  Interpreter interpreter(&reporter);
  ASSERT_NE(interpreter.Invoke(), kTfLiteOk);
  ASSERT_EQ(reporter.all_reports, "Invoke called on model that is not ready.");
  ASSERT_EQ(reporter.calls, 1);
}

// Test fixture that allows playing with execution plans. It creates a two
// node graph that can be executed in either [0,1] order or [1,0] order.
// The CopyOp records when it is invoked in the class member run_order_
// so we can test whether the execution plan was honored.
class TestExecutionPlan : public ::testing::Test {
  // Encapsulates the node ids and provides them to a C primitive data type
  // Allocatable with placement new, but never destructed, so make sure this
  // doesn't own any heap allocated data. This is then is used as op local
  // data to allow access to the test fixture data.
  class CallReporting {
   public:
    CallReporting(int node_id, std::vector<int>* run_order)
        : node_id_(node_id), run_order_(run_order) {}

    void Record() { run_order_->push_back(node_id_); }

   private:
    // The node id for this particular node
    int node_id_;
    // A pointer to the global run-order
    std::vector<int>* run_order_;
  };

  // Build a kernel registration for an op that copies its one input
  // to an output
  TfLiteRegistration CopyOpRegistration() {
    TfLiteRegistration reg = {nullptr, nullptr, nullptr, nullptr};

    reg.prepare = [](TfLiteContext* context, TfLiteNode* node) {
      // Set output size to input size
      TfLiteTensor* tensor0 = &context->tensors[node->inputs->data[0]];
      TfLiteTensor* tensor1 = &context->tensors[node->outputs->data[0]];
      TfLiteIntArray* newSize = TfLiteIntArrayCopy(tensor0->dims);
      return context->ResizeTensor(context, tensor1, newSize);
    };

    reg.invoke = [](TfLiteContext* context, TfLiteNode* node) {
      CallReporting* call_reporting =
          reinterpret_cast<CallReporting*>(node->builtin_data);
      // Copy input data to output data.
      TfLiteTensor* a0 = &context->tensors[node->inputs->data[0]];
      TfLiteTensor* a1 = &context->tensors[node->outputs->data[0]];
      int num = a0->dims->data[0];
      for (int i = 0; i < num; i++) {
        a1->data.f[i] = a0->data.f[i];
      }
      call_reporting->Record();
      return kTfLiteOk;
    };
    return reg;
  }

  // Adds a copy node going from tensor `input` to output tensor `output`.
  // Note, input is used as the node_id. Inject run_order as op accessible
  // data. Note: this is a little strange of a way to do this, but it is
  // using op functionality to avoid static global variables.
  void MakeCopyNode(int input, int output) {
    // Ownership of call_reporting is taken by interpreter (malloc is used due
    // to nodes being a C99 interface so free() is used).
    TfLiteRegistration copy_op = CopyOpRegistration();
    CallReporting* call_reporting_1 =
        reinterpret_cast<CallReporting*>(malloc(sizeof(CallReporting)));
    new (call_reporting_1) CallReporting(input, &run_order_);
    ASSERT_EQ(interpreter_.AddNodeWithParameters(
                  {0}, {2}, nullptr, 0,
                  reinterpret_cast<void*>(call_reporting_1), &copy_op),
              kTfLiteOk);
    ASSERT_EQ(interpreter_.ResizeInputTensor(input, {3}), kTfLiteOk);
  }

  void SetUp() final {
    // Add two inputs and two outputs that don't depend on each other
    ASSERT_EQ(interpreter_.AddTensors(4), kTfLiteOk);
    interpreter_.SetInputs({0, 1});
    interpreter_.SetOutputs({2, 3});
    TfLiteQuantizationParams quantized;
    for (int tensor_index = 0; tensor_index < 4; tensor_index++) {
      ASSERT_EQ(interpreter_.SetTensorParametersReadWrite(
                    tensor_index, kTfLiteFloat32, "", {3}, quantized),
                kTfLiteOk);
    }

    // Define two copy functions that also use the user_data to report that
    // they were called.
    // i.e. tensor[2] = copy(tensor[0]); tensor[3] = copy(tensor[1]);
    // thus we can reorder the two nodes arbitrary and still satisfy dependency
    // order.
    MakeCopyNode(0, 2);
    MakeCopyNode(1, 3);

    ASSERT_EQ(interpreter_.AllocateTensors(), kTfLiteOk);
  }

 protected:
  Interpreter interpreter_;

  // list of node_ids that were run
  std::vector<int> run_order_;
};

TEST_F(TestExecutionPlan, DefaultExecutionPlan) {
  // Check default order
  ASSERT_EQ(interpreter_.Invoke(), kTfLiteOk);
  ASSERT_EQ(run_order_, std::vector<int>({0, 1}));
}

TEST_F(TestExecutionPlan, ReversedExecutionPlan) {
  // Check reversed order
  interpreter_.SetExecutionPlan({1, 0});
  ASSERT_EQ(interpreter_.Invoke(), kTfLiteOk);
  ASSERT_EQ(run_order_, std::vector<int>({1, 0}));
}

TEST_F(TestExecutionPlan, SubsetExecutionPlan) {
  // Check running only node index 1
  interpreter_.SetExecutionPlan({1});
  ASSERT_EQ(interpreter_.Invoke(), kTfLiteOk);
  ASSERT_EQ(run_order_, std::vector<int>({1}));
}

TEST_F(TestExecutionPlan, NullExecutionPlan) {
  // Check nothing executed.
  interpreter_.SetExecutionPlan({});
  ASSERT_EQ(interpreter_.Invoke(), kTfLiteOk);
  ASSERT_EQ(run_order_, std::vector<int>());
}

}  // namespace
}  // namespace tflite

int main(int argc, char** argv) {
  ::tflite::LogToStderr();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
