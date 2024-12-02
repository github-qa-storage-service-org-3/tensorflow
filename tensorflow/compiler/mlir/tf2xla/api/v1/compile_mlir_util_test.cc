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

#include "tensorflow/compiler/mlir/tf2xla/api/v1/compile_mlir_util.h"

#include <initializer_list>
#include <memory>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/DialectRegistry.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "tensorflow/compiler/jit/xla_compile_util.h"
#include "tensorflow/compiler/mlir/tensorflow/dialect_registration.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/serialize_mlir_module_utils.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "xla/client/xla_builder.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/monitoring/cell_reader.h"
#include "tensorflow/core/platform/types.h"
#include "tsl/lib/core/status_test_util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace tensorflow {
namespace {

using ::mlir::OpPassManager;
using ::tensorflow::monitoring::testing::CellReader;
using ::testing::HasSubstr;

static constexpr char kMlirModuleStr[] = R"(
  module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 268 : i32}} {
    func.func @main() -> tensor<1xi32> {
      %0 = "tf.Const"() {value = dense<1000> : tensor<1xi32>} : () -> tensor<1xi32>
      func.return %0 : tensor<1xi32>
    }
  })";

TEST(LegalizeMlirTest, LegalizesModule) {
  mlir::DialectRegistry mlir_registry;
  RegisterAllTensorFlowDialects(mlir_registry);

  std::vector<tensorflow::TensorShape> arg_shapes;
  XlaCompilationResult compilation_result;
  auto status = CompileSerializedMlirToXlaHlo(
      kMlirModuleStr, arg_shapes, /*device_type=*/"XLA_TPU_JIT",
      /*use_tuple_args=*/true, /*enable_op_fallback=*/false,
      /*shape_determination_fns=*/{}, &compilation_result);

  EXPECT_TRUE(status.ok());
  EXPECT_THAT(status.value(), HasSubstr("mhlo.const"));
}

TEST(LegalizeMlirTest, FailsLegalizesModule) {
  constexpr char failed_legalization[] = R"(
  module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 268 : i32}} {
    func.func @main() -> tensor<1xi32> {
      %0 = "tf.DoesntExist"() : () -> tensor<1xi32>
      func.return %0 : tensor<1xi32>
    }
  })";
  CellReader<int64_t> count(
      "/tensorflow/core/tf2xla/v1/mlir_failed_xla_legalize_tf_pass_count");

  std::vector<tensorflow::TensorShape> arg_shapes;
  XlaCompilationResult compilation_result;
  auto status = CompileSerializedMlirToXlaHlo(
      failed_legalization, arg_shapes, /*device_type=*/"XLA_TPU_JIT",
      /*use_tuple_args=*/true, /*enable_op_fallback=*/false,
      /*shape_determination_fns=*/{}, &compilation_result);

  EXPECT_FALSE(status.ok());
  EXPECT_EQ(count.Delta("tf.DoesntExist", "Unknown"), 1);
}

TEST(CompileMlirUtil, CreatesPipeline) {
  OpPassManager pass_manager;
  llvm::StringRef device_type = "XLA_CPU_JIT";

  CreateConvertMlirToXlaHloPipeline(pass_manager, device_type,
                                    /*enable_op_fallback=*/false,
                                    /*custom_legalization_passes*/ {});

  EXPECT_FALSE(pass_manager.getPasses().empty());
}

TEST(CompileMlirUtil, HasLegalizationPass) {
  OpPassManager pass_manager;
  llvm::StringRef device_type = "XLA_CPU_JIT";
  absl::string_view kLegalizeTfPass = "xla-legalize-tf";

  CreateConvertMlirToXlaHloPipeline(pass_manager, device_type,
                                    /*enable_op_fallback=*/true,
                                    /*custom_legalization_passes*/ {});

  std::string pass_description;
  llvm::raw_string_ostream raw_stream(pass_description);
  pass_manager.printAsTextualPipeline(raw_stream);

  EXPECT_THAT(pass_description, HasSubstr(kLegalizeTfPass));
}

TEST(CompileMlirUtil, DoesNotHaveLegalizationPass) {
  OpPassManager pass_manager;
  llvm::StringRef device_type = "XLA_CPU_JIT";
  absl::string_view kLegalizeTfPass = "xla-legalize-tf";

  CreateConvertMlirToXlaHloPipeline(pass_manager, device_type,
                                    /*enable_op_fallback=*/false,
                                    /*custom_legalization_passes*/ {},
                                    /*lower_to_xla_hlo=*/false);

  std::string pass_description;
  llvm::raw_string_ostream raw_stream(pass_description);
  pass_manager.printAsTextualPipeline(raw_stream);

  EXPECT_THAT(pass_description, Not(HasSubstr(kLegalizeTfPass)));
}

TEST(CompileMlirUtil, DoesNotLowerWhenTold) {
  mlir::DialectRegistry mlir_registry;
  RegisterAllTensorFlowDialects(mlir_registry);

  std::vector<tensorflow::TensorShape> arg_shapes;
  XlaCompilationResult compilation_result;
  auto status = CompileSerializedMlirToXlaHlo(
      kMlirModuleStr, arg_shapes, /*device_type=*/"XLA_TPU_JIT",
      /*use_tuple_args=*/true, /*enable_op_fallback=*/false,
      /*shape_determination_fns=*/{}, &compilation_result,
      /*custom_legalization_passes=*/{},
      /*module_name=*/"",
      /*lower_to_xla_hlo=*/false);

  EXPECT_TRUE(status.ok());
  EXPECT_THAT(status.value(), HasSubstr("tf.Const"));
}

TEST(CompileMlirUtil, CanonicalizationIsExplicitDuringInlining) {
  OpPassManager pass_manager;
  llvm::StringRef device_type = "XLA_CPU_JIT";
  absl::string_view kInlinePass =
      "inline{default-pipeline=canonicalize "
      "inlining-threshold=4294967295 max-iterations=4 }";

  CreateConvertMlirToXlaHloPipeline(pass_manager, device_type,
                                    /*enable_op_fallback=*/true,
                                    /*custom_legalization_passes*/ {});

  std::string pass_description;
  llvm::raw_string_ostream raw_stream(pass_description);
  pass_manager.printAsTextualPipeline(raw_stream);

  EXPECT_THAT(pass_description, HasSubstr(kInlinePass));
}

TEST(LegalizeMlirTest, LegalizesModuleWithDynamicShape) {
  constexpr char legalization[] = R"(
  module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 268 : i32}} {
    func.func @main(%arg0: tensor<?xi32, #mhlo.type_extensions<bounds = [1]>>) -> tensor<?xi32, #mhlo.type_extensions<bounds = [1]>> {
      %0 = "tf.Identity"(%arg0) : (tensor<?xi32, #mhlo.type_extensions<bounds = [1]>>) -> tensor<?xi32, #mhlo.type_extensions<bounds = [1]>>
      func.return %0 : tensor<?xi32, #mhlo.type_extensions<bounds = [1]>>
    }
  })";

  std::vector<tensorflow::TensorShape> arg_shapes = {{1}};
  XlaCompilationResult compilation_result;
  auto status = CompileSerializedMlirToXlaHlo(
      legalization, arg_shapes, /*device_type=*/"XLA_TPU_JIT",
      /*use_tuple_args=*/true, /*enable_op_fallback=*/false,
      /*shape_determination_fns=*/{}, &compilation_result);

  EXPECT_TRUE(status.ok());
}

absl::StatusOr<std::unique_ptr<Graph>> BuildOpGraphWithOutputShapes() {
  DataType data_type = DT_INT32;
  std::initializer_list<int64_t> dims = {2, 3, 4, 5};
  Tensor tensor(data_type, TensorShape(dims));
  for (int i = 0; i < 2 * 3 * 4 * 5; ++i) {
    tensor.flat<int32>()(i) = i;
  }

  NodeDef node;
  auto builder = NodeDefBuilder("some_node", "Const")
                     .Attr("dtype", data_type)
                     .Attr("value", tensor);
  // Create a bad output shape attr.
  AttrValue shape_attr;
  TensorShapeProto* shape_proto = shape_attr.mutable_list()->add_shape();
  shape_proto->add_dim()->set_size(1);
  builder.Attr("_output_shapes", shape_attr);

  TF_RETURN_IF_ERROR(builder.Finalize(&node));

  return CreateSingleOpGraph(node, {}, {DataType::DT_INT32});
}

absl::Status BuildHloFromGraph(Graph& graph, bool use_output_shapes) {
  xla::XlaBuilder builder(
      ::testing::UnitTest::GetInstance()->current_test_info()->name());
  mlir::MLIRContext mlir_context;
  llvm::SmallVector<xla::XlaOp, 4> xla_params;
  std::vector<xla::XlaOp> returns(1);
  return BuildHloFromGraph(graph, builder, mlir_context, xla_params, returns,
                           use_output_shapes, /*args=*/{},
                           /*control_rets=*/{}, DEVICE_TPU,
                           FunctionLibraryDefinition(OpRegistry::Global()),
                           /*debug_info=*/{},
                           /*custom_legalization_passes=*/{});
}

TEST(CompileMlirUtil, UsesCorrectOriginalShapeWithoutOutputShapes) {
  TF_ASSERT_OK_AND_ASSIGN(auto graph, BuildOpGraphWithOutputShapes());

  auto build_result = BuildHloFromGraph(*graph, /*use_output_shapes=*/false);
  TF_ASSERT_OK(build_result);
}

TEST(CompileMlirUtil, UsesIncorrectOutputShapesWhenPresent) {
  TF_ASSERT_OK_AND_ASSIGN(auto graph, BuildOpGraphWithOutputShapes());

  auto build_result = BuildHloFromGraph(*graph, /*use_output_shapes=*/true);
  ASSERT_FALSE(build_result.ok());
  EXPECT_THAT(build_result.message(),
              HasSubstr("op operand type 'tensor<2x3x4x5xi32>' and result type "
                        "'tensor<1xi32>' are cast incompatible"));
}

}  // namespace
}  // namespace tensorflow
