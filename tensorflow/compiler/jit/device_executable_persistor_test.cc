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

#include "tensorflow/compiler/jit/device_executable_persistor.h"

#include <stdlib.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/function_ops.h"
#include "tensorflow/cc/ops/math_ops.h"
#include "tensorflow/compiler/jit/device_compiler_client.h"
#include "tensorflow/compiler/jit/xla_compilation_cache.pb.h"
#include "tensorflow/compiler/jit/xla_device_compiler_client.h"
#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/compiler/xla/client/executable_build_options.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status_matchers.h"
#include "tensorflow/core/platform/statusor.h"

namespace tensorflow {
namespace {

using ::testing::_;
using ::testing::ByMove;
using ::testing::Return;
using XlaDeviceExecutablePersistor =
    DeviceExecutablePersistor<xla::LocalExecutable, xla::LocalClient>;

class DeviceExecutionPersistorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    compiler_client_ = std::make_unique<XlaDeviceCompilerClient>(
        xla::ClientLibrary::LocalClientOrDie());

    XlaOpRegistry::RegisterCompilationKernels();

    flib_def_ = std::make_unique<FunctionLibraryDefinition>(
        OpRegistry::Global(), FunctionDefLibrary());

    cache_dir_ = testing::TmpDir();
    TF_ASSERT_OK_AND_ASSIGN(compilation_result_add_,
                            BuildSampleCompilationResult());
  }

  StatusOr<std::unique_ptr<xla::LocalExecutable>> BuildSampleExecutable() {
    return compiler_client_->BuildExecutable(DefaultOptions(),
                                             compilation_result_add_);
  }

  StatusOr<XlaCompiler::CompilationResult> BuildSampleCompilationResult(
      bool mul = false) {
    std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
    Scope scope = Scope::NewRootScope().ExitOnError();
    auto a = ops::_Arg(scope.WithOpName("A"), DT_INT32, 0);
    auto b = ops::_Arg(scope.WithOpName("B"), DT_INT32, 1);
    if (mul) {
      auto c = ops::Mul(scope.WithOpName("C"), a, b);
      auto d = ops::_Retval(scope.WithOpName("D"), c, 0);
      TF_RETURN_IF_ERROR(scope.ToGraph(graph.get()));
    } else {
      auto c = ops::Add(scope.WithOpName("C"), a, b);
      auto d = ops::_Retval(scope.WithOpName("D"), c, 0);
      TF_RETURN_IF_ERROR(scope.ToGraph(graph.get()));
    }

    // Builds a description of the arguments.
    std::vector<XlaCompiler::Argument> args(2);
    args[0].kind = XlaCompiler::Argument::kParameter;
    args[0].type = DT_INT32;
    args[0].shape = TensorShape({2});
    args[1].kind = XlaCompiler::Argument::kParameter;
    args[1].type = DT_INT32;
    args[1].shape = TensorShape({2});

    // Compiles the graph.
    XlaCompiler compiler(DefaultOptions());

    XlaCompiler::CompilationResult compilation_result;
    TF_RETURN_IF_ERROR(compiler.CompileGraph(XlaCompiler::CompileOptions(),
                                             "graph", std::move(graph), args,
                                             &compilation_result));
    return compilation_result;
  }

  XlaCompiler::Options DefaultOptions() {
    XlaCompiler::Options options;
    options.device_type = DeviceType(DEVICE_CPU_XLA_JIT);
    options.client = compiler_client_->client();
    options.flib_def = flib_def_.get();
    return options;
  }

  std::unique_ptr<FunctionLibraryDefinition> flib_def_;
  std::unique_ptr<XlaDeviceCompilerClient> compiler_client_;
  XlaCompiler::CompilationResult compilation_result_add_;
  std::string serialized_executable_ = "serialized_executable";
  std::string cache_dir_;
};

// Using a mock to make testing different branches and triggering errors easier.
// Currently the `XlaDeviceCompilerClient`'s load/serialize functions don't work
// with the current test setup.
// TODO(b/255826209): Look into using a real object for most tests.
class MockCompilerClient : public XlaDeviceCompilerClient {
 public:
  MockCompilerClient() : XlaDeviceCompilerClient(nullptr) {}
  MOCK_METHOD(StatusOr<std::string>, SerializeExecutable,
              (const xla::LocalExecutable& executable), (override));
  MOCK_METHOD(StatusOr<std::string>, BuildSerializedExecutable,
              (const XlaCompiler::Options& options,
               const XlaCompiler::CompilationResult& result),
              (override));
  MOCK_METHOD(StatusOr<std::unique_ptr<xla::LocalExecutable>>, LoadExecutable,
              (const XlaCompiler::Options& options,
               const XlaCompiler::CompilationResult& result,
               const std::string& serialized_executable),
              (override));
};

std::string GetFilePath(XlaSerializedCacheKey key,
                        const std::string& persistent_cache_dir) {
  static constexpr char kXlaSerializedCacheKeySeparator[] = "__";

  std::string file_name = absl::StrCat(
      key.prefix(), key.prefix().empty() ? "" : kXlaSerializedCacheKeySeparator,
      key.signature_fingerprint(), kXlaSerializedCacheKeySeparator,
      key.cluster_fingerprint(), kXlaSerializedCacheKeySeparator,
      key.device_type(), ".pb");

  return io::JoinPath(persistent_cache_dir, file_name);
}

StatusOr<XlaSerializedCacheEntry> ReadCacheEntryFromFile(
    XlaSerializedCacheKey key, const std::string& persistent_cache_dir) {
  std::string file_path = GetFilePath(key, persistent_cache_dir);
  XlaSerializedCacheEntry entry;
  TF_RETURN_IF_ERROR(ReadTextOrBinaryProto(Env::Default(), file_path, &entry));
  return entry;
}

XlaSerializedCacheKey CreateCacheKey(
    uint64 signature_hash,
    const XlaCompiler::CompilationResult& compilation_result,
    const DeviceType& device_type, const std::string& persistence_prefix) {
  XlaSerializedCacheKey key;
  key.set_signature_fingerprint(signature_hash);
  key.set_cluster_fingerprint(
      DeterministicProtoHash64(compilation_result.computation->proto()));
  key.set_device_type(device_type.type_string());
  key.set_prefix(persistence_prefix);
  return key;
}

TEST_F(DeviceExecutionPersistorTest, PersistCacheDirNotSet) {
  XlaDeviceExecutablePersistor::Config config(
      /*persistent_cache_directory=*/"",
      /*disable_strict_signature_checks=*/false,
      /*persistence_prefix=*/"xla");
  XlaDeviceExecutablePersistor persistor(config, DefaultOptions().device_type);

  MockCompilerClient mock_client;
  TF_ASSERT_OK_AND_ASSIGN(auto executable, BuildSampleExecutable());
  TF_EXPECT_OK(persistor.TryToPersistExecutable(
      /*signature_hash=*/123, "signature_string", DefaultOptions(),
      compilation_result_add_, *executable, &mock_client));

  auto key =
      CreateCacheKey(/*signature_hash=*/123, compilation_result_add_,
                     persistor.device_type(), persistor.persistence_prefix());
  auto entry = ReadCacheEntryFromFile(key, "");
  EXPECT_FALSE(entry.ok());
}

TEST_F(DeviceExecutionPersistorTest, PersistSerializeAlreadyBuiltExecutable) {
  XlaDeviceExecutablePersistor::Config config(
      /*persistent_cache_directory=*/cache_dir_,
      /*disable_strict_signature_checks=*/false,
      /*persistence_prefix=*/"xla");
  XlaDeviceExecutablePersistor persistor(config, DefaultOptions().device_type);

  MockCompilerClient mock_client;
  EXPECT_CALL(mock_client, SerializeExecutable(_))
      .WillOnce(Return(StatusOr<std::string>(serialized_executable_)));

  TF_ASSERT_OK_AND_ASSIGN(auto executable, BuildSampleExecutable());
  TF_EXPECT_OK(persistor.TryToPersistExecutable(
      /*signature_hash=*/123, "signature_string", DefaultOptions(),
      compilation_result_add_, *executable, &mock_client));

  auto key =
      CreateCacheKey(/*signature_hash=*/123, compilation_result_add_,
                     persistor.device_type(), persistor.persistence_prefix());
  TF_ASSERT_OK_AND_ASSIGN(auto entry, ReadCacheEntryFromFile(key, cache_dir_));

  EXPECT_EQ(entry.executable(), serialized_executable_);
}

TEST_F(DeviceExecutionPersistorTest, PersistBuildSerializedExecutable) {
  XlaDeviceExecutablePersistor::Config config(
      /*persistent_cache_directory=*/cache_dir_,
      /*disable_strict_signature_checks=*/false,
      /*persistence_prefix=*/"xla");
  XlaDeviceExecutablePersistor persistor(config, DefaultOptions().device_type);

  MockCompilerClient mock_client;
  EXPECT_CALL(mock_client, SerializeExecutable(_))
      .WillOnce(Return(errors::Unimplemented("Unimplemented.")));
  EXPECT_CALL(mock_client, BuildSerializedExecutable(_, _))
      .WillOnce(Return(serialized_executable_));

  TF_ASSERT_OK_AND_ASSIGN(auto executable, BuildSampleExecutable());
  TF_EXPECT_OK(persistor.TryToPersistExecutable(
      /*signature_hash=*/123, "signature_string", DefaultOptions(),
      compilation_result_add_, *executable, &mock_client));

  auto key =
      CreateCacheKey(/*signature_hash=*/123, compilation_result_add_,
                     persistor.device_type(), persistor.persistence_prefix());
  TF_ASSERT_OK_AND_ASSIGN(auto entry, ReadCacheEntryFromFile(key, cache_dir_));

  EXPECT_EQ(entry.executable(), serialized_executable_);
}

TEST_F(DeviceExecutionPersistorTest, PersistSerializeExecutableError) {
  XlaDeviceExecutablePersistor::Config config(
      /*persistent_cache_directory=*/cache_dir_,
      /*disable_strict_signature_checks=*/false,
      /*persistence_prefix=*/"xla");
  XlaDeviceExecutablePersistor persistor(config, DefaultOptions().device_type);

  MockCompilerClient mock_client;
  EXPECT_CALL(mock_client, SerializeExecutable(_))
      .WillOnce(Return(errors::InvalidArgument("InvalidArgument.")));

  TF_ASSERT_OK_AND_ASSIGN(auto executable, BuildSampleExecutable());
  EXPECT_THAT(persistor.TryToPersistExecutable(
                  /*signature_hash=*/123, "signature_string", DefaultOptions(),
                  compilation_result_add_, *executable, &mock_client),
              testing::StatusIs(error::INVALID_ARGUMENT));
}

TEST_F(DeviceExecutionPersistorTest, PersistExecutableEmpty) {
  XlaDeviceExecutablePersistor::Config config(
      /*persistent_cache_directory=*/cache_dir_,
      /*disable_strict_signature_checks=*/false,
      /*persistence_prefix=*/"xla");
  XlaDeviceExecutablePersistor persistor(config, DefaultOptions().device_type);

  MockCompilerClient mock_client;
  xla::LocalExecutable empty_executable(
      nullptr, nullptr,
      GetExecutableBuildOptions(DefaultOptions(), compilation_result_add_, 0));

  TF_ASSERT_OK_AND_ASSIGN(auto executable, BuildSampleExecutable());
  EXPECT_THAT(persistor.TryToPersistExecutable(
                  /*signature_hash=*/123, "signature_string", DefaultOptions(),
                  compilation_result_add_, empty_executable, &mock_client),
              testing::StatusIs(error::FAILED_PRECONDITION));
}

TEST_F(DeviceExecutionPersistorTest, LoadCacheDirNotSet) {
  XlaDeviceExecutablePersistor::Config config(
      /*persistent_cache_directory=*/"",
      /*disable_strict_signature_checks=*/false,
      /*persistence_prefix=*/"xla");
  XlaDeviceExecutablePersistor persistor(config, DefaultOptions().device_type);

  MockCompilerClient mock_client;
  auto executable =
      persistor.TryToLoadExecutable(123, "signature_string", DefaultOptions(),
                                    compilation_result_add_, &mock_client);
  EXPECT_FALSE(executable.has_value());
}

TEST_F(DeviceExecutionPersistorTest, LoadSuccess) {
  XlaDeviceExecutablePersistor::Config config(
      /*persistent_cache_directory=*/cache_dir_,
      /*disable_strict_signature_checks=*/false,
      /*persistence_prefix=*/"xla");
  XlaDeviceExecutablePersistor persistor(config, DefaultOptions().device_type);

  MockCompilerClient mock_client;
  TF_ASSERT_OK_AND_ASSIGN(auto executable, BuildSampleExecutable());
  EXPECT_CALL(mock_client, LoadExecutable(_, _, serialized_executable_))
      .WillOnce(Return(ByMove(std::move(executable))));

  auto loaded_executable = persistor.TryToLoadExecutable(
      /*signature_hash=*/123, "signature_string", DefaultOptions(),
      compilation_result_add_, &mock_client);

  EXPECT_TRUE(loaded_executable.has_value());
  EXPECT_TRUE(loaded_executable.value().ok());
  EXPECT_TRUE((*loaded_executable.value())->executable() != nullptr);
}

TEST_F(DeviceExecutionPersistorTest, LoadFileDoesntExist) {
  XlaDeviceExecutablePersistor::Config config(
      /*persistent_cache_directory=*/cache_dir_,
      /*disable_strict_signature_checks=*/false,
      /*persistence_prefix=*/"xla");
  XlaDeviceExecutablePersistor persistor(config, DefaultOptions().device_type);

  MockCompilerClient mock_client;
  // Try to load an executable for a different signature hash (which hasn't been
  // persisted).
  auto loaded_executable = persistor.TryToLoadExecutable(
      /*signature_hash=*/12345, "different_signature", DefaultOptions(),
      compilation_result_add_, &mock_client);

  EXPECT_FALSE(loaded_executable.has_value());
}

TEST_F(DeviceExecutionPersistorTest, LoadSerializedKeyMismatch) {
  XlaDeviceExecutablePersistor::Config config(
      /*persistent_cache_directory=*/cache_dir_,
      /*disable_strict_signature_checks=*/false,
      /*persistence_prefix=*/"xla");
  XlaDeviceExecutablePersistor persistor(config, DefaultOptions().device_type);

  auto key1 =
      CreateCacheKey(/*signature_hash=*/123, compilation_result_add_,
                     persistor.device_type(), persistor.persistence_prefix());
  auto key2 =
      CreateCacheKey(/*signature_hash=*/456, compilation_result_add_,
                     persistor.device_type(), persistor.persistence_prefix());
  // File for key2 contains the same content as key1.
  TF_ASSERT_OK(Env::Default()->CopyFile(
      GetFilePath(key1, persistor.persistent_cache_directory()),
      GetFilePath(key2, persistor.persistent_cache_directory())));

  MockCompilerClient mock_client;
  // Try to load an executable from file corresponding to key2 (whose file
  // content corresponds to key1).
  auto loaded_executable = persistor.TryToLoadExecutable(
      /*signature_hash=*/456, "different_signature", DefaultOptions(),
      compilation_result_add_, &mock_client);

  EXPECT_TRUE(loaded_executable.has_value());
  EXPECT_FALSE(loaded_executable->ok());
  EXPECT_THAT(loaded_executable.value(),
              testing::StatusIs(error::INVALID_ARGUMENT));
}

TEST_F(DeviceExecutionPersistorTest, LoadSerializedHloMismatch) {
  XlaDeviceExecutablePersistor::Config config(
      /*persistent_cache_directory=*/cache_dir_,
      /*disable_strict_signature_checks=*/false,
      /*persistence_prefix=*/"xla");
  XlaDeviceExecutablePersistor persistor(config, DefaultOptions().device_type);

  TF_ASSERT_OK_AND_ASSIGN(auto compilation_result_mul,
                          BuildSampleCompilationResult(true));

  auto key1 =
      CreateCacheKey(/*signature_hash=*/123, compilation_result_add_,
                     persistor.device_type(), persistor.persistence_prefix());
  auto key2 =
      CreateCacheKey(/*signature_hash=*/123, compilation_result_mul,
                     persistor.device_type(), persistor.persistence_prefix());

  // Read serialized entry corresponding to key1.
  XlaSerializedCacheEntry entry;
  TF_ASSERT_OK(ReadTextOrBinaryProto(
      Env::Default(), GetFilePath(key1, persistor.persistent_cache_directory()),
      &entry));
  // Change the entry's key to key2.
  *entry.mutable_key() = key2;
  // Write the modified entry to file corresponding to key2.
  TF_ASSERT_OK(WriteBinaryProto(
      Env::Default(), GetFilePath(key2, persistor.persistent_cache_directory()),
      entry));

  MockCompilerClient mock_client;
  // Try to load executable corresponding to key2 (whose file contains HLO
  // corresponding to key1).
  auto loaded_executable = persistor.TryToLoadExecutable(
      /*signature_hash=*/123, "signature", DefaultOptions(),
      compilation_result_mul, &mock_client);

  EXPECT_TRUE(loaded_executable.has_value());
  EXPECT_FALSE(loaded_executable->ok());
  EXPECT_THAT(loaded_executable.value(),
              testing::StatusIs(error::INVALID_ARGUMENT));
}

TEST_F(DeviceExecutionPersistorTest, LoadStrictChecksDisabled) {
  XlaDeviceExecutablePersistor::Config config(
      /*persistent_cache_directory=*/cache_dir_,
      /*disable_strict_signature_checks=*/true,
      /*persistence_prefix=*/"xla");
  XlaDeviceExecutablePersistor persistor(config, DefaultOptions().device_type);

  TF_ASSERT_OK_AND_ASSIGN(auto compilation_result_mul,
                          BuildSampleCompilationResult(true));

  auto key1 =
      CreateCacheKey(/*signature_hash=*/123, compilation_result_add_,
                     persistor.device_type(), persistor.persistence_prefix());
  auto key2 =
      CreateCacheKey(/*signature_hash=*/123, compilation_result_mul,
                     persistor.device_type(), persistor.persistence_prefix());

  // Read serialized entry corresponding to key1.
  XlaSerializedCacheEntry entry;
  TF_ASSERT_OK(ReadTextOrBinaryProto(
      Env::Default(), GetFilePath(key1, persistor.persistent_cache_directory()),
      &entry));
  // Change the entry's key to key2.
  *entry.mutable_key() = key2;
  // Write the modified entry to file corresponding to key2.
  TF_ASSERT_OK(WriteBinaryProto(
      Env::Default(), GetFilePath(key2, persistor.persistent_cache_directory()),
      entry));

  MockCompilerClient mock_client;
  TF_ASSERT_OK_AND_ASSIGN(auto executable, BuildSampleExecutable());
  EXPECT_CALL(mock_client, LoadExecutable(_, _, serialized_executable_))
      .WillOnce(Return(ByMove(std::move(executable))));

  auto loaded_executable = persistor.TryToLoadExecutable(
      123, "signature", DefaultOptions(), compilation_result_mul, &mock_client);

  EXPECT_TRUE(loaded_executable.has_value());
  EXPECT_TRUE(loaded_executable->ok());
}

TEST_F(DeviceExecutionPersistorTest, LoadSerializedExecutableEmpty) {
  XlaDeviceExecutablePersistor::Config config(
      /*persistent_cache_directory=*/cache_dir_,
      /*disable_strict_signature_checks=*/false,
      /*persistence_prefix=*/"xla");
  XlaDeviceExecutablePersistor persistor(config, DefaultOptions().device_type);

  auto key =
      CreateCacheKey(/*signature_hash=*/123, compilation_result_add_,
                     persistor.device_type(), persistor.persistence_prefix());

  // Read serialized entry.
  XlaSerializedCacheEntry entry;
  TF_ASSERT_OK(ReadTextOrBinaryProto(
      Env::Default(), GetFilePath(key, persistor.persistent_cache_directory()),
      &entry));
  entry.clear_executable();
  // Write entry to another file.
  TF_ASSERT_OK(WriteBinaryProto(
      Env::Default(), GetFilePath(key, persistor.persistent_cache_directory()),
      entry));

  MockCompilerClient mock_client;
  auto loaded_executable = persistor.TryToLoadExecutable(
      /*signature_hash=*/123, "signature", DefaultOptions(),
      compilation_result_add_, &mock_client);

  EXPECT_TRUE(loaded_executable.has_value());
  EXPECT_FALSE(loaded_executable->ok());
  EXPECT_THAT(loaded_executable.value(),
              testing::StatusIs(error::INVALID_ARGUMENT));
}

}  // namespace
}  // namespace tensorflow
