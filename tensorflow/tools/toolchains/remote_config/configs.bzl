"""Configurations of RBE builds used with remote config."""

load("//tensorflow/tools/toolchains/remote_config:rbe_config.bzl", "sigbuild_tf_configs", "tensorflow_local_config", "tensorflow_rbe_config", "tensorflow_rbe_win_config")

def initialize_rbe_configs():
    tensorflow_local_config(
        name = "local_execution",
    )

    tensorflow_rbe_config(
        name = "ubuntu20.04-clang_manylinux2014-cuda12.3-cudnn8.9",
        compiler = "/usr/lib/llvm-18/bin/clang",
        cuda_version = "12.3.2",
        cudnn_version = "8.9.7.29",
        os = "ubuntu20.04-manylinux2014-multipython",
    )

    tensorflow_rbe_config(
        name = "ubuntu20.04-clang_manylinux2014-cuda12.3-cudnn9.1",
        compiler = "/usr/lib/llvm-18/bin/clang",
        cuda_version = "12.3.2",
        cudnn_version = "9.1.1",
        os = "ubuntu20.04-manylinux2014-multipython",
    )

    tensorflow_rbe_config(
        name = "ubuntu20.04-gcc9_manylinux2014-cuda12.3-cudnn8.9",
        compiler = "/dt9/usr/bin/gcc",
        compiler_prefix = "/usr/bin",
        cuda_version = "12.3.2",
        cudnn_version = "8.9.7.29",
        os = "ubuntu20.04-manylinux2014-multipython",
    )

    tensorflow_rbe_config(
        name = "ubuntu22.04-clang_manylinux2014-cuda12.3-cudnn8.9",
        compiler = "/usr/lib/llvm-18/bin/clang",
        cuda_version = "12.3.2",
        cudnn_version = "8.9.7.29",
        os = "ubuntu22.04-manylinux2014-multipython",
    )

    tensorflow_rbe_config(
        name = "ubuntu22.04-gcc9_manylinux2014-cuda12.3-cudnn8.9",
        compiler = "/dt9/usr/bin/gcc",
        compiler_prefix = "/usr/bin",
        cuda_version = "12.3.2",
        cudnn_version = "8.9.7.29",
        os = "ubuntu22.04-manylinux2014-multipython",
    )

    tensorflow_rbe_win_config(
        name = "windows_py37",
        python_bin_path = "C:/Python37/python.exe",
    )

    # TF-Version-Specific SIG Build RBE Configs. The crosstool generated from these
    # configs are python-version-independent because they only care about the
    # tooling paths; the container mapping is useful only so that TF RBE users
    # may specify a specific Python version container. Yes, we could use the tag name instead,
    # but for vague security reasons we're obligated to use the pinned hash and update manually.
    # The name_container_map is helpfully auto-generated by a GitHub Action. You have to run it
    # manually. See go/tf-devinfra/docker#how-do-i-update-rbe-images

    sigbuild_tf_configs(
        name_container_map = {
            "sigbuild-r2.16": "docker://gcr.io/tensorflow-sigs/build@sha256:842a5ba84d3658c5bf1f8a31e16284f7becc35409da0dfd71816afa3cd28d728",
            "sigbuild-r2.16-python3.9": "docker://gcr.io/tensorflow-sigs/build@sha256:22d863e6fe3f98946015b9e1264b2eeb8e56e504535a6c1d5e564cae65ae5d37",
            "sigbuild-r2.16-python3.10": "docker://gcr.io/tensorflow-sigs/build@sha256:da15288c8464153eadd35da720540a544b76aa9d78cceb42a6821b2f3e70a0fa",
            "sigbuild-r2.16-python3.11": "docker://gcr.io/tensorflow-sigs/build@sha256:842a5ba84d3658c5bf1f8a31e16284f7becc35409da0dfd71816afa3cd28d728",
            "sigbuild-r2.16-python3.12": "docker://gcr.io/tensorflow-sigs/build@sha256:40fcd1d05c672672b599d9cb3784dcf379d6aa876f043b46c6ab18237d5d4e10",
        },
        # Unclear why LIBC is set to 2.19 here, and yet manylinux2010 is 2.12
        # and manylinux2014 is 2.17.
        env = {
            "ABI_LIBC_VERSION": "glibc_2.19",
            "ABI_VERSION": "gcc",
            "BAZEL_COMPILER": "/dt9/usr/bin/gcc",
            "BAZEL_HOST_SYSTEM": "i686-unknown-linux-gnu",
            "BAZEL_TARGET_CPU": "k8",
            "BAZEL_TARGET_LIBC": "glibc_2.19",
            "BAZEL_TARGET_SYSTEM": "x86_64-unknown-linux-gnu",
            "CC": "/dt9/usr/bin/gcc",
            "CC_TOOLCHAIN_NAME": "linux_gnu_x86",
            "CLEAR_CACHE": "1",
            "GCC_HOST_COMPILER_PATH": "/dt9/usr/bin/gcc",
            "GCC_HOST_COMPILER_PREFIX": "/usr/bin",
            "HOST_CXX_COMPILER": "/dt9/usr/bin/gcc",
            "HOST_C_COMPILER": "/dt9/usr/bin/gcc",
            "TF_ENABLE_XLA": "1",
        },
    )

    sigbuild_tf_configs(
        name_container_map = {
            "sigbuild-r2.16-clang": "docker://gcr.io/tensorflow-sigs/build@sha256:842a5ba84d3658c5bf1f8a31e16284f7becc35409da0dfd71816afa3cd28d728",
            "sigbuild-r2.16-clang-python3.9": "docker://gcr.io/tensorflow-sigs/build@sha256:22d863e6fe3f98946015b9e1264b2eeb8e56e504535a6c1d5e564cae65ae5d37",
            "sigbuild-r2.16-clang-python3.10": "docker://gcr.io/tensorflow-sigs/build@sha256:da15288c8464153eadd35da720540a544b76aa9d78cceb42a6821b2f3e70a0fa",
            "sigbuild-r2.16-clang-python3.11": "docker://gcr.io/tensorflow-sigs/build@sha256:842a5ba84d3658c5bf1f8a31e16284f7becc35409da0dfd71816afa3cd28d728",
            "sigbuild-r2.16-clang-python3.12": "docker://gcr.io/tensorflow-sigs/build@sha256:40fcd1d05c672672b599d9cb3784dcf379d6aa876f043b46c6ab18237d5d4e10",
        },
        # Unclear why LIBC is set to 2.19 here, and yet manylinux2010 is 2.12
        # and manylinux2014 is 2.17.
        env = {
            "ABI_LIBC_VERSION": "glibc_2.19",
            "ABI_VERSION": "gcc",
            "BAZEL_COMPILER": "/usr/lib/llvm-17/bin/clang",
            "BAZEL_HOST_SYSTEM": "i686-unknown-linux-gnu",
            "BAZEL_TARGET_CPU": "k8",
            "BAZEL_TARGET_LIBC": "glibc_2.19",
            "BAZEL_TARGET_SYSTEM": "x86_64-unknown-linux-gnu",
            "CC": "/usr/lib/llvm-17/bin/clang",
            "CC_TOOLCHAIN_NAME": "linux_gnu_x86",
            "CLEAR_CACHE": "1",
            "HOST_CXX_COMPILER": "/usr/lib/llvm-17/bin/clang",
            "HOST_C_COMPILER": "/usr/lib/llvm-17/bin/clang",
            "TF_ENABLE_XLA": "1",
        },
    )

    sigbuild_tf_configs(
        name_container_map = {
            "sigbuild-r2.17": "docker://gcr.io/tensorflow-sigs/build@sha256:b6f572a897a69fa3311773f949b9aa9e81bc393e4fbe2c0d56d8afb03a6de080",
            "sigbuild-r2.17-python3.9": "docker://gcr.io/tensorflow-sigs/build@sha256:d0f27a4c7b97dbe9d530703dca3449afd464758e56b3ac4e1609c701223a0572",
            "sigbuild-r2.17-python3.10": "docker://gcr.io/tensorflow-sigs/build@sha256:64e68a1d65ac265a2a59c8c2f6eb1f2148a323048a679a08e53239d467fa1478",
            "sigbuild-r2.17-python3.11": "docker://gcr.io/tensorflow-sigs/build@sha256:b6f572a897a69fa3311773f949b9aa9e81bc393e4fbe2c0d56d8afb03a6de080",
            "sigbuild-r2.17-python3.12": "docker://gcr.io/tensorflow-sigs/build@sha256:8b856ad736147bb9c8bc9e1ec2c8e1ab17d36397905da7a5b63dadeff9310f0c",
        },
        # Unclear why LIBC is set to 2.19 here, and yet manylinux2010 is 2.12
        # and manylinux2014 is 2.17.
        env = {
            "ABI_LIBC_VERSION": "glibc_2.19",
            "ABI_VERSION": "gcc",
            "BAZEL_COMPILER": "/dt9/usr/bin/gcc",
            "BAZEL_HOST_SYSTEM": "i686-unknown-linux-gnu",
            "BAZEL_TARGET_CPU": "k8",
            "BAZEL_TARGET_LIBC": "glibc_2.19",
            "BAZEL_TARGET_SYSTEM": "x86_64-unknown-linux-gnu",
            "CC": "/dt9/usr/bin/gcc",
            "CC_TOOLCHAIN_NAME": "linux_gnu_x86",
            "CLEAR_CACHE": "1",
            "GCC_HOST_COMPILER_PATH": "/dt9/usr/bin/gcc",
            "GCC_HOST_COMPILER_PREFIX": "/usr/bin",
            "HOST_CXX_COMPILER": "/dt9/usr/bin/gcc",
            "HOST_C_COMPILER": "/dt9/usr/bin/gcc",
            "TF_ENABLE_XLA": "1",
        },
    )

    sigbuild_tf_configs(
        name_container_map = {
            "sigbuild-r2.17-clang": "docker://gcr.io/tensorflow-sigs/build@sha256:2d737fc9fe931507a89927eee792b1bb934215e6aaae58b1941586e3400e2645",
            "sigbuild-r2.17-clang-python3.9": "docker://gcr.io/tensorflow-sigs/build@sha256:0e9cd35dafd2b91bc59cea377ad84a6089ba5e5542709c6e80ada3f366fd2338",
            "sigbuild-r2.17-clang-python3.10": "docker://gcr.io/tensorflow-sigs/build@sha256:89730ded5c2268e53238bf8c5fb6d162b105baa31ab0480a94a7d0204203de66",
            "sigbuild-r2.17-clang-python3.11": "docker://gcr.io/tensorflow-sigs/build@sha256:2d737fc9fe931507a89927eee792b1bb934215e6aaae58b1941586e3400e2645",
            "sigbuild-r2.17-clang-python3.12": "docker://gcr.io/tensorflow-sigs/build@sha256:45ea78e79305f91cdae5a26094f80233bba54bbfbc612623381012f097035b9a",
        },
        # Unclear why LIBC is set to 2.19 here, and yet manylinux2010 is 2.12
        # and manylinux2014 is 2.17.
        env = {
            "ABI_LIBC_VERSION": "glibc_2.19",
            "ABI_VERSION": "gcc",
            "BAZEL_COMPILER": "/usr/lib/llvm-18/bin/clang",
            "BAZEL_HOST_SYSTEM": "i686-unknown-linux-gnu",
            "BAZEL_TARGET_CPU": "k8",
            "BAZEL_TARGET_LIBC": "glibc_2.19",
            "BAZEL_TARGET_SYSTEM": "x86_64-unknown-linux-gnu",
            "CC": "/usr/lib/llvm-18/bin/clang",
            "CC_TOOLCHAIN_NAME": "linux_gnu_x86",
            "CLEAR_CACHE": "1",
            "HOST_CXX_COMPILER": "/usr/lib/llvm-18/bin/clang",
            "HOST_C_COMPILER": "/usr/lib/llvm-18/bin/clang",
            "TF_ENABLE_XLA": "1",
        },
    )

    sigbuild_tf_configs(
        name_container_map = {
            "sigbuild-r2.17-clang-cudnn9": "docker://gcr.io/tensorflow-sigs/build@sha256:daa5bdd802fe3def188e2200ed707c73d278f6f1930bf26c933d6ba041b0e027",
            "sigbuild-r2.17-clang-cudnn9-python3.9": "docker://gcr.io/tensorflow-sigs/build@sha256:1c4f06b98ab1ad092facf2d6fcac9f7496bd599a67ad998b82d80e98ef7defa8",
            "sigbuild-r2.17-clang-cudnn9-python3.10": "docker://gcr.io/tensorflow-sigs/build@sha256:c3df6982305d70dfb44cbfbedee3465782d6cbf791f7920e6246de0140216da0",
            "sigbuild-r2.17-clang-cudnn9-python3.11": "docker://gcr.io/tensorflow-sigs/build@sha256:daa5bdd802fe3def188e2200ed707c73d278f6f1930bf26c933d6ba041b0e027",
            "sigbuild-r2.17-clang-cudnn9-python3.12": "docker://gcr.io/tensorflow-sigs/build@sha256:23e477895dd02e45df1056d4a0a9c4229dec3a20c23fb2f3fb5832ecbd0a29bc",
        },
        # Unclear why LIBC is set to 2.19 here, and yet manylinux2010 is 2.12
        # and manylinux2014 is 2.17.
        env = {
            "ABI_LIBC_VERSION": "glibc_2.19",
            "ABI_VERSION": "gcc",
            "BAZEL_COMPILER": "/usr/lib/llvm-18/bin/clang",
            "BAZEL_HOST_SYSTEM": "i686-unknown-linux-gnu",
            "BAZEL_TARGET_CPU": "k8",
            "BAZEL_TARGET_LIBC": "glibc_2.19",
            "BAZEL_TARGET_SYSTEM": "x86_64-unknown-linux-gnu",
            "CC": "/usr/lib/llvm-18/bin/clang",
            "CC_TOOLCHAIN_NAME": "linux_gnu_x86",
            "CLEAR_CACHE": "1",
            "HOST_CXX_COMPILER": "/usr/lib/llvm-18/bin/clang",
            "HOST_C_COMPILER": "/usr/lib/llvm-18/bin/clang",
            "TF_ENABLE_XLA": "1",
        },
    )
