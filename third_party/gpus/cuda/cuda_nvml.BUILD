licenses(["restricted"])  # NVIDIA proprietary license

filegroup(
    name = "include",
    srcs = glob([
        "include/**",
    ]),
)

cc_library(
    name = "headers",
    hdrs = [":include"],
    include_prefix = "third_party/gpus/cuda/nvml/include",
    includes = ["include"],
    strip_include_prefix = "include",
    visibility = ["@local_config_cuda//cuda:__pkg__"],
)
