#include <algorithm>
#include <limits>

#include "tensorflow/compiler/plugin/poplar/driver/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/util.h"
#include "tensorflow/compiler/plugin/poplar/driver/vertex_templates.h"

#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_query.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/window_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/util/bcast.h"
#include "tensorflow/stream_executor/lib/strcat.h"

#include <poplar/Engine.hpp>
#include <poplar/Graph.hpp>
#include <popnn/Pooling.hpp>
#include <popnn/PoolingDef.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Pad.hpp>
#include <popops/Reduce.hpp>

namespace xla {
namespace poplarplugin {

static poplar::Tensor duplicate(poplar::Graph& graph,
                                poplar::program::Sequence& sequence,
                                poplar::Tensor t) {
  poplar::Tensor result = graph.clone(t);
  sequence.add(poplar::program::Copy(t, result));

  return result;
}

StatusOr<poplar::program::Program> CreateSort(poplar::Graph& graph,
                                              CompilerResources& res,
                                              const HloInstruction* inst,
                                              TensorMap& tensor_map) {
  const HloSortInstruction* sort = Cast<HloSortInstruction>(inst);

  if (sort->operand_count() == 1) {
    TF_ASSIGN_OR_RETURN(poplar::Tensor to_sort,
                        FindInstructionInput(tensor_map, inst, 0));

    poplar::program::Sequence prog;
    poplar::Tensor to_sort_duplicate = duplicate(graph, prog, to_sort);

    TF_ASSIGN_OR_RETURN(
        poplar::program::Sequence sort_prog,
        CreateSort(graph, to_sort_duplicate, sort->dimensions(0)));

    prog.add(sort_prog);
    TF_CHECK_OK(AddOutputTensor(graph, res, prog, tensor_map, inst, 0,
                                to_sort_duplicate)
                    .status());

    return prog;
  } else {
    TF_ASSIGN_OR_RETURN(poplar::Tensor key,
                        FindInstructionInput(tensor_map, inst, 0));
    TF_ASSIGN_OR_RETURN(poplar::Tensor value,
                        FindInstructionInput(tensor_map, inst, 1));

    poplar::program::Sequence prog;
    poplar::Tensor key_dup = duplicate(graph, prog, key);
    poplar::Tensor value_dup = duplicate(graph, prog, value);

    TF_ASSIGN_OR_RETURN(
        poplar::program::Sequence sort_prog,
        CreateSort(graph, key_dup, value_dup, sort->dimensions(0)));

    prog.add(sort_prog);
    TF_CHECK_OK(AddOutputTensor(graph, res, prog, tensor_map, inst, 0, key_dup)
                    .status());

    TF_CHECK_OK(
        AddOutputTensor(graph, res, prog, tensor_map, inst, 1, value_dup)
            .status());

    return prog;
  }
}

static poplar::Tensor flatten_dimension(poplar::Tensor input,
                                        const int64 dimension) {
  std::vector<unsigned> permutation(input.rank());
  std::iota(permutation.begin(), permutation.end(), 0);
  std::swap(permutation.back(), permutation[dimension]);
  poplar::Tensor input_view = input.dimShuffle(permutation);
  return input_view.reshape(
      {input.numElements() / input.dim(dimension), input.dim(dimension)});
}

static bool interval_comp(const poplar::Interval& a,
                          const poplar::Interval& b) {
  return a.begin() < b.begin();
}

static bool interval_not_empty(const poplar::Interval& a) {
  return a.size() != 0;
}

static poplar::program::Program Swap(poplar::Graph& graph, poplar::Tensor a,
                                     poplar::Tensor b) {
  poplar::program::Sequence result;
  poplar::Tensor a_tmp = graph.addVariable(a.elementType(), a.shape());
  poplar::Tensor b_tmp = graph.addVariable(a.elementType(), a.shape());
  graph.setTileMapping(a_tmp, graph.getTileMapping(a));
  graph.setTileMapping(b_tmp, graph.getTileMapping(b));

  result.add(poplar::program::Copy(a, a_tmp));
  result.add(poplar::program::Copy(b, b_tmp));
  result.add(poplar::program::Copy(a_tmp, b));
  result.add(poplar::program::Copy(b_tmp, a));

  return result;
}

static std::string HeapSortVertex(poplar::Type a) {
  return "HeapSortVertex<" + a.toString() + ">";
}

static std::string HeapSortVertex(poplar::Type a, poplar::Type b) {
  return "HeapSortVertexKV<" + a.toString() + "," + b.toString() + ">";
}

static poplar::program::Sequence sort_slice(poplar::Graph& graph,
                                            poplar::ComputeSet& sort_cs,
                                            poplar::Tensor in) {
  const std::string vertex_type = HeapSortVertex(in.elementType());

  for (std::size_t i = 0; i < in.dim(0); ++i) {
    poplar::Tensor input_slice = in[i];

    const auto tile_intervals = graph.getTileMapping(input_slice);
    for (std::size_t tile = 0; tile < tile_intervals.size(); ++tile) {
      for (const auto& interval : tile_intervals[tile]) {
        auto v = graph.addVertex(sort_cs, vertex_type);
        graph.setTileMapping(v, tile);
        graph.setCycleEstimate(v, 20 * interval.size());

        graph.connect(v["out"], input_slice.slice(interval));
      }
    }
  }

  return poplar::program::Sequence();
}

static poplar::program::Sequence sort_slice(poplar::Graph& graph,
                                            poplar::ComputeSet& sort_cs,
                                            poplar::Tensor key,
                                            poplar::Tensor value) {
  const std::string vertex_type =
      HeapSortVertex(key.elementType(), value.elementType());

  for (std::size_t i = 0; i < key.dim(0); ++i) {
    poplar::Tensor key_slice = key[i];
    poplar::Tensor value_slice = value[i];

    const auto tile_intervals = graph.getTileMapping(key_slice);
    for (std::size_t tile = 0; tile < tile_intervals.size(); ++tile) {
      for (const auto& interval : tile_intervals[tile]) {
        auto v = graph.addVertex(sort_cs, vertex_type);
        graph.setTileMapping(v, tile);
        graph.setCycleEstimate(v, 20 * interval.size());

        graph.connect(v["key"], key_slice.slice(interval));
        graph.connect(v["value"], value_slice.slice(interval));
      }
    }
  }

  return poplar::program::Sequence();
}

template <typename T>
static std::vector<T> flatten(const std::vector<std::vector<T>>& input) {
  std::vector<T> result;

  for (const auto& a : input) {
    for (const auto& b : a) {
      result.push_back(b);
    }
  }

  return result;
}

static poplar::Tensor is_sorted_predicate(poplar::Graph& graph,
                                          poplar::program::Sequence& prog,
                                          poplar::Tensor input) {
  poplar::Tensor result = graph.addConstant(poplar::BOOL, {}, true);

  for (std::size_t i = 0; i < input.dim(0); ++i) {
    poplar::Tensor input_slice = input[i];

    auto intervals = flatten(graph.getTileMapping(input_slice));
    const auto new_end =
        std::partition(intervals.begin(), intervals.end(), interval_not_empty);
    intervals.erase(new_end, intervals.end());
    std::sort(intervals.begin(), intervals.end(), interval_comp);

    if (intervals.size() > 0) {
      for (std::size_t k = 0; k < intervals.size() - 1; ++k) {
        poplar::Tensor l_max = input_slice[intervals[k].end() - 1];
        poplar::Tensor r_min = input_slice[intervals[k + 1].begin()];

        result = popops::logicalAnd(
            graph, result, popops::lteq(graph, l_max, r_min, prog), prog);
      }
    }
  }

  return result;
}

static poplar::program::Sequence create_exchange(
    poplar::Graph& graph, poplar::Tensor input, const std::size_t start_index) {
  poplar::program::Sequence result;

  for (std::size_t i = 0; i < input.dim(0); ++i) {
    poplar::Tensor input_slice = input[i];

    auto intervals = flatten(graph.getTileMapping(input_slice));
    const auto new_end =
        std::partition(intervals.begin(), intervals.end(), interval_not_empty);
    intervals.erase(new_end, intervals.end());
    std::sort(intervals.begin(), intervals.end(), interval_comp);

    if (intervals.size() > 0) {
      for (std::size_t k = start_index; k < intervals.size() - 1; k += 2) {
        poplar::Tensor l_max = input_slice[intervals[k].end() - 1];
        poplar::Tensor r_min = input_slice[intervals[k + 1].begin()];

        poplar::Tensor predicate = popops::lteq(graph, l_max, r_min, result);

        result.add(poplar::program::If(predicate, poplar::program::Sequence(),
                                       Swap(graph, l_max, r_min)));
      }
    }
  }

  return result;
}

static poplar::program::Sequence create_exchange(
    poplar::Graph& graph, poplar::Tensor key, poplar::Tensor value,
    const std::size_t start_index) {
  poplar::program::Sequence result;

  for (std::size_t i = 0; i < key.dim(0); ++i) {
    poplar::Tensor key_slice = key[i];
    poplar::Tensor value_slice = value[i];

    auto intervals = flatten(graph.getTileMapping(key_slice));
    const auto new_end =
        std::partition(intervals.begin(), intervals.end(), interval_not_empty);
    intervals.erase(new_end, intervals.end());
    std::sort(intervals.begin(), intervals.end(), interval_comp);

    if (intervals.size() > 0) {
      for (std::size_t k = start_index; k < intervals.size() - 1; k += 2) {
        poplar::Tensor l_max = key_slice[intervals[k].end() - 1];
        poplar::Tensor r_min = key_slice[intervals[k + 1].begin()];

        poplar::Tensor predicate = popops::lteq(graph, l_max, r_min, result);

        poplar::program::Sequence swap_elems;
        swap_elems.add(Swap(graph, l_max, r_min));

        l_max = value_slice[intervals[k].end() - 1];
        r_min = value_slice[intervals[k + 1].begin()];
        swap_elems.add(Swap(graph, l_max, r_min));
        result.add(poplar::program::If(predicate, poplar::program::Sequence(),
                                       swap_elems));
      }
    }
  }

  return result;
}

static poplar::program::Sequence create_even_exchange(poplar::Graph& graph,
                                                      poplar::Tensor input) {
  return create_exchange(graph, input, 0);
}

static poplar::program::Sequence create_odd_exchange(poplar::Graph& graph,
                                                     poplar::Tensor input) {
  return create_exchange(graph, input, 1);
}

static poplar::program::Sequence create_even_exchange(poplar::Graph& graph,
                                                      poplar::Tensor key,
                                                      poplar::Tensor value) {
  return create_exchange(graph, key, value, 0);
}

static poplar::program::Sequence create_odd_exchange(poplar::Graph& graph,
                                                     poplar::Tensor key,
                                                     poplar::Tensor value) {
  return create_exchange(graph, key, value, 1);
}

StatusOr<poplar::program::Sequence> CreateSort(poplar::Graph& graph,
                                               poplar::Tensor input,
                                               const int64 dimension,
                                               const std::string& debug_name) {
  poplar::program::Sequence seq;

  poplar::Tensor input_view = flatten_dimension(input, dimension);

  auto sort_cs = graph.addComputeSet(debug_name);

  seq.add(sort_slice(graph, sort_cs, input_view));

  poplar::program::Sequence sort_step;
  sort_step.add(create_even_exchange(graph, input_view));
  sort_step.add(poplar::program::Execute(sort_cs));
  sort_step.add(create_odd_exchange(graph, input_view));
  sort_step.add(poplar::program::Execute(sort_cs));

  poplar::program::Sequence cond;
  poplar::Tensor pred = is_sorted_predicate(graph, cond, input_view);
  poplar::program::RepeatWhileFalse repeat(cond, pred, sort_step);

  seq.add(poplar::program::Execute(sort_cs));
  seq.add(repeat);
  return seq;
}

StatusOr<poplar::program::Sequence> CreateSort(poplar::Graph& graph,
                                               poplar::Tensor key,
                                               poplar::Tensor value,
                                               const int64 dimension,
                                               const std::string& debug_name) {
  poplar::program::Sequence seq;

  poplar::Tensor key_view = flatten_dimension(key, dimension);
  poplar::Tensor value_view = flatten_dimension(value, dimension);

  auto sort_cs = graph.addComputeSet(debug_name);

  seq.add(sort_slice(graph, sort_cs, key_view, value_view));

  poplar::program::Sequence sort_step;
  sort_step.add(create_even_exchange(graph, key_view, value_view));
  sort_step.add(poplar::program::Execute(sort_cs));
  sort_step.add(create_odd_exchange(graph, key_view, value_view));
  sort_step.add(poplar::program::Execute(sort_cs));

  poplar::program::Sequence cond;
  poplar::Tensor pred = is_sorted_predicate(graph, cond, key_view);
  poplar::program::RepeatWhileFalse repeat(cond, pred, sort_step);

  seq.add(poplar::program::Execute(sort_cs));
  seq.add(repeat);
  return seq;
}

}  // namespace poplarplugin
}  // namespace xla
