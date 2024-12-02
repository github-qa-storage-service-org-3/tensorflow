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
// limitations under the License.

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_C_LITERT_OP_CODE_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_C_LITERT_OP_CODE_H_

#include "tensorflow/lite/builtin_ops.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef enum {
  kLiteRtOpCodeTflAdd = kTfLiteBuiltinAdd,
  kLiteRtOpCodeTflAveragePool2d = kTfLiteBuiltinAveragePool2d,
  kLiteRtOpCodeTflConcatenation = kTfLiteBuiltinConcatenation,
  kLiteRtOpCodeTflConv2d = kTfLiteBuiltinConv2d,
  kLiteRtOpCodeTflDepthwiseConv2d = kTfLiteBuiltinDepthwiseConv2d,
  kLiteRtOpCodeTflDepthToSpace = kTfLiteBuiltinDepthToSpace,
  kLiteRtOpCodeTflDequantize = kTfLiteBuiltinDequantize,
  kLiteRtOpCodeTflEmbeddingLookup = kTfLiteBuiltinEmbeddingLookup,
  kLiteRtOpCodeTflFloor = kTfLiteBuiltinFloor,
  kLiteRtOpCodeTflFullyConnected = kTfLiteBuiltinFullyConnected,
  kLiteRtOpCodeTflHashtableLookup = kTfLiteBuiltinHashtableLookup,
  kLiteRtOpCodeTflL2Normalization = kTfLiteBuiltinL2Normalization,
  kLiteRtOpCodeTflL2Pool2d = kTfLiteBuiltinL2Pool2d,
  kLiteRtOpCodeTflLocalResponseNormalization =
      kTfLiteBuiltinLocalResponseNormalization,
  kLiteRtOpCodeTflLogistic = kTfLiteBuiltinLogistic,
  kLiteRtOpCodeTflLshProjection = kTfLiteBuiltinLshProjection,
  kLiteRtOpCodeTflLstm = kTfLiteBuiltinLstm,
  kLiteRtOpCodeTflMaxPool2d = kTfLiteBuiltinMaxPool2d,
  kLiteRtOpCodeTflMul = kTfLiteBuiltinMul,
  kLiteRtOpCodeTflRelu = kTfLiteBuiltinRelu,
  kLiteRtOpCodeTflReluN1To1 = kTfLiteBuiltinReluN1To1,
  kLiteRtOpCodeTflRelu6 = kTfLiteBuiltinRelu6,
  kLiteRtOpCodeTflReshape = kTfLiteBuiltinReshape,
  kLiteRtOpCodeTflResizeBilinear = kTfLiteBuiltinResizeBilinear,
  kLiteRtOpCodeTflRnn = kTfLiteBuiltinRnn,
  kLiteRtOpCodeTflSoftmax = kTfLiteBuiltinSoftmax,
  kLiteRtOpCodeTflSpaceToDepth = kTfLiteBuiltinSpaceToDepth,
  kLiteRtOpCodeTflSvdf = kTfLiteBuiltinSvdf,
  kLiteRtOpCodeTflTanh = kTfLiteBuiltinTanh,
  kLiteRtOpCodeTflConcatEmbeddings = kTfLiteBuiltinConcatEmbeddings,
  kLiteRtOpCodeTflSkipGram = kTfLiteBuiltinSkipGram,
  kLiteRtOpCodeTflCall = kTfLiteBuiltinCall,
  kLiteRtOpCodeTflCustom = kTfLiteBuiltinCustom,
  kLiteRtOpCodeTflEmbeddingLookupSparse = kTfLiteBuiltinEmbeddingLookupSparse,
  kLiteRtOpCodeTflPad = kTfLiteBuiltinPad,
  kLiteRtOpCodeTflUnidirectionalSequenceRnn =
      kTfLiteBuiltinUnidirectionalSequenceRnn,
  kLiteRtOpCodeTflGather = kTfLiteBuiltinGather,
  kLiteRtOpCodeTflBatchToSpaceNd = kTfLiteBuiltinBatchToSpaceNd,
  kLiteRtOpCodeTflSpaceToBatchNd = kTfLiteBuiltinSpaceToBatchNd,
  kLiteRtOpCodeTflTranspose = kTfLiteBuiltinTranspose,
  kLiteRtOpCodeTflMean = kTfLiteBuiltinMean,
  kLiteRtOpCodeTflSub = kTfLiteBuiltinSub,
  kLiteRtOpCodeTflDiv = kTfLiteBuiltinDiv,
  kLiteRtOpCodeTflSqueeze = kTfLiteBuiltinSqueeze,
  kLiteRtOpCodeTflUnidirectionalSequenceLstm =
      kTfLiteBuiltinUnidirectionalSequenceLstm,
  kLiteRtOpCodeTflStridedSlice = kTfLiteBuiltinStridedSlice,
  kLiteRtOpCodeTflBidirectionalSequenceRnn =
      kTfLiteBuiltinBidirectionalSequenceRnn,
  kLiteRtOpCodeTflExp = kTfLiteBuiltinExp,
  kLiteRtOpCodeTflTopkV2 = kTfLiteBuiltinTopkV2,
  kLiteRtOpCodeTflSplit = kTfLiteBuiltinSplit,
  kLiteRtOpCodeTflLogSoftmax = kTfLiteBuiltinLogSoftmax,
  kLiteRtOpCodeTflDelegate = kTfLiteBuiltinDelegate,
  kLiteRtOpCodeTflBidirectionalSequenceLstm =
      kTfLiteBuiltinBidirectionalSequenceLstm,
  kLiteRtOpCodeTflCast = kTfLiteBuiltinCast,
  kLiteRtOpCodeTflPrelu = kTfLiteBuiltinPrelu,
  kLiteRtOpCodeTflMaximum = kTfLiteBuiltinMaximum,
  kLiteRtOpCodeTflArgMax = kTfLiteBuiltinArgMax,
  kLiteRtOpCodeTflMinimum = kTfLiteBuiltinMinimum,
  kLiteRtOpCodeTflLess = kTfLiteBuiltinLess,
  kLiteRtOpCodeTflNeg = kTfLiteBuiltinNeg,
  kLiteRtOpCodeTflPadv2 = kTfLiteBuiltinPadv2,
  kLiteRtOpCodeTflGreater = kTfLiteBuiltinGreater,
  kLiteRtOpCodeTflGreaterEqual = kTfLiteBuiltinGreaterEqual,
  kLiteRtOpCodeTflLessEqual = kTfLiteBuiltinLessEqual,
  kLiteRtOpCodeTflSelect = kTfLiteBuiltinSelect,
  kLiteRtOpCodeTflSlice = kTfLiteBuiltinSlice,
  kLiteRtOpCodeTflSin = kTfLiteBuiltinSin,
  kLiteRtOpCodeTflTransposeConv = kTfLiteBuiltinTransposeConv,
  kLiteRtOpCodeTflSparseToDense = kTfLiteBuiltinSparseToDense,
  kLiteRtOpCodeTflTile = kTfLiteBuiltinTile,
  kLiteRtOpCodeTflExpandDims = kTfLiteBuiltinExpandDims,
  kLiteRtOpCodeTflEqual = kTfLiteBuiltinEqual,
  kLiteRtOpCodeTflNotEqual = kTfLiteBuiltinNotEqual,
  kLiteRtOpCodeTflLog = kTfLiteBuiltinLog,
  kLiteRtOpCodeTflSum = kTfLiteBuiltinSum,
  kLiteRtOpCodeTflSqrt = kTfLiteBuiltinSqrt,
  kLiteRtOpCodeTflRsqrt = kTfLiteBuiltinRsqrt,
  kLiteRtOpCodeTflShape = kTfLiteBuiltinShape,
  kLiteRtOpCodeTflPow = kTfLiteBuiltinPow,
  kLiteRtOpCodeTflArgMin = kTfLiteBuiltinArgMin,
  kLiteRtOpCodeTflFakeQuant = kTfLiteBuiltinFakeQuant,
  kLiteRtOpCodeTflReduceProd = kTfLiteBuiltinReduceProd,
  kLiteRtOpCodeTflReduceMax = kTfLiteBuiltinReduceMax,
  kLiteRtOpCodeTflPack = kTfLiteBuiltinPack,
  kLiteRtOpCodeTflLogicalOr = kTfLiteBuiltinLogicalOr,
  kLiteRtOpCodeTflOneHot = kTfLiteBuiltinOneHot,
  kLiteRtOpCodeTflLogicalAnd = kTfLiteBuiltinLogicalAnd,
  kLiteRtOpCodeTflLogicalNot = kTfLiteBuiltinLogicalNot,
  kLiteRtOpCodeTflUnpack = kTfLiteBuiltinUnpack,
  kLiteRtOpCodeTflReduceMin = kTfLiteBuiltinReduceMin,
  kLiteRtOpCodeTflFloorDiv = kTfLiteBuiltinFloorDiv,
  kLiteRtOpCodeTflReduceAny = kTfLiteBuiltinReduceAny,
  kLiteRtOpCodeTflSquare = kTfLiteBuiltinSquare,
  kLiteRtOpCodeTflZerosLike = kTfLiteBuiltinZerosLike,
  kLiteRtOpCodeTflFill = kTfLiteBuiltinFill,
  kLiteRtOpCodeTflFloorMod = kTfLiteBuiltinFloorMod,
  kLiteRtOpCodeTflRange = kTfLiteBuiltinRange,
  kLiteRtOpCodeTflResizeNearestNeighbor = kTfLiteBuiltinResizeNearestNeighbor,
  kLiteRtOpCodeTflLeakyRelu = kTfLiteBuiltinLeakyRelu,
  kLiteRtOpCodeTflSquaredDifference = kTfLiteBuiltinSquaredDifference,
  kLiteRtOpCodeTflMirrorPad = kTfLiteBuiltinMirrorPad,
  kLiteRtOpCodeTflAbs = kTfLiteBuiltinAbs,
  kLiteRtOpCodeTflSplitV = kTfLiteBuiltinSplitV,
  kLiteRtOpCodeTflUnique = kTfLiteBuiltinUnique,
  kLiteRtOpCodeTflCeil = kTfLiteBuiltinCeil,
  kLiteRtOpCodeTflReverseV2 = kTfLiteBuiltinReverseV2,
  kLiteRtOpCodeTflAddN = kTfLiteBuiltinAddN,
  kLiteRtOpCodeTflGatherNd = kTfLiteBuiltinGatherNd,
  kLiteRtOpCodeTflCos = kTfLiteBuiltinCos,
  kLiteRtOpCodeTflWhere = kTfLiteBuiltinWhere,
  kLiteRtOpCodeTflRank = kTfLiteBuiltinRank,
  kLiteRtOpCodeTflElu = kTfLiteBuiltinElu,
  kLiteRtOpCodeTflReverseSequence = kTfLiteBuiltinReverseSequence,
  kLiteRtOpCodeTflMatrixDiag = kTfLiteBuiltinMatrixDiag,
  kLiteRtOpCodeTflQuantize = kTfLiteBuiltinQuantize,
  kLiteRtOpCodeTflMatrixSetDiag = kTfLiteBuiltinMatrixSetDiag,
  kLiteRtOpCodeTflRound = kTfLiteBuiltinRound,
  kLiteRtOpCodeTflHardSwish = kTfLiteBuiltinHardSwish,
  kLiteRtOpCodeTflIf = kTfLiteBuiltinIf,
  kLiteRtOpCodeTflWhile = kTfLiteBuiltinWhile,
  kLiteRtOpCodeTflNonMaxSuppressionV4 = kTfLiteBuiltinNonMaxSuppressionV4,
  kLiteRtOpCodeTflNonMaxSuppressionV5 = kTfLiteBuiltinNonMaxSuppressionV5,
  kLiteRtOpCodeTflScatterNd = kTfLiteBuiltinScatterNd,
  kLiteRtOpCodeTflSelectV2 = kTfLiteBuiltinSelectV2,
  kLiteRtOpCodeTflDensify = kTfLiteBuiltinDensify,
  kLiteRtOpCodeTflSegmentSum = kTfLiteBuiltinSegmentSum,
  kLiteRtOpCodeTflBatchMatmul = kTfLiteBuiltinBatchMatmul,
  kLiteRtOpCodeTflPlaceholderForGreaterOpCodeTfls =
      kTfLiteBuiltinPlaceholderForGreaterOpCodes,
  kLiteRtOpCodeTflCumsum = kTfLiteBuiltinCumsum,
  kLiteRtOpCodeTflCallOnce = kTfLiteBuiltinCallOnce,
  kLiteRtOpCodeTflBroadcastTo = kTfLiteBuiltinBroadcastTo,
  kLiteRtOpCodeTflRfft2d = kTfLiteBuiltinRfft2d,
  kLiteRtOpCodeTflConv3d = kTfLiteBuiltinConv3d,
  kLiteRtOpCodeTflImag = kTfLiteBuiltinImag,
  kLiteRtOpCodeTflReal = kTfLiteBuiltinReal,
  kLiteRtOpCodeTflComplexAbs = kTfLiteBuiltinComplexAbs,
  kLiteRtOpCodeTflHashtable = kTfLiteBuiltinHashtable,
  kLiteRtOpCodeTflHashtableFind = kTfLiteBuiltinHashtableFind,
  kLiteRtOpCodeTflHashtableImport = kTfLiteBuiltinHashtableImport,
  kLiteRtOpCodeTflHashtableSize = kTfLiteBuiltinHashtableSize,
  kLiteRtOpCodeTflReduceAll = kTfLiteBuiltinReduceAll,
  kLiteRtOpCodeTflConv3dTranspose = kTfLiteBuiltinConv3dTranspose,
  kLiteRtOpCodeTflVarHandle = kTfLiteBuiltinVarHandle,
  kLiteRtOpCodeTflReadVariable = kTfLiteBuiltinReadVariable,
  kLiteRtOpCodeTflAssignVariable = kTfLiteBuiltinAssignVariable,
  kLiteRtOpCodeTflBroadcastArgs = kTfLiteBuiltinBroadcastArgs,
  kLiteRtOpCodeTflRandomStandardNormal = kTfLiteBuiltinRandomStandardNormal,
  kLiteRtOpCodeTflBucketize = kTfLiteBuiltinBucketize,
  kLiteRtOpCodeTflRandomUniform = kTfLiteBuiltinRandomUniform,
  kLiteRtOpCodeTflMultinomial = kTfLiteBuiltinMultinomial,
  kLiteRtOpCodeTflGelu = kTfLiteBuiltinGelu,
  kLiteRtOpCodeTflDynamicUpdateSlice = kTfLiteBuiltinDynamicUpdateSlice,
  kLiteRtOpCodeTflRelu0To1 = kTfLiteBuiltinRelu0To1,
  kLiteRtOpCodeTflUnsortedSegmentProd = kTfLiteBuiltinUnsortedSegmentProd,
  kLiteRtOpCodeTflUnsortedSegmentMax = kTfLiteBuiltinUnsortedSegmentMax,
  kLiteRtOpCodeTflUnsortedSegmentSum = kTfLiteBuiltinUnsortedSegmentSum,
  kLiteRtOpCodeTflAtan2 = kTfLiteBuiltinAtan2,
  kLiteRtOpCodeTflUnsortedSegmentMin = kTfLiteBuiltinUnsortedSegmentMin,
  kLiteRtOpCodeTflSign = kTfLiteBuiltinSign,
  kLiteRtOpCodeTflBitcast = kTfLiteBuiltinBitcast,
  kLiteRtOpCodeTflBitwiseXor = kTfLiteBuiltinBitwiseXor,
  kLiteRtOpCodeTflRightShift = kTfLiteBuiltinRightShift,
  kLiteRtOpCodeShloLogistic = kTfLiteBuiltinStablehloLogistic,
  kLiteRtOpCodeShloAdd = kTfLiteBuiltinStablehloAdd,
  kLiteRtOpCodeShloDivide = kTfLiteBuiltinStablehloDivide,
  kLiteRtOpCodeShloMultiply = kTfLiteBuiltinStablehloMultiply,
  kLiteRtOpCodeShloMaximum = kTfLiteBuiltinStablehloMaximum,
  kLiteRtOpCodeShloReshape = kTfLiteBuiltinStablehloReshape,
  kLiteRtOpCodeShloClamp = kTfLiteBuiltinStablehloClamp,
  kLiteRtOpCodeShloConcatenate = kTfLiteBuiltinStablehloConcatenate,
  kLiteRtOpCodeShloBroadcastInDim = kTfLiteBuiltinStablehloBroadcastInDim,
  kLiteRtOpCodeShloConvolution = kTfLiteBuiltinStablehloConvolution,
  kLiteRtOpCodeShloSlice = kTfLiteBuiltinStablehloSlice,
  kLiteRtOpCodeShloCustomCall = kTfLiteBuiltinStablehloCustomCall,
  kLiteRtOpCodeShloReduce = kTfLiteBuiltinStablehloReduce,
  kLiteRtOpCodeShloAbs = kTfLiteBuiltinStablehloAbs,
  kLiteRtOpCodeShloAnd = kTfLiteBuiltinStablehloAnd,
  kLiteRtOpCodeShloCosine = kTfLiteBuiltinStablehloCosine,
  kLiteRtOpCodeShloExponential = kTfLiteBuiltinStablehloExponential,
  kLiteRtOpCodeShloFloor = kTfLiteBuiltinStablehloFloor,
  kLiteRtOpCodeShloLog = kTfLiteBuiltinStablehloLog,
  kLiteRtOpCodeShloMinimum = kTfLiteBuiltinStablehloMinimum,
  kLiteRtOpCodeShloNegate = kTfLiteBuiltinStablehloNegate,
  kLiteRtOpCodeShloOr = kTfLiteBuiltinStablehloOr,
  kLiteRtOpCodeShloPower = kTfLiteBuiltinStablehloPower,
  kLiteRtOpCodeShloRemainder = kTfLiteBuiltinStablehloRemainder,
  kLiteRtOpCodeShloRsqrt = kTfLiteBuiltinStablehloRsqrt,
  kLiteRtOpCodeShloSelect = kTfLiteBuiltinStablehloSelect,
  kLiteRtOpCodeShloSubtract = kTfLiteBuiltinStablehloSubtract,
  kLiteRtOpCodeShloTanh = kTfLiteBuiltinStablehloTanh,
  kLiteRtOpCodeShloScatter = kTfLiteBuiltinStablehloScatter,
  kLiteRtOpCodeShloCompare = kTfLiteBuiltinStablehloCompare,
  kLiteRtOpCodeShloConvert = kTfLiteBuiltinStablehloConvert,
  kLiteRtOpCodeShloDynamicSlice = kTfLiteBuiltinStablehloDynamicSlice,
  kLiteRtOpCodeShloDynamicUpdateSlice =
      kTfLiteBuiltinStablehloDynamicUpdateSlice,
  kLiteRtOpCodeShloPad = kTfLiteBuiltinStablehloPad,
  kLiteRtOpCodeShloIota = kTfLiteBuiltinStablehloIota,
  kLiteRtOpCodeShloGeneral = kTfLiteBuiltinStablehloDotGeneral,
  kLiteRtOpCodeShloWindow = kTfLiteBuiltinStablehloReduceWindow,
  kLiteRtOpCodeShloSort = kTfLiteBuiltinStablehloSort,
  kLiteRtOpCodeShloWhile = kTfLiteBuiltinStablehloWhile,
  kLiteRtOpCodeShloGather = kTfLiteBuiltinStablehloGather,
  kLiteRtOpCodeShloTranspose = kTfLiteBuiltinStablehloTranspose,
  kLiteRtOpCodeTflDilate = kTfLiteBuiltinDilate,
  kLiteRtOpCodeShloRngBitGenerator = kTfLiteBuiltinStablehloRngBitGenerator,
  kLiteRtOpCodeTflReduceWindow = kTfLiteBuiltinReduceWindow,
  kLiteRtOpCodeShloComposite = kTfLiteBuiltinStablehloComposite,
} LiteRtOpCode;

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_C_LITERT_OP_CODE_H_
