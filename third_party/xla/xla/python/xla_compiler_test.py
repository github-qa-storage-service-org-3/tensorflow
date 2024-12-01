# Copyright 2024 The OpenXLA Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from absl.testing import absltest
import numpy as np
from xla.python import xla_extension


class LiteralTest(absltest.TestCase):
  def test_create_literal_from_ndarray_rank_1(self):
    input_array = np.array(range(10), dtype=np.float32)
    shape = xla_extension.Shape.array_shape(
        input_array.dtype, input_array.shape
    )
    literal = xla_extension.Literal(shape)
    array = literal.numpy_view()
    np.copyto(array, input_array)
    # Intentionally check against `np.from_dlpack(literal)`` instead of `array``
    # to ensure that the underlying literal is actually updated and not some
    # rebinding to a new object.
    np.testing.assert_array_equal(np.from_dlpack(literal), input_array)

  def test_create_literal_from_ndarray_rank_2(self):
    input_array = np.array(range(100), dtype=np.float32).reshape(20, 5)
    shape = xla_extension.Shape.array_shape(
        input_array.dtype, input_array.shape, [1, 0]
    )
    literal = xla_extension.Literal(shape)
    array = literal.numpy_view()
    np.copyto(array, input_array)
    np.testing.assert_array_equal(np.from_dlpack(literal), input_array)

  def test_create_literal_from_ndarray_rank_2_reverse_layout(self):
    input_array = np.array(range(100), dtype=np.float32).reshape(25, 4)
    shape = xla_extension.Shape.array_shape(
        input_array.dtype, input_array.shape, [0, 1]
    )
    literal = xla_extension.Literal(shape)
    array = literal.numpy_view()
    np.copyto(array, input_array)
    np.testing.assert_array_equal(np.from_dlpack(literal), input_array)


if __name__ == "__main__":
  absltest.main()
