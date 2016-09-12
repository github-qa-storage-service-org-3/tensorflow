# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for Substr op from string_ops."""

from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import numpy as np


class SubstrOpTest(tf.test.TestCase):

  def _testScalarString(self, dtype):
    test_string = b"Hello"
    position = np.array(1, dtype)
    length = np.array(3, dtype)
    expected_value = b"ell"

    substr_op = tf.substr(test_string, position, length)
    with self.test_session():
      substr = substr_op.eval()
      self.assertAllEqual(substr, expected_value)

  def _testVectorStrings(self, dtype):
    test_string = [b"Hello", b"World"]
    position = np.array(1, dtype)
    length = np.array(3, dtype)
    expected_value = [b"ell", b"orl"]

    substr_op = tf.substr(test_string, position, length)
    with self.test_session():
      substr = substr_op.eval()
      self.assertAllEqual(substr, expected_value)

  def _testMatrixStrings(self, dtype):
    test_string = [[b"ten", b"eleven", b"twelve"],
                   [b"thirteen", b"fourteen", b"fifteen"],
                   [b"sixteen", b"seventeen", b"eighteen"]]
    position = np.array(1, dtype)
    length = np.array(4, dtype)
    expected_value = [[b"en", b"leve", b"welv"],
                      [b"hirt", b"ourt", b"ifte"],
                      [b"ixte", b"even", b"ight"]]

    substr_op = tf.substr(test_string, position, length)
    with self.test_session():
      substr = substr_op.eval()
      self.assertAllEqual(substr, expected_value)

  def _testOutOfRangeError(self, dtype):
    test_string = b"Hello"
    position = np.array(7, dtype)
    length = np.array(3, dtype)
    substr_op = tf.substr(test_string, position, length)
    with self.test_session():
      with self.assertRaises(tf.errors.InvalidArgumentError):
        substr = substr_op.eval()

    test_string = [b"good", b"good", b"bad", b"good"]
    position = np.array(3, dtype)
    length = np.array(1, dtype)
    substr_op = tf.substr(test_string, position, length)
    with self.test_session():
      with self.assertRaises(tf.errors.InvalidArgumentError):
        substr = substr_op.eval()

    test_string = b"Hello"
    position = np.array(-1, dtype)
    length = np.array(3, dtype)
    substr_op = tf.substr(test_string, position, length)
    with self.test_session():
      with self.assertRaises(tf.errors.InvalidArgumentError):
        substr = substr_op.eval()

  def _testAll(self, dtype):
    self._testScalarString(dtype)
    self._testVectorStrings(dtype)
    self._testMatrixStrings(dtype)
    self._testOutOfRangeError(dtype)

  def testInt32(self):
    self._testAll(np.int32)
 
  def testInt64(self):
    self._testAll(np.int64)

  def testWrongDtype(self):
    with self.test_session():
      with self.assertRaises(TypeError):
        tf.substr(b"test", 3.0, 1)
      with self.assertRaises(TypeError):
        tf.substr(b"test", 3, 1.0)

      

      
if __name__ == "__main__":
  tf.test.main()
