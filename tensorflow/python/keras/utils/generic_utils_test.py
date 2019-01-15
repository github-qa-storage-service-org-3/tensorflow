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
"""Tests for Keras generic Python utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import marshal
import numpy as np

from absl.testing import parameterized

from tensorflow.python import keras
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.platform import test


class HasArgTest(test.TestCase):

  def test_has_arg(self):

    def f_x(x):
      return x

    def f_x_args(x, *args):
      _ = args
      return x

    def f_x_kwargs(x, **kwargs):
      _ = kwargs
      return x

    self.assertTrue(generic_utils.has_arg(f_x, 'x', accept_all=False))
    self.assertFalse(generic_utils.has_arg(f_x, 'y', accept_all=False))
    self.assertTrue(generic_utils.has_arg(f_x_args, 'x', accept_all=False))
    self.assertFalse(generic_utils.has_arg(f_x_args, 'y', accept_all=False))
    self.assertTrue(generic_utils.has_arg(f_x_kwargs, 'x', accept_all=False))
    self.assertFalse(generic_utils.has_arg(f_x_kwargs, 'y', accept_all=False))
    self.assertTrue(generic_utils.has_arg(f_x_kwargs, 'y', accept_all=True))


class TestCustomObjectScope(test.TestCase):

  def test_custom_object_scope(self):

    def custom_fn():
      pass

    class CustomClass(object):
      pass

    with generic_utils.custom_object_scope(
        {'CustomClass': CustomClass, 'custom_fn': custom_fn}):
      act = keras.activations.get('custom_fn')
      self.assertEqual(act, custom_fn)
      cl = keras.regularizers.get('CustomClass')
      self.assertEqual(cl.__class__, CustomClass)


class SerializeKerasObjectTest(test.TestCase):

  def test_serialize_none(self):
    serialized = generic_utils.serialize_keras_object(None)
    self.assertEqual(serialized, None)
    deserialized = generic_utils.deserialize_keras_object(serialized)
    self.assertEqual(deserialized, None)


class FuncDumpAndLoadTest(test.TestCase, parameterized.TestCase):
  @parameterized.parameters(['simple_function', 'closured_function'])
  def test_func_dump_and_load(self, test_function_type):
    if test_function_type == 'simple_function':
      def test_func():
        return r'\u'

    elif test_function_type == 'closured_function':
      def get_test_func():
        x = r'\u'

        def test_func():
          return x
        return test_func
      test_func = get_test_func()
    else:
      raise Exception('Unknown test case for test_func_dump_and_load')

    serialized = generic_utils.func_dump(test_func)
    deserialized = generic_utils.func_load(serialized)
    self.assertEqual(deserialized.__code__, test_func.__code__)
    self.assertEqual(deserialized.__closure__, test_func.__closure__)
    self.assertEqual(deserialized.__defaults__, test_func.__defaults__)

  def test_func_dump_and_load_closure(self):
    y = 0
    test_func = lambda x: x + y
    serialized, _, closure = generic_utils.func_dump(test_func)
    deserialized = generic_utils.func_load(serialized, closure=closure)
    self.assertEqual(deserialized.__code__, test_func.__code__)
    self.assertEqual(deserialized.__closure__, test_func.__closure__)
    self.assertEqual(deserialized.__defaults__, test_func.__defaults__)


  @parameterized.parameters(
      [keras.activations.softmax, np.argmax, lambda x: x**2, lambda x: x])
  def test_func_dump_and_load_backwards_compat(self, test_func):
    serialized = marshal.dumps(test_func.__code__).decode(
        'raw_unicode_escape')

    deserialized = generic_utils.func_load(
        serialized, defaults=test_func.__defaults__)
    self.assertEqual(deserialized.__code__, test_func.__code__)
    self.assertEqual(deserialized.__closure__, test_func.__closure__)
    self.assertEqual(deserialized.__defaults__, test_func.__defaults__)


if __name__ == '__main__':
  test.main()
