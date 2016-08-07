# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
""" Tests for ExtractImagePatches gradient. """

from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


class ExtractImagePatchesGradTest(tf.test.TestCase):
    """ Gradient-checking for ExtractImagePatches op."""

    test_cases = [
        {
            'in_shape': [2, 5, 5, 3],
            'ksizes': [1, 1, 1, 1],
            'strides': [1, 2, 3, 1],
            'rates': [1, 1, 1, 1],
        },
        {
            'in_shape': [2, 7, 7, 3],
            'ksizes': [1, 3, 3, 1],
            'strides': [1, 1, 1, 1],
            'rates': [1, 1, 1, 1],
        },
        {
            'in_shape': [2, 8, 7, 3],
            'ksizes': [1, 2, 2, 1],
            'strides': [1, 1, 1, 1],
            'rates': [1, 1, 1, 1],
        },
        {
            'in_shape': [2, 7, 8, 3],
            'ksizes': [1, 3, 2, 1],
            'strides': [1, 4, 3, 1],
            'rates': [1, 1, 1, 1],
        },
        {
            'in_shape': [1, 15, 20, 3],
            'ksizes': [1, 4, 3, 1],
            'strides': [1, 1, 1, 1],
            'rates': [1, 2, 4, 1],
        },
        {
            'in_shape': [2, 7, 8, 1],
            'ksizes': [1, 3, 2, 1],
            'strides': [1, 3, 2, 1],
            'rates': [1, 2, 2, 1],
        },
        {
            'in_shape': [2, 8, 9, 4],
            'ksizes': [1, 2, 2, 1],
            'strides': [1, 4, 2, 1],
            'rates': [1, 3, 2, 1],
        },
    ]

    def testGradient(self):
        with self.test_session():
            for test in self.test_cases:
                in_shape = test['in_shape']
                in_val = tf.constant(np.random.random(in_shape),
                                     dtype=tf.float32)

                for padding in ['VALID', 'SAME']:
                    out_val = tf.extract_image_patches(in_val,
                                                       test['ksizes'],
                                                       test['strides'],
                                                       test['rates'],
                                                       padding)
                    out_shape = out_val.get_shape().as_list()

                    err = tf.test.compute_gradient_error(
                        in_val, in_shape, out_val, out_shape
                    )

                    print('extract_image_patches gradient err: %.4e' % err)
                    self.assertLess(err, 1e-4)

if __name__ == "__main__":
    tf.test.main()
