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
"""Tests for tf.layers.normalization."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np

from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.layers import convolutional as conv_layers
from tensorflow.python.layers import normalization as normalization_layers
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import gradient_descent
from tensorflow.python.training import saver as saver_lib


class BNTest(test.TestCase):

  def _simple_model(self, image, fused, freeze_mode):
    output_channels, kernel_size = 2, 3
    conv = conv_layers.conv2d(
        image,
        output_channels,
        kernel_size,
        use_bias=False,
        kernel_initializer=init_ops.ones_initializer())
    bn_layer = normalization_layers.BatchNormalization(fused=fused)
    bn_layer._bessels_correction_test_only = False
    training = not freeze_mode
    bn = bn_layer.apply(conv, training=training)
    loss = math_ops.reduce_sum(math_ops.abs(bn))
    optimizer = gradient_descent.GradientDescentOptimizer(0.01)
    if not freeze_mode:
      update_ops = ops.get_collection(ops.GraphKeys.UPDATE_OPS)
      with ops.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss)
    else:
      train_op = optimizer.minimize(loss)
    saver = saver_lib.Saver(write_version=saver_pb2.SaverDef.V2)
    return loss, train_op, saver

  def _train(self,
             checkpoint_path,
             shape,
             use_gpu,
             is_fused,
             restore=False,
             freeze_mode=False):
    ops.reset_default_graph()
    graph = ops.get_default_graph()
    with self.test_session(graph=graph, use_gpu=use_gpu) as sess:
      image = array_ops.placeholder(dtype='float32', shape=shape)
      loss, train_op, saver = self._simple_model(image, is_fused, freeze_mode)
      if restore:
        saver.restore(sess, checkpoint_path)
      else:
        sess.run(variables.global_variables_initializer())
      np.random.seed(0)
      for _ in range(2):
        image_val = np.random.rand(*shape).astype(np.float32)
        sess.run([loss, train_op], feed_dict={image: image_val})
      if restore:
        all_vars = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
        all_vars_values = [var.eval() for var in all_vars]
        return all_vars_values
      else:
        saver.save(sess, checkpoint_path)

  def _infer(self, checkpoint_path, image_val, shape, use_gpu, is_fused):
    ops.reset_default_graph()
    graph = ops.get_default_graph()
    with self.test_session(graph=graph, use_gpu=use_gpu) as sess:
      image = array_ops.placeholder(dtype='float32', shape=shape)
      loss, _, saver = self._simple_model(image, is_fused, True)
      saver.restore(sess, checkpoint_path)
      loss_val = sess.run(loss, feed_dict={image: image_val})
      return loss_val

  def _testCheckpoint(self, is_fused_checkpoint_a, is_fused_checkpoint_b,
                      use_gpu_checkpoint_a, use_gpu_checkpoint_b,
                      use_gpu_test_a, use_gpu_test_b, freeze_mode):
    batch, height, width, input_channels = 2, 4, 5, 3
    shape = [batch, height, width, input_channels]
    base_path = '%s_%s_%s_%s_%s_%s' % (is_fused_checkpoint_a,
                                       is_fused_checkpoint_b,
                                       use_gpu_checkpoint_a,
                                       use_gpu_checkpoint_b, use_gpu_test_a,
                                       use_gpu_test_b)

    checkpoint_path_a = os.path.join(self.get_temp_dir(),
                                     'checkpoint_a_%s' % base_path)
    self._train(
        checkpoint_path_a,
        shape,
        use_gpu_checkpoint_a,
        is_fused_checkpoint_a,
        restore=False,
        freeze_mode=freeze_mode)
    checkpoint_path_b = os.path.join(self.get_temp_dir(),
                                     'checkpoint_b_%s' % base_path)
    self._train(
        checkpoint_path_b,
        shape,
        use_gpu_checkpoint_b,
        is_fused_checkpoint_b,
        restore=False,
        freeze_mode=freeze_mode)

    vars_fused = self._train(
        checkpoint_path_a,
        shape,
        use_gpu_test_a,
        True,
        restore=True,
        freeze_mode=freeze_mode)
    vars_nonfused = self._train(
        checkpoint_path_b,
        shape,
        use_gpu_test_b,
        False,
        restore=True,
        freeze_mode=freeze_mode)
    self.assertEqual(len(vars_fused), 5)
    self.assertEqual(len(vars_nonfused), 5)
    for var_fused, var_nonfused in zip(vars_fused, vars_nonfused):
      self.assertAllClose(var_fused, var_nonfused, atol=1e-6)

    image_val = np.random.rand(batch, height, width,
                               input_channels).astype(np.float32)
    loss_fused_val = self._infer(checkpoint_path_a, image_val, shape,
                                 use_gpu_test_a, True)
    loss_nonfused_val = self._infer(checkpoint_path_b, image_val, shape,
                                    use_gpu_test_b, False)
    self.assertAllClose(loss_fused_val, loss_nonfused_val, atol=1e-6)

  def _testCheckpointCrossDevice(self, ckpt_a_fused, ckpt_a_use_gpu,
                                 ckpt_b_fused, ckpt_b_use_gpu):
    for use_gpu_test_a in [True, False]:
      for use_gpu_test_b in [True, False]:
        for freeze_mode in [True, False]:
          self._testCheckpoint(ckpt_a_fused, ckpt_a_use_gpu, ckpt_b_fused,
                               ckpt_b_use_gpu, use_gpu_test_a, use_gpu_test_b,
                               freeze_mode)

  def testCheckpointFusedCPUAndFusedGPU(self):
    self._testCheckpointCrossDevice(True, False, True, True)

  def testCheckpointFusedCPUAndFusedCPU(self):
    self._testCheckpointCrossDevice(True, False, True, False)

  def testCheckpointFusedGPUAndFusedGPU(self):
    self._testCheckpointCrossDevice(True, True, True, True)

  def testCheckpointNonFusedCPUAndNonFusedGPU(self):
    self._testCheckpointCrossDevice(False, False, False, True)

  def testCheckpointNonFusedCPUAndNonFusedCPU(self):
    self._testCheckpointCrossDevice(False, False, False, False)

  def testCheckpointNonFusedGPUAndNonFusedGPU(self):
    self._testCheckpointCrossDevice(False, True, False, True)

  def testCheckpointNonFusedGPUAndFusedGPU(self):
    self._testCheckpointCrossDevice(False, True, True, True)

  def testCheckpointNonFusedGPUAndFusedCPU(self):
    self._testCheckpointCrossDevice(False, True, True, False)

  def testCheckpointNonFusedCPUAndFusedCPU(self):
    self._testCheckpointCrossDevice(False, False, True, False)

  def testCreateBN(self):
    # Call layer.
    bn = normalization_layers.BatchNormalization(axis=1)
    inputs = random_ops.random_uniform((5, 4, 3), seed=1)
    training = array_ops.placeholder(dtype='bool')
    outputs = bn.apply(inputs, training=training)

    # Verify shape.
    self.assertListEqual(outputs.get_shape().as_list(), [5, 4, 3])

    # Verify layer attributes.
    self.assertEqual(len(bn.updates), 2)
    self.assertEqual(len(bn.variables), 4)
    self.assertEqual(len(bn.trainable_variables), 2)
    self.assertEqual(len(bn.non_trainable_variables), 2)

    # Test that updates were created and added to UPDATE_OPS.
    self.assertEqual(len(bn.updates), 2)
    self.assertListEqual(
        ops.get_collection(ops.GraphKeys.UPDATE_OPS), bn.updates)

    # Test that weights were created and added to TRAINABLE_VARIABLES.
    self.assertListEqual(
        ops.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES),
        bn.trainable_variables)

  def test3DInputAxis1(self):
    epsilon = 1e-3
    bn = normalization_layers.BatchNormalization(
        axis=1, epsilon=epsilon, momentum=0.9)
    inputs = variables.Variable(
        np.random.random((5, 4, 3)) + 100, dtype=dtypes.float32)
    training = array_ops.placeholder(dtype='bool')
    outputs = bn.apply(inputs, training=training)

    with self.test_session() as sess:
      # Test training with placeholder learning phase.
      sess.run(variables.global_variables_initializer())

      np_gamma, np_beta = sess.run([bn.gamma, bn.beta])
      np_gamma = np.reshape(np_gamma, (1, 4, 1))
      np_beta = np.reshape(np_beta, (1, 4, 1))

      for _ in range(100):
        np_output, _, _ = sess.run([outputs] + bn.updates,
                                   feed_dict={training: True})
        # Verify that the axis is normalized during training.
        normed_np_output = ((np_output - epsilon) * np_gamma) + np_beta
        self.assertAlmostEqual(np.mean(normed_np_output), 0., places=1)
        self.assertAlmostEqual(np.std(normed_np_output), 1., places=1)

      # Verify that the statistics are updated during training.
      moving_mean, moving_var = sess.run([bn.moving_mean, bn.moving_variance])
      np_inputs = sess.run(inputs)
      mean = np.mean(np_inputs, axis=(0, 2))
      std = np.std(np_inputs, axis=(0, 2))
      variance = np.square(std)
      self.assertAllClose(mean, moving_mean, atol=1e-2)
      self.assertAllClose(variance, moving_var, atol=1e-2)

      # Test inference with placeholder learning phase.
      np_output = sess.run(outputs, feed_dict={training: False})

      # Verify that the axis is normalized during inference.
      normed_np_output = ((np_output - epsilon) * np_gamma) + np_beta
      self.assertAlmostEqual(np.mean(normed_np_output), 0., places=1)
      self.assertAlmostEqual(np.std(normed_np_output), 1., places=1)

  def test3DInputAxis2(self):
    epsilon = 1e-3
    bn = normalization_layers.BatchNormalization(
        axis=2, epsilon=epsilon, momentum=0.9)
    inputs = variables.Variable(
        np.random.random((5, 4, 3)) + 100, dtype=dtypes.float32)
    training = array_ops.placeholder(dtype='bool')
    outputs = bn.apply(inputs, training=training)

    with self.test_session() as sess:
      # Test training with placeholder learning phase.
      sess.run(variables.global_variables_initializer())
      np_gamma, np_beta = sess.run([bn.gamma, bn.beta])
      np_gamma = np.reshape(np_gamma, (1, 1, 3))
      np_beta = np.reshape(np_beta, (1, 1, 3))
      for _ in range(100):
        np_output, _, _ = sess.run([outputs] + bn.updates,
                                   feed_dict={training: True})
        # Verify that the axis is normalized during training.
        normed_np_output = ((np_output - epsilon) * np_gamma) + np_beta
        self.assertAlmostEqual(np.mean(normed_np_output), 0., places=1)
        self.assertAlmostEqual(np.std(normed_np_output), 1., places=1)

      # Verify that the statistics are updated during training.
      moving_mean, moving_var = sess.run([bn.moving_mean, bn.moving_variance])
      np_inputs = sess.run(inputs)
      mean = np.mean(np_inputs, axis=(0, 1))
      std = np.std(np_inputs, axis=(0, 1))
      variance = np.square(std)
      self.assertAllClose(mean, moving_mean, atol=1e-2)
      self.assertAllClose(variance, moving_var, atol=1e-2)

      # Test inference with placeholder learning phase.
      np_output = sess.run(outputs, feed_dict={training: False})

      # Verify that the axis is normalized during inference.
      normed_np_output = ((np_output - epsilon) * np_gamma) + np_beta
      self.assertAlmostEqual(np.mean(normed_np_output), 0., places=1)
      self.assertAlmostEqual(np.std(normed_np_output), 1., places=1)

  def test4DInputAxis1(self):
    if test.is_gpu_available(cuda_only=True):
      epsilon = 1e-3
      bn = normalization_layers.BatchNormalization(
          axis=1, epsilon=epsilon, momentum=0.9)
      inputs = variables.Variable(
          np.random.random((5, 4, 3, 6)) + 100, dtype=dtypes.float32)
      training = array_ops.placeholder(dtype='bool')
      outputs = bn.apply(inputs, training=training)

      with self.test_session(use_gpu=True) as sess:
        # Test training with placeholder learning phase.
        sess.run(variables.global_variables_initializer())
        np_gamma, np_beta = sess.run([bn.gamma, bn.beta])
        np_gamma = np.reshape(np_gamma, (1, 4, 1, 1))
        np_beta = np.reshape(np_beta, (1, 4, 1, 1))
        for _ in range(100):
          np_output, _, _ = sess.run(
              [outputs] + bn.updates, feed_dict={training: True})
          # Verify that the axis is normalized during training.
          normed_np_output = ((np_output - epsilon) * np_gamma) + np_beta
          self.assertAlmostEqual(np.mean(normed_np_output), 0., places=1)
          self.assertAlmostEqual(np.std(normed_np_output), 1., places=1)

        # Verify that the statistics are updated during training.
        moving_mean, moving_var = sess.run([bn.moving_mean, bn.moving_variance])
        np_inputs = sess.run(inputs)
        mean = np.mean(np_inputs, axis=(0, 2, 3))
        std = np.std(np_inputs, axis=(0, 2, 3))
        variance = np.square(std)
        self.assertAllClose(mean, moving_mean, atol=1e-2)
        self.assertAllClose(variance, moving_var, atol=1e-2)

        # Test inference with placeholder learning phase.
        np_output = sess.run(outputs, feed_dict={training: False})

        # Verify that the axis is normalized during inference.
        normed_np_output = ((np_output - epsilon) * np_gamma) + np_beta
        self.assertAlmostEqual(np.mean(normed_np_output), 0., places=1)
        self.assertAlmostEqual(np.std(normed_np_output), 1., places=1)

  def test4DInputAxis2(self):
    epsilon = 1e-3
    bn = normalization_layers.BatchNormalization(
        axis=2, epsilon=epsilon, momentum=0.9)
    inputs = variables.Variable(
        np.random.random((5, 4, 3, 6)) + 100, dtype=dtypes.float32)
    training = array_ops.placeholder(dtype='bool')
    outputs = bn.apply(inputs, training=training)

    with self.test_session() as sess:
      # Test training with placeholder learning phase.
      sess.run(variables.global_variables_initializer())
      np_gamma, np_beta = sess.run([bn.gamma, bn.beta])
      np_gamma = np.reshape(np_gamma, (1, 1, 3, 1))
      np_beta = np.reshape(np_beta, (1, 1, 3, 1))
      for _ in range(100):
        np_output, _, _ = sess.run([outputs] + bn.updates,
                                   feed_dict={training: True})
        # Verify that the axis is normalized during training.
        normed_np_output = ((np_output - epsilon) * np_gamma) + np_beta
        self.assertAlmostEqual(np.mean(normed_np_output), 0., places=1)
        self.assertAlmostEqual(np.std(normed_np_output), 1., places=1)

      # Verify that the statistics are updated during training.
      moving_mean, moving_var = sess.run([bn.moving_mean, bn.moving_variance])
      np_inputs = sess.run(inputs)
      mean = np.mean(np_inputs, axis=(0, 1, 3))
      std = np.std(np_inputs, axis=(0, 1, 3))
      variance = np.square(std)
      self.assertAllClose(mean, moving_mean, atol=1e-2)
      self.assertAllClose(variance, moving_var, atol=1e-2)

      # Test inference with placeholder learning phase.
      np_output = sess.run(outputs, feed_dict={training: False})

      # Verify that the axis is normalized during inference.
      normed_np_output = ((np_output - epsilon) * np_gamma) + np_beta
      self.assertAlmostEqual(np.mean(normed_np_output), 0., places=1)
      self.assertAlmostEqual(np.std(normed_np_output), 1., places=1)

  def test4DInputAxis3(self):
    epsilon = 1e-3
    bn = normalization_layers.BatchNormalization(
        axis=3, epsilon=epsilon, momentum=0.9)
    inputs = variables.Variable(
        np.random.random((5, 4, 3, 6)) + 100, dtype=dtypes.float32)
    training = array_ops.placeholder(dtype='bool')
    outputs = bn.apply(inputs, training=training)

    with self.test_session() as sess:
      # Test training with placeholder learning phase.
      sess.run(variables.global_variables_initializer())
      np_gamma, np_beta = sess.run([bn.gamma, bn.beta])
      np_gamma = np.reshape(np_gamma, (1, 1, 1, 6))
      np_beta = np.reshape(np_beta, (1, 1, 1, 6))
      for _ in range(100):
        np_output, _, _ = sess.run([outputs] + bn.updates,
                                   feed_dict={training: True})
        # Verify that the axis is normalized during training.
        normed_np_output = ((np_output - epsilon) * np_gamma) + np_beta
        self.assertAlmostEqual(np.mean(normed_np_output), 0., places=1)
        self.assertAlmostEqual(np.std(normed_np_output), 1., places=1)

      # Verify that the statistics are updated during training.
      moving_mean, moving_var = sess.run([bn.moving_mean, bn.moving_variance])
      np_inputs = sess.run(inputs)
      mean = np.mean(np_inputs, axis=(0, 1, 2))
      std = np.std(np_inputs, axis=(0, 1, 2))
      variance = np.square(std)
      self.assertAllClose(mean, moving_mean, atol=1e-2)
      self.assertAllClose(variance, moving_var, atol=1e-2)

      # Test inference with placeholder learning phase.
      np_output = sess.run(outputs, feed_dict={training: False})

      # Verify that the axis is normalized during inference.
      normed_np_output = ((np_output - epsilon) * np_gamma) + np_beta
      self.assertAlmostEqual(np.mean(normed_np_output), 0., places=1)
      self.assertAlmostEqual(np.std(normed_np_output), 1., places=1)

  def test4DInputAxis3Fused(self):
    epsilon = 1e-3
    bn = normalization_layers.BatchNormalization(
        axis=3, epsilon=epsilon, momentum=0.9, fused=True)
    inputs = variables.Variable(
        np.random.random((5, 4, 3, 6)) + 100, dtype=dtypes.float32)
    training = array_ops.placeholder(dtype='bool')
    outputs = bn.apply(inputs, training=training)

    with self.test_session() as sess:
      # Test training with placeholder learning phase.
      sess.run(variables.global_variables_initializer())
      np_gamma, np_beta = sess.run([bn.gamma, bn.beta])
      np_gamma = np.reshape(np_gamma, (1, 1, 1, 6))
      np_beta = np.reshape(np_beta, (1, 1, 1, 6))
      for _ in range(100):
        np_output, _, _ = sess.run(
            [outputs] + bn.updates, feed_dict={training: True})
        # Verify that the axis is normalized during training.
        normed_np_output = ((np_output - epsilon) * np_gamma) + np_beta
        self.assertAlmostEqual(np.mean(normed_np_output), 0., places=1)
        self.assertAlmostEqual(np.std(normed_np_output), 1., places=1)

      # Verify that the statistics are updated during training.
      moving_mean, moving_var = sess.run([bn.moving_mean, bn.moving_variance])
      np_inputs = sess.run(inputs)
      mean = np.mean(np_inputs, axis=(0, 1, 2))
      std = np.std(np_inputs, axis=(0, 1, 2))
      variance = np.square(std)
      self.assertAllClose(mean, moving_mean, atol=1e-2)
      self.assertAllClose(variance, moving_var, atol=1e-2)

      # Test inference with placeholder learning phase.
      np_output = sess.run(outputs, feed_dict={training: False})

      # Verify that the axis is normalized during inference.
      normed_np_output = ((np_output - epsilon) * np_gamma) + np_beta
      self.assertAlmostEqual(np.mean(normed_np_output), 0., places=1)
      self.assertAlmostEqual(np.std(normed_np_output), 1., places=1)

  def test4DInputAxis1Fused(self):
    if test.is_gpu_available(cuda_only=True):
      epsilon = 1e-3
      bn = normalization_layers.BatchNormalization(
          axis=1, epsilon=epsilon, momentum=0.9, fused=True)
      inputs = variables.Variable(
          np.random.random((5, 4, 3, 6)) + 100, dtype=dtypes.float32)
      training = array_ops.placeholder(dtype='bool')
      outputs = bn.apply(inputs, training=training)

      with self.test_session() as sess:
        # Test training with placeholder learning phase.
        sess.run(variables.global_variables_initializer())
        np_gamma, np_beta = sess.run([bn.gamma, bn.beta])
        np_gamma = np.reshape(np_gamma, (1, 4, 1, 1))
        np_beta = np.reshape(np_beta, (1, 4, 1, 1))
        for _ in range(100):
          np_output, _, _ = sess.run(
              [outputs] + bn.updates, feed_dict={training: True})
          # Verify that the axis is normalized during training.
          normed_np_output = ((np_output - epsilon) * np_gamma) + np_beta
          self.assertAlmostEqual(np.mean(normed_np_output), 0., places=1)
          self.assertAlmostEqual(np.std(normed_np_output), 1., places=1)

        # Verify that the statistics are updated during training.
        moving_mean, moving_var = sess.run([bn.moving_mean, bn.moving_variance])
        np_inputs = sess.run(inputs)
        mean = np.mean(np_inputs, axis=(0, 2, 3))
        std = np.std(np_inputs, axis=(0, 2, 3))
        variance = np.square(std)
        self.assertAllClose(mean, moving_mean, atol=1e-2)
        self.assertAllClose(variance, moving_var, atol=1e-2)

        # Test inference with placeholder learning phase.
        np_output = sess.run(outputs, feed_dict={training: False})

        # Verify that the axis is normalized during inference.
        normed_np_output = ((np_output - epsilon) * np_gamma) + np_beta
        self.assertAlmostEqual(np.mean(normed_np_output), 0., places=1)
        self.assertAlmostEqual(np.std(normed_np_output), 1., places=1)

  def testNegativeAxis(self):
    epsilon = 1e-3
    bn = normalization_layers.BatchNormalization(
        axis=-1, epsilon=epsilon, momentum=0.9)
    inputs = variables.Variable(
        np.random.random((5, 4, 3, 6)) + 100, dtype=dtypes.float32)
    training = array_ops.placeholder(dtype='bool')
    outputs = bn.apply(inputs, training=training)

    with self.test_session() as sess:
      # Test training with placeholder learning phase.
      sess.run(variables.global_variables_initializer())
      np_gamma, np_beta = sess.run([bn.gamma, bn.beta])
      np_gamma = np.reshape(np_gamma, (1, 1, 1, 6))
      np_beta = np.reshape(np_beta, (1, 1, 1, 6))
      for _ in range(100):
        np_output, _, _ = sess.run([outputs] + bn.updates,
                                   feed_dict={training: True})

        # Verify that the axis is normalized during training.
        normed_np_output = ((np_output - epsilon) * np_gamma) + np_beta
        self.assertAlmostEqual(np.mean(normed_np_output), 0., places=1)
        self.assertAlmostEqual(np.std(normed_np_output), 1., places=1)

      # Verify that the statistics are updated during training.
      moving_mean, moving_var = sess.run([bn.moving_mean, bn.moving_variance])
      np_inputs = sess.run(inputs)
      mean = np.mean(np_inputs, axis=(0, 1, 2))
      std = np.std(np_inputs, axis=(0, 1, 2))
      variance = np.square(std)
      self.assertAllClose(mean, moving_mean, atol=1e-2)
      self.assertAllClose(variance, moving_var, atol=1e-2)

      # Test inference with placeholder learning phase.
      np_output = sess.run(outputs, feed_dict={training: False})

      # Verify that the axis is normalized during inference.
      normed_np_output = ((np_output - epsilon) * np_gamma) + np_beta
      self.assertAlmostEqual(np.mean(normed_np_output), 0., places=1)
      self.assertAlmostEqual(np.std(normed_np_output), 1., places=1)

  def testBooleanLearningPhase(self):
    epsilon = 1e-3
    bn = normalization_layers.BatchNormalization(
        axis=-1, epsilon=epsilon, momentum=0.9)
    inputs = variables.Variable(
        np.random.random((5, 4, 3, 6)) + 100, dtype=dtypes.float32)
    outputs_training = bn.apply(inputs, training=True)
    outputs_infer = bn.apply(inputs, training=False)

    with self.test_session() as sess:
      # Test training with placeholder learning phase.
      sess.run(variables.global_variables_initializer())
      np_gamma, np_beta = sess.run([bn.gamma, bn.beta])
      np_gamma = np.reshape(np_gamma, (1, 1, 1, 6))
      np_beta = np.reshape(np_beta, (1, 1, 1, 6))
      for _ in range(100):
        np_output, _, _ = sess.run([outputs_training] + bn.updates)
        # Verify that the axis is normalized during training.
        normed_np_output = ((np_output - epsilon) * np_gamma) + np_beta
        self.assertAlmostEqual(np.mean(normed_np_output), 0., places=2)
        self.assertAlmostEqual(np.std(normed_np_output), 1., places=1)

      # Verify that the statistics are updated during training.
      moving_mean, moving_var = sess.run([bn.moving_mean, bn.moving_variance])
      np_inputs = sess.run(inputs)
      mean = np.mean(np_inputs, axis=(0, 1, 2))
      std = np.std(np_inputs, axis=(0, 1, 2))
      variance = np.square(std)
      self.assertAllClose(mean, moving_mean, atol=1e-2)
      self.assertAllClose(variance, moving_var, atol=1e-2)

      # Test inference with placeholder learning phase.
      np_output = sess.run(outputs_infer)

      # Verify that the axis is normalized during inference.
      normed_np_output = ((np_output - epsilon) * np_gamma) + np_beta
      self.assertAlmostEqual(np.mean(normed_np_output), 0., places=1)
      self.assertAlmostEqual(np.std(normed_np_output), 1., places=1)

  def testFunctionalNoReuse(self):
    inputs = variables.Variable(
        np.random.random((5, 4, 3, 6)), dtype=dtypes.float32)
    epsilon = 1e-3
    training = array_ops.placeholder(dtype='bool')
    outputs = normalization_layers.batch_norm(
        inputs,
        axis=-1,
        momentum=0.9,
        epsilon=epsilon,
        training=training,
        name='bn')

    updates = ops.get_collection(ops.GraphKeys.UPDATE_OPS)
    all_vars = dict([(v.name, v) for v in variables.global_variables()])
    moving_mean = all_vars['bn/moving_mean:0']
    moving_variance = all_vars['bn/moving_variance:0']
    beta = all_vars['bn/beta:0']
    gamma = all_vars['bn/gamma:0']

    with self.test_session() as sess:
      # Test training with placeholder learning phase.
      sess.run(variables.global_variables_initializer())
      np_gamma, np_beta = sess.run([gamma, beta])
      np_gamma = np.reshape(np_gamma, (1, 1, 1, 6))
      np_beta = np.reshape(np_beta, (1, 1, 1, 6))
      for _ in range(100):
        np_output, _, _ = sess.run([outputs] + updates,
                                   feed_dict={training: True})
        # Verify that the axis is normalized during training.
        normed_np_output = ((np_output - epsilon) * np_gamma) + np_beta
        self.assertAlmostEqual(np.mean(normed_np_output), 0., places=1)
        self.assertAlmostEqual(np.std(normed_np_output), 1., places=1)

      # Verify that the statistics are updated during training.
      np_moving_mean, np_moving_var = sess.run([moving_mean, moving_variance])
      np_inputs = sess.run(inputs)
      np_mean = np.mean(np_inputs, axis=(0, 1, 2))
      np_std = np.std(np_inputs, axis=(0, 1, 2))
      np_variance = np.square(np_std)
      self.assertAllClose(np_mean, np_moving_mean, atol=1e-2)
      self.assertAllClose(np_variance, np_moving_var, atol=1e-2)

      # Test inference with placeholder learning phase.
      np_output = sess.run(outputs, feed_dict={training: False})

      # Verify that the axis is normalized during inference.
      normed_np_output = ((np_output - epsilon) * np_gamma) + np_beta
      self.assertAlmostEqual(np.mean(normed_np_output), 0., places=1)
      self.assertAlmostEqual(np.std(normed_np_output), 1., places=1)

  def testFunctionalReuse(self):
    inputs1 = variables.Variable(
        np.random.random((5, 4, 3, 6)), dtype=dtypes.float32)
    inputs2 = variables.Variable(
        np.random.random((5, 4, 3, 6)), dtype=dtypes.float32)
    epsilon = 1e-3
    training = array_ops.placeholder(dtype='bool')
    _ = normalization_layers.batch_norm(
        inputs1,
        axis=-1,
        momentum=0.9,
        epsilon=epsilon,
        training=training,
        name='bn')
    outputs2 = normalization_layers.batch_norm(
        inputs2,
        axis=-1,
        momentum=0.9,
        epsilon=epsilon,
        training=training,
        name='bn',
        reuse=True)

    # Last 2 update ops
    updates = ops.get_collection(ops.GraphKeys.UPDATE_OPS)[-2:]
    all_vars = dict([(v.name, v) for v in variables.global_variables()])
    moving_mean = all_vars['bn/moving_mean:0']
    moving_variance = all_vars['bn/moving_variance:0']
    beta = all_vars['bn/beta:0']
    gamma = all_vars['bn/gamma:0']

    with self.test_session() as sess:
      # Test training with placeholder learning phase.
      sess.run(variables.global_variables_initializer())
      for _ in range(100):
        np_output, _, _ = sess.run([outputs2] + updates,
                                   feed_dict={training: True})

      # Verify that the statistics are updated during training.
      np_moving_mean, np_moving_var = sess.run([moving_mean, moving_variance])
      np_inputs = sess.run(inputs2)
      np_mean = np.mean(np_inputs, axis=(0, 1, 2))
      np_std = np.std(np_inputs, axis=(0, 1, 2))
      np_variance = np.square(np_std)
      self.assertAllClose(np_mean, np_moving_mean, atol=1e-2)
      self.assertAllClose(np_variance, np_moving_var, atol=1e-2)

      # Verify that the axis is normalized during training.
      np_gamma, np_beta = sess.run([gamma, beta])
      np_gamma = np.reshape(np_gamma, (1, 1, 1, 6))
      np_beta = np.reshape(np_beta, (1, 1, 1, 6))
      normed_np_output = ((np_output - epsilon) * np_gamma) + np_beta
      self.assertAlmostEqual(np.mean(normed_np_output), 0., places=2)
      self.assertAlmostEqual(np.std(normed_np_output), 1., places=1)

      # Test inference with placeholder learning phase.
      np_output = sess.run(outputs2, feed_dict={training: False})

      # Verify that the axis is normalized during inference.
      normed_np_output = ((np_output - epsilon) * np_gamma) + np_beta
      self.assertAlmostEqual(np.mean(normed_np_output), 0., places=2)
      self.assertAlmostEqual(np.std(normed_np_output), 1., places=1)

  def testFunctionalReuseFromScope(self):
    inputs = variables.Variable(
        np.random.random((5, 4, 3, 6)), dtype=dtypes.float32)
    epsilon = 1e-3
    training = array_ops.placeholder(dtype='bool')
    with variable_scope.variable_scope('scope'):
      _ = normalization_layers.batch_norm(
          inputs, axis=-1, momentum=0.9, epsilon=epsilon, training=training)
      self.assertEqual(len(variables.global_variables()), 5)
    with variable_scope.variable_scope('scope', reuse=True):
      _ = normalization_layers.batch_norm(
          inputs, axis=-1, momentum=0.9, epsilon=epsilon, training=training)
      self.assertEqual(len(variables.global_variables()), 5)

  def testNoCenter(self):
    bn = normalization_layers.BatchNormalization(axis=1, center=False)
    inputs = random_ops.random_uniform((5, 4, 3), seed=1)
    training = array_ops.placeholder(dtype='bool')
    outputs = bn.apply(inputs, training=training)

    # Verify shape.
    self.assertListEqual(outputs.get_shape().as_list(), [5, 4, 3])

    # Verify layer attributes.
    self.assertEqual(len(bn.updates), 2)
    self.assertEqual(len(bn.variables), 3)
    self.assertEqual(len(bn.trainable_variables), 1)
    self.assertEqual(len(bn.non_trainable_variables), 2)

  def testNoScale(self):
    bn = normalization_layers.BatchNormalization(axis=1, scale=False)
    inputs = random_ops.random_uniform((5, 4, 3), seed=1)
    training = array_ops.placeholder(dtype='bool')
    outputs = bn.apply(inputs, training=training)

    # Verify shape.
    self.assertListEqual(outputs.get_shape().as_list(), [5, 4, 3])

    # Verify layer attributes.
    self.assertEqual(len(bn.updates), 2)
    self.assertEqual(len(bn.variables), 3)
    self.assertEqual(len(bn.trainable_variables), 1)
    self.assertEqual(len(bn.non_trainable_variables), 2)

  def testRegularizers(self):
    reg = lambda x: 0.1 * math_ops.reduce_sum(x)
    bn = normalization_layers.BatchNormalization(axis=1, beta_regularizer=reg)
    inputs = random_ops.random_uniform((5, 4, 3), seed=1)
    training = array_ops.placeholder(dtype='bool')
    _ = bn.apply(inputs, training=training)
    self.assertEqual(len(bn.losses), 1)

    bn = normalization_layers.BatchNormalization(axis=1, gamma_regularizer=reg)
    inputs = random_ops.random_uniform((5, 4, 3), seed=1)
    training = array_ops.placeholder(dtype='bool')
    _ = bn.apply(inputs, training=training)
    self.assertEqual(len(bn.losses), 1)

  def testRenorm(self):
    shape = (4, 3)
    xt = array_ops.placeholder(dtypes.float32, shape)
    momentum = 0.99
    renorm_momentum = 0.8
    rmax = 1.1
    rmin = 0.9
    dmax = 0.1
    gamma = 2.
    beta = 3.
    epsilon = 0.001
    bn = normalization_layers.BatchNormalization(
        axis=1,
        gamma_initializer=init_ops.constant_initializer(gamma),
        beta_initializer=init_ops.constant_initializer(beta),
        epsilon=epsilon,
        momentum=momentum,
        renorm=True,
        renorm_clipping={'rmax': rmax, 'rmin': rmin, 'dmax': dmax},
        renorm_momentum=renorm_momentum)
    training = array_ops.placeholder(dtypes.bool)
    yt = bn.apply(xt, training=training)

    moving_mean = 0.
    moving_variance = 1.
    renorm_mean = renorm_stddev = 0.
    renorm_weight = 0.
    with self.test_session(use_gpu=True) as sess:
      sess.run(variables.global_variables_initializer())
      for _ in range(5):
        x = np.random.random(shape)

        mean = x.mean(0)
        stddev = np.sqrt(x.var(0) + epsilon)
        adj_mean = renorm_mean + (1. - renorm_weight) * mean
        adj_stddev = renorm_stddev + (1. - renorm_weight) * stddev
        r = (stddev / adj_stddev).clip(rmin, rmax)
        d = ((mean - adj_mean) / adj_stddev).clip(-dmax, dmax)
        y_train = ((x - mean) / stddev * r + d) * gamma + beta
        renorm_mean += (mean - renorm_mean) * (1. - renorm_momentum)
        renorm_stddev += (stddev - renorm_stddev) * (1. - renorm_momentum)
        renorm_weight += (1. - renorm_weight) * (1. - renorm_momentum)
        moving_mean += (renorm_mean / renorm_weight -
                        moving_mean) * (1. - momentum)
        moving_variance += ((renorm_stddev / renorm_weight) ** 2 - epsilon -
                            moving_variance) * (1. - momentum)

        y_test = ((x - moving_mean) / (moving_variance + epsilon) ** 0.5 *
                  gamma) + beta

        yt_val_train, _, _ = sess.run([yt] + bn.updates,
                                      feed_dict={xt: x, training: True})
        yt_val_test, _, _ = sess.run([yt] + bn.updates,
                                     feed_dict={xt: x, training: False})

        self.assertAllClose(y_train, yt_val_train, atol=1e-5)
        self.assertAllClose(y_test, yt_val_test, atol=1e-5)


if __name__ == '__main__':
  test.main()
