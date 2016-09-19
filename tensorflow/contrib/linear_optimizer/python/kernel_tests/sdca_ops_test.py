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
"""Tests for SdcaModel."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from threading import Thread
import tensorflow as tf

from tensorflow.contrib.linear_optimizer.python.ops.sdca_ops import _sdca_ops
from tensorflow.contrib.linear_optimizer.python.ops.sdca_ops import _ShardedMutableHashTable
from tensorflow.contrib.linear_optimizer.python.ops.sdca_ops import SdcaModel
from tensorflow.contrib.linear_optimizer.python.ops.sdca_ops import SparseFeatureColumn
from tensorflow.python.framework.test_util import TensorFlowTestCase
from tensorflow.python.platform import googletest

_MAX_ITERATIONS = 100
_SHARD_NUMBERS = [None, 1, 3, 10]
_NUM_PARTITIONS = [2, 4]

def make_example_proto(feature_dict, target, value=1.0):
  e = tf.train.Example()
  features = e.features

  features.feature['target'].float_list.value.append(target)

  for key, values in feature_dict.items():
    features.feature[key + '_indices'].int64_list.value.extend(values)
    features.feature[key + '_values'].float_list.value.extend([value] *
                                                              len(values))

  return e


def make_example_dict(example_protos, example_weights):

  def parse_examples(example_protos):
    features = {
        'target': tf.FixedLenFeature(shape=[1],
                                     dtype=tf.float32,
                                     default_value=0),
        'age_indices': tf.VarLenFeature(dtype=tf.int64),
        'age_values': tf.VarLenFeature(dtype=tf.float32),
        'gender_indices': tf.VarLenFeature(dtype=tf.int64),
        'gender_values': tf.VarLenFeature(dtype=tf.float32)
    }
    return tf.parse_example(
        [e.SerializeToString() for e in example_protos], features)

  parsed = parse_examples(example_protos)
  sparse_features = [
      SparseFeatureColumn(
          tf.reshape(
              tf.split(1, 2, parsed['age_indices'].indices)[0], [-1]),
          tf.reshape(parsed['age_indices'].values, [-1]),
          tf.reshape(parsed['age_values'].values, [-1])), SparseFeatureColumn(
              tf.reshape(
                  tf.split(1, 2, parsed['gender_indices'].indices)[0], [-1]),
              tf.reshape(parsed['gender_indices'].values, [-1]),
              tf.reshape(parsed['gender_values'].values, [-1]))
  ]
  return dict(sparse_features=sparse_features,
              dense_features=[],
              example_weights=example_weights,
              example_labels=tf.reshape(parsed['target'], [-1]),
              example_ids=['%d' % i for i in range(0, len(example_protos))])


def make_variable_dict(max_age, max_gender):
  # TODO(sibyl-toe9oF2e):  Figure out how to derive max_age & max_gender from
  # examples_dict.
  age_weights = tf.Variable(tf.zeros([max_age + 1], dtype=tf.float32))
  gender_weights = tf.Variable(tf.zeros([max_gender + 1], dtype=tf.float32))
  return dict(sparse_features_weights=[age_weights, gender_weights],
              dense_features_weights=[])


def make_dense_examples_and_variables_dicts(dense_features_values, weights,
                                            labels):
  """Creates examples and variables dictionaries for dense features.

  Variables shapes are inferred from the list of dense feature values passed as
  argument.

  Args:
    dense_features_values: The values of the dense features
    weights: The example weights.
    labels: The example labels.
  Returns:
    One dictionary for the examples and one for the variables.
  """
  dense_tensors = []
  dense_weights = []
  for dense_feature in dense_features_values:
    dense_tensor = tf.convert_to_tensor(dense_feature, dtype=tf.float32)
    check_shape_op = tf.Assert(
        tf.less_equal(tf.rank(dense_tensor), 2),
        ['dense_tensor shape must be [batch_size, dimension] or [batch_size]'])
    # Reshape to [batch_size, dense_column_dimension].
    with tf.control_dependencies([check_shape_op]):
      dense_tensor = tf.reshape(dense_tensor,
                                [dense_tensor.get_shape().as_list()[0], -1])
    dense_tensors.append(dense_tensor)
    # Add variables of shape [feature_column_dimension].
    dense_weights.append(
        tf.Variable(
            tf.zeros(
                [dense_tensor.get_shape().as_list()[1]], dtype=tf.float32)))

  examples_dict = dict(
      sparse_features=[],
      dense_features=dense_tensors,
      example_weights=weights,
      example_labels=labels,
      example_ids=['%d' % i for i in range(0, len(labels))])
  variables_dict = dict(
      sparse_features_weights=[], dense_features_weights=dense_weights)

  return examples_dict, variables_dict


def get_binary_predictions_for_logistic(predictions, cutoff=0.5):
  return tf.cast(
      tf.greater_equal(predictions, tf.ones_like(predictions) * cutoff),
      dtype=tf.int32)


def get_binary_predictions_for_hinge(predictions):
  return tf.cast(
      tf.greater_equal(predictions, tf.zeros_like(predictions)),
      dtype=tf.int32)


# TODO(sibyl-Mooth6ku): Add tests that exercise L1 and Shrinking.
# TODO(sibyl-vie3Poto): Refactor tests to avoid repetition of boilerplate code.
class SdcaModelTest(TensorFlowTestCase):
  """Base SDCA optimizer test class for any loss type."""

  def _single_threaded_test_session(self):
    config = tf.ConfigProto(inter_op_parallelism_threads=1,
                            intra_op_parallelism_threads=1)
    return self.test_session(use_gpu=False, config=config)


class SdcaWithLogisticLossTest(SdcaModelTest):
  """SDCA optimizer test class for logistic loss."""

  def testSimple(self):
    # Setup test data
    example_protos = [
        make_example_proto(
            {'age': [0],
             'gender': [0]}, 0),
        make_example_proto(
            {'age': [1],
             'gender': [1]}, 1),
    ]
    example_weights = [1.0, 1.0]
    for num_shards in _SHARD_NUMBERS:
      with self._single_threaded_test_session():
        examples = make_example_dict(example_protos, example_weights)
        variables = make_variable_dict(1, 1)
        options = dict(symmetric_l2_regularization=1,
                       symmetric_l1_regularization=0,
                       num_table_shards=num_shards,
                       loss_type='logistic_loss')

        lr = SdcaModel(examples, variables, options)
        tf.initialize_all_variables().run()
        unregularized_loss = lr.unregularized_loss(examples)
        loss = lr.regularized_loss(examples)
        predictions = lr.predictions(examples)
        self.assertAllClose(0.693147, unregularized_loss.eval())
        self.assertAllClose(0.693147, loss.eval())
        train_op = lr.minimize()
        for _ in range(_MAX_ITERATIONS):
          train_op.run()
        # The high tolerance in unregularized_loss comparisons is due to the
        # fact that it's possible to trade off unregularized_loss vs.
        # regularization and still have a sum that is quite close to the
        # optimal regularized_loss value.  SDCA's duality gap only ensures that
        # the regularized_loss is within 0.01 of optimal.
        # 0.525457 is the optimal regularized_loss.
        # 0.411608 is the unregularized_loss at that optimum.
        self.assertAllClose(0.411608, unregularized_loss.eval(), atol=0.05)
        self.assertAllClose(0.525457, loss.eval(), atol=0.01)
        predicted_labels = get_binary_predictions_for_logistic(predictions)
        self.assertAllEqual([0, 1], predicted_labels.eval())
        self.assertAllClose(0.01,
                            lr.approximate_duality_gap().eval(),
                            rtol=1e-2,
                            atol=1e-2)

  def testDistributedSimple(self):
    # Setup test data
    example_protos = [
        make_example_proto({'age': [0],
                            'gender': [0]}, 0),
        make_example_proto({'age': [1],
                            'gender': [1]}, 1),
    ]
    example_weights = [1.0, 1.0]
    for num_shards in _SHARD_NUMBERS:
      for num_partitions in _NUM_PARTITIONS:
        with self._single_threaded_test_session():
          examples = make_example_dict(example_protos, example_weights)
          variables = make_variable_dict(1, 1)
          options = dict(
              symmetric_l2_regularization=1,
              symmetric_l1_regularization=0,
              loss_type='logistic_loss',
              num_table_shards=num_shards,
              num_partitions=num_partitions)

          lr = SdcaModel(examples, variables, options)
          tf.initialize_all_variables().run()
          unregularized_loss = lr.unregularized_loss(examples)
          loss = lr.regularized_loss(examples)
          predictions = lr.predictions(examples)
          self.assertAllClose(0.693147, unregularized_loss.eval())
          self.assertAllClose(0.693147, loss.eval())

          train_op = lr.minimize()

          def Minimize():
            with self._single_threaded_test_session():
              for _ in range(_MAX_ITERATIONS):
                train_op.run()

          threads = []
          for _ in range(num_partitions):
            threads.append(Thread(target=Minimize))
            threads[-1].start()

          for t in threads:
            t.join()

          # The high tolerance in unregularized_loss comparisons is due to the
          # fact that it's possible to trade off unregularized_loss vs.
          # regularization and still have a sum that is quite close to the
          # optimal regularized_loss value.  SDCA's duality gap only ensures
          # that the regularized_loss is within 0.01 of optimal.
          # 0.525457 is the optimal regularized_loss.
          # 0.411608 is the unregularized_loss at that optimum.
          self.assertAllClose(0.411608, unregularized_loss.eval(), atol=0.05)
          self.assertAllClose(0.525457, loss.eval(), atol=0.01)
          predicted_labels = get_binary_predictions_for_logistic(predictions)
          self.assertAllEqual([0, 1], predicted_labels.eval())
          self.assertTrue(lr.approximate_duality_gap().eval() < 0.02)

  def testSimpleNoL2(self):
    # Same as test above (so comments from above apply) but without an L2.
    # The algorithm should behave as if we have an L2 of 1 in optimization but
    # 0 in regularized_loss.
    example_protos = [
        make_example_proto(
            {'age': [0],
             'gender': [0]}, 0),
        make_example_proto(
            {'age': [1],
             'gender': [1]}, 1),
    ]
    example_weights = [1.0, 1.0]
    for num_shards in _SHARD_NUMBERS:
      with self._single_threaded_test_session():
        examples = make_example_dict(example_protos, example_weights)
        variables = make_variable_dict(1, 1)
        options = dict(symmetric_l2_regularization=0,
                       symmetric_l1_regularization=0,
                       num_table_shards=num_shards,
                       loss_type='logistic_loss')

        lr = SdcaModel(examples, variables, options)
        tf.initialize_all_variables().run()
        unregularized_loss = lr.unregularized_loss(examples)
        loss = lr.regularized_loss(examples)
        predictions = lr.predictions(examples)
        self.assertAllClose(0.693147, unregularized_loss.eval())
        self.assertAllClose(0.693147, loss.eval())
        train_op = lr.minimize()
        for _ in range(_MAX_ITERATIONS):
          train_op.run()

        # There is neither L1 nor L2 loss, so regularized and unregularized
        # losses should be exactly the same.
        self.assertAllClose(0.40244, unregularized_loss.eval(), atol=0.01)
        self.assertAllClose(0.40244, loss.eval(), atol=0.01)
        predicted_labels = get_binary_predictions_for_logistic(predictions)
        self.assertAllEqual([0, 1], predicted_labels.eval())
        self.assertAllClose(0.01,
                            lr.approximate_duality_gap().eval(),
                            rtol=1e-2,
                            atol=1e-2)

  def testSomeUnweightedExamples(self):
    # Setup test data with 4 examples, but should produce the same
    # results as testSimple.
    example_protos = [
        # Will be used.
        make_example_proto(
            {'age': [0],
             'gender': [0]}, 0),
        # Will be ignored.
        make_example_proto(
            {'age': [1],
             'gender': [0]}, 0),
        # Will be used.
        make_example_proto(
            {'age': [1],
             'gender': [1]}, 1),
        # Will be ignored.
        make_example_proto(
            {'age': [1],
             'gender': [0]}, 1),
    ]
    example_weights = [1.0, 0.0, 1.0, 0.0]
    for num_shards in _SHARD_NUMBERS:
      with self._single_threaded_test_session():
        # Only use examples 0 and 2
        examples = make_example_dict(example_protos, example_weights)
        variables = make_variable_dict(1, 1)
        options = dict(symmetric_l2_regularization=1,
                       symmetric_l1_regularization=0,
                       num_table_shards=num_shards,
                       loss_type='logistic_loss')

        lr = SdcaModel(examples, variables, options)
        tf.initialize_all_variables().run()
        unregularized_loss = lr.unregularized_loss(examples)
        loss = lr.regularized_loss(examples)
        predictions = lr.predictions(examples)
        train_op = lr.minimize()
        for _ in range(_MAX_ITERATIONS):
          train_op.run()

        self.assertAllClose(0.411608, unregularized_loss.eval(), atol=0.05)
        self.assertAllClose(0.525457, loss.eval(), atol=0.01)
        predicted_labels = get_binary_predictions_for_logistic(predictions)
        self.assertAllClose([0, 1, 1, 1], predicted_labels.eval())
        self.assertAllClose(0.01,
                            lr.approximate_duality_gap().eval(),
                            rtol=1e-2,
                            atol=1e-2)

  def testFractionalExampleLabel(self):
    # Setup test data with 1 positive, and 1 mostly-negative example.
    example_protos = [
        make_example_proto(
            {'age': [0],
             'gender': [0]}, 0.1),
        make_example_proto(
            {'age': [1],
             'gender': [1]}, 1),
    ]
    example_weights = [1.0, 1.0]
    for num_shards in _SHARD_NUMBERS:
      with self._single_threaded_test_session():
        examples = make_example_dict(example_protos, example_weights)
        variables = make_variable_dict(1, 1)
        options = dict(symmetric_l2_regularization=1,
                       symmetric_l1_regularization=0,
                       num_table_shards=num_shards,
                       loss_type='logistic_loss')

        lr = SdcaModel(examples, variables, options)
        tf.initialize_all_variables().run()
        with self.assertRaisesOpError(
            'Only labels of 0.0 or 1.0 are supported right now.'):
          lr.minimize().run()

  def testImbalanced(self):
    # Setup test data with 1 positive, and 3 negative examples.
    example_protos = [
        make_example_proto(
            {'age': [0],
             'gender': [0]}, 0),
        make_example_proto(
            {'age': [2],
             'gender': [0]}, 0),
        make_example_proto(
            {'age': [3],
             'gender': [0]}, 0),
        make_example_proto(
            {'age': [1],
             'gender': [1]}, 1),
    ]
    example_weights = [1.0, 1.0, 1.0, 1.0]
    for num_shards in _SHARD_NUMBERS:
      with self._single_threaded_test_session():
        examples = make_example_dict(example_protos, example_weights)
        variables = make_variable_dict(3, 1)
        options = dict(symmetric_l2_regularization=1,
                       symmetric_l1_regularization=0,
                       num_table_shards=num_shards,
                       loss_type='logistic_loss')

        lr = SdcaModel(examples, variables, options)
        tf.initialize_all_variables().run()
        unregularized_loss = lr.unregularized_loss(examples)
        loss = lr.regularized_loss(examples)
        predictions = lr.predictions(examples)
        train_op = lr.minimize()
        for _ in range(_MAX_ITERATIONS):
          train_op.run()

        self.assertAllClose(0.226487 + 0.102902,
                            unregularized_loss.eval(),
                            atol=0.08)
        self.assertAllClose(0.328394 + 0.131364, loss.eval(), atol=0.01)
        predicted_labels = get_binary_predictions_for_logistic(predictions)
        self.assertAllEqual([0, 0, 0, 1], predicted_labels.eval())
        self.assertAllClose(0.0,
                            lr.approximate_duality_gap().eval(),
                            rtol=2e-2,
                            atol=1e-2)

  def testImbalancedWithExampleWeights(self):
    # Setup test data with 1 positive, and 1 negative example.
    example_protos = [
        make_example_proto(
            {'age': [0],
             'gender': [0]}, 0),
        make_example_proto(
            {'age': [1],
             'gender': [1]}, 1),
    ]
    example_weights = [3.0, 1.0]
    for num_shards in _SHARD_NUMBERS:
      with self._single_threaded_test_session():
        examples = make_example_dict(example_protos, example_weights)
        variables = make_variable_dict(1, 1)
        options = dict(symmetric_l2_regularization=1,
                       symmetric_l1_regularization=0,
                       num_table_shards=num_shards,
                       loss_type='logistic_loss')

        lr = SdcaModel(examples, variables, options)
        tf.initialize_all_variables().run()
        unregularized_loss = lr.unregularized_loss(examples)
        loss = lr.regularized_loss(examples)
        predictions = lr.predictions(examples)
        train_op = lr.minimize()
        for _ in range(_MAX_ITERATIONS):
          train_op.run()

        self.assertAllClose(0.284860, unregularized_loss.eval(), atol=0.08)
        self.assertAllClose(0.408044, loss.eval(), atol=0.012)
        predicted_labels = get_binary_predictions_for_logistic(predictions)
        self.assertAllEqual([0, 1], predicted_labels.eval())
        self.assertAllClose(0.0,
                            lr.approximate_duality_gap().eval(),
                            rtol=2e-2,
                            atol=1e-2)

  def testInstancesOfOneClassOnly(self):
    # Setup test data with 1 positive (ignored), and 1 negative example.
    example_protos = [
        make_example_proto(
            {'age': [0],
             'gender': [0]}, 0),
        make_example_proto(
            {'age': [1],
             'gender': [0]}, 1),  # Shares gender with the instance above.
    ]
    example_weights = [1.0, 0.0]  # Second example "omitted" from training.
    for num_shards in _SHARD_NUMBERS:
      with self._single_threaded_test_session():
        examples = make_example_dict(example_protos, example_weights)
        variables = make_variable_dict(1, 1)
        options = dict(symmetric_l2_regularization=1,
                       symmetric_l1_regularization=0,
                       num_table_shards=num_shards,
                       loss_type='logistic_loss')

        lr = SdcaModel(examples, variables, options)
        tf.initialize_all_variables().run()
        unregularized_loss = lr.unregularized_loss(examples)
        loss = lr.regularized_loss(examples)
        predictions = lr.predictions(examples)
        train_op = lr.minimize()
        for _ in range(_MAX_ITERATIONS):
          train_op.run()
        self.assertAllClose(0.411608, unregularized_loss.eval(), atol=0.05)
        self.assertAllClose(0.525457, loss.eval(), atol=0.01)
        predicted_labels = get_binary_predictions_for_logistic(predictions)
        self.assertAllEqual([0, 0], predicted_labels.eval())
        self.assertAllClose(0.01,
                            lr.approximate_duality_gap().eval(),
                            rtol=1e-2,
                            atol=1e-2)

  def testOutOfRangeSparseFeatures(self):
    # Setup test data
    example_protos = [
        make_example_proto({'age': [0],
                            'gender': [0]}, 0),
        make_example_proto({'age': [1],
                            'gender': [1]}, 1),
    ]
    example_weights = [1.0, 1.0]
    with self._single_threaded_test_session():
      examples = make_example_dict(example_protos, example_weights)
      variables = make_variable_dict(0, 0)
      options = dict(
          symmetric_l2_regularization=1,
          symmetric_l1_regularization=0,
          loss_type='logistic_loss')

      lr = SdcaModel(examples, variables, options)
      tf.initialize_all_variables().run()
      train_op = lr.minimize()
      with self.assertRaisesRegexp(tf.errors.InvalidArgumentError,
                                   'Found sparse feature indices out.*'):
        train_op.run()

  def testOutOfRangeDenseFeatures(self):
    with self._single_threaded_test_session():
      examples, variables = make_dense_examples_and_variables_dicts(
          dense_features_values=[[[1.0, 0.0], [0.0, 1.0]]],
          weights=[20.0, 10.0],
          labels=[1.0, 0.0])
      # Replace with a variable of size 1 instead of 2.
      variables['dense_features_weights'] = [
          tf.Variable(tf.zeros(
              [1], dtype=tf.float32))
      ]
      options = dict(
          symmetric_l2_regularization=1.0,
          symmetric_l1_regularization=0,
          loss_type='logistic_loss')
      lr = SdcaModel(examples, variables, options)
      tf.initialize_all_variables().run()
      train_op = lr.minimize()
      with self.assertRaisesRegexp(
          tf.errors.InvalidArgumentError,
          'More dense features than we have parameters for.*'):
        train_op.run()

  # TODO(katsiaspis): add a test for the case when examples at the end of an
  # epoch are repeated, since example id may be duplicated.


class SdcaWithLinearLossTest(SdcaModelTest):
  """SDCA optimizer test class for linear (squared) loss."""

  def testSimple(self):
    # Setup test data
    example_protos = [
        make_example_proto(
            {'age': [0],
             'gender': [0]}, -10.0),
        make_example_proto(
            {'age': [1],
             'gender': [1]}, 14.0),
    ]
    example_weights = [1.0, 1.0]
    with self._single_threaded_test_session():
      examples = make_example_dict(example_protos, example_weights)
      variables = make_variable_dict(1, 1)
      options = dict(symmetric_l2_regularization=1,
                     symmetric_l1_regularization=0,
                     loss_type='squared_loss')

      lr = SdcaModel(examples, variables, options)
      tf.initialize_all_variables().run()
      predictions = lr.predictions(examples)
      train_op = lr.minimize()
      for _ in range(_MAX_ITERATIONS):
        train_op.run()

      # Predictions should be 2/3 of label due to minimizing regularized loss:
      #   (label - 2 * weight)^2 / 2 + L2 * 2 * weight^2
      self.assertAllClose([-20.0 / 3.0, 28.0 / 3.0],
                          predictions.eval(),
                          rtol=0.005)
      # Approximate gap should be very close to 0.0. (In fact, because the gap
      # is only approximate, it is likely that upon convergence the duality gap
      # can have a tiny negative value).
      self.assertAllClose(0.0,
                          lr.approximate_duality_gap().eval(),
                          atol=1e-2)

  def testL2Regularization(self):
    # Setup test data
    example_protos = [
        # 2 identical examples
        make_example_proto(
            {'age': [0],
             'gender': [0]}, -10.0),
        make_example_proto(
            {'age': [0],
             'gender': [0]}, -10.0),
        # 2 more identical examples
        make_example_proto(
            {'age': [1],
             'gender': [1]}, 14.0),
        make_example_proto(
            {'age': [1],
             'gender': [1]}, 14.0),
    ]
    example_weights = [1.0, 1.0, 1.0, 1.0]
    with self._single_threaded_test_session():
      examples = make_example_dict(example_protos, example_weights)
      variables = make_variable_dict(1, 1)
      options = dict(symmetric_l2_regularization=16,
                     symmetric_l1_regularization=0,
                     loss_type='squared_loss')

      lr = SdcaModel(examples, variables, options)
      tf.initialize_all_variables().run()
      predictions = lr.predictions(examples)

      train_op = lr.minimize()
      for _ in range(_MAX_ITERATIONS):
        train_op.run()

      # Predictions should be 1/5 of label due to minimizing regularized loss:
      #   (label - 2 * weight)^2 + L2 * 16 * weight^2
      optimal1 = -10.0 / 5.0
      optimal2 = 14.0 / 5.0
      self.assertAllClose(
          [optimal1, optimal1, optimal2, optimal2],
          predictions.eval(),
          rtol=0.01)

  def testL1Regularization(self):
    # Setup test data
    example_protos = [
        make_example_proto(
            {'age': [0],
             'gender': [0]}, -10.0),
        make_example_proto(
            {'age': [1],
             'gender': [1]}, 14.0),
    ]
    example_weights = [1.0, 1.0]
    with self._single_threaded_test_session():
      examples = make_example_dict(example_protos, example_weights)
      variables = make_variable_dict(1, 1)
      options = dict(symmetric_l2_regularization=1.0,
                     symmetric_l1_regularization=4.0,
                     loss_type='squared_loss')
      lr = SdcaModel(examples, variables, options)
      tf.initialize_all_variables().run()
      prediction = lr.predictions(examples)
      loss = lr.regularized_loss(examples)

      train_op = lr.minimize()
      for _ in range(_MAX_ITERATIONS):
        train_op.run()

      # Predictions should be -4.0, 48/5 due to minimizing regularized loss:
      #   (label - 2 * weight)^2 / 2 + L2 * 2 * weight^2 + L1 * 4 * weight
      self.assertAllClose([-4.0, 20.0 / 3.0], prediction.eval(), rtol=0.08)

      # Loss should be the sum of the regularized loss value from above per
      # example after plugging in the optimal weights.
      self.assertAllClose(308.0 / 6.0, loss.eval(), atol=0.01)

  def testFeatureValues(self):
    # Setup test data
    example_protos = [
        make_example_proto(
            {'age': [0],
             'gender': [0]}, -10.0, -2.0),
        make_example_proto(
            {'age': [1],
             'gender': [1]}, 14.0, 2.0),
    ]
    example_weights = [5.0, 3.0]
    with self._single_threaded_test_session():
      examples = make_example_dict(example_protos, example_weights)

      variables = make_variable_dict(1, 1)
      options = dict(symmetric_l2_regularization=1,
                     symmetric_l1_regularization=0,
                     loss_type='squared_loss')

      lr = SdcaModel(examples, variables, options)
      tf.initialize_all_variables().run()
      predictions = lr.predictions(examples)

      train_op = lr.minimize()
      for _ in range(_MAX_ITERATIONS):
        train_op.run()

      # There are 4 (sparse) variable weights to be learned. 2 for age and 2 for
      # gender. Let w_1, w_2 be age weights, w_3, w_4 be gender weights, y_1,
      # y_2 be the labels for examples 1 and 2 respectively and s_1, s_2 the
      # corresponding *example* weights. With the given feature values, the loss
      # function is given by:
      # s_1/2(y_1 + 2w_1 + 2w_3)^2 + s_2/2(y_2 - 2w_2 - 2w_4)^2
      # + \lambda/2 (w_1^2 + w_2^2 + w_3^2 + w_4^2). Solving for the optimal, it
      # can be verified that:
      # w_1* = w_3* = -2.0 s_1 y_1/(\lambda + 8 s_1) and
      # w_2* = w_4* = 2 \cdot s_2 y_2/(\lambda + 8 s_2). Equivalently, due to
      # regularization and example weights, the predictions are within:
      # 8 \cdot s_i /(\lambda + 8 \cdot s_i) of the labels.
      self.assertAllClose([-10 * 40.0 / 41.0, 14.0 * 24 / 25.0],
                          predictions.eval(),
                          atol=0.01)

  def testDenseFeaturesWithDefaultWeights(self):
    with self._single_threaded_test_session():
      examples, variables = make_dense_examples_and_variables_dicts(
          dense_features_values=[[[1.0], [0.0]], [0.0, 1.0]],
          weights=[1.0, 1.0],
          labels=[10.0, -5.0])
      options = dict(symmetric_l2_regularization=1.0,
                     symmetric_l1_regularization=0,
                     loss_type='squared_loss')
      lr = SdcaModel(examples, variables, options)
      tf.initialize_all_variables().run()
      predictions = lr.predictions(examples)

      train_op = lr.minimize()
      for _ in range(_MAX_ITERATIONS):
        train_op.run()

      # The loss function for these particular features is given by:
      # 1/2(label_1-w_1)^2 + 1/2(label_2-w_2)^2 + \lambda/2 (w_1^2 + w_2^2). So,
      # differentiating wrt to w_1, w_2 yields the following optimal values:
      # w_1* = label_1/(\lambda + 1)= 10/2, w_2* =label_2/(\lambda + 1)= -5/2.
      # In this case the (unnormalized regularized) loss will be:
      # 1/2(10-5)^2 + 1/2(5-5/2)^2 + 1/2(5^2 + (5/2)^2) = 125.0/4. The actual
      # loss should be further normalized by the sum of example weights.
      self.assertAllClose([5.0, -2.5],
                          predictions.eval(),
                          rtol=0.01)
      loss = lr.regularized_loss(examples)
      self.assertAllClose(125.0 / 8.0, loss.eval(), atol=0.01)

  def testDenseFeaturesWithArbitraryWeights(self):
    with self._single_threaded_test_session():
      examples, variables = make_dense_examples_and_variables_dicts(
          dense_features_values=[[[1.0, 0.0], [0.0, 1.0]]],
          weights=[20.0, 10.0],
          labels=[10.0, -5.0])
      options = dict(symmetric_l2_regularization=5.0,
                     symmetric_l1_regularization=0,
                     loss_type='squared_loss')
      lr = SdcaModel(examples, variables, options)
      tf.initialize_all_variables().run()
      predictions = lr.predictions(examples)

      train_op = lr.minimize()
      for _ in range(_MAX_ITERATIONS):
        train_op.run()

      # The loss function for these particular features is given by:
      # 1/2 s_1 (label_1-w_1)^2 + 1/2 s_2(label_2-w_2)^2 +
      # \lambda/2 (w_1^2 + w_2^2) where s_1, s_2 are the *example weights. It
      # turns out that the optimal (variable) weights are given by:
      # w_1* = label_1 \cdot s_1/(\lambda + s_1)= 8.0 and
      # w_2* =label_2 \cdot s_2/(\lambda + s_2)= -10/3.
      # In this case the (unnormalized regularized) loss will be:
      # s_1/2(8-10)^2 + s_2/2(5-10/3)^2 + 5.0/2(8^2 + (10/3)^2) = 2175.0/9. The
      # actual loss should be further normalized by the sum of example weights.
      self.assertAllClose([8.0, -10.0/3],
                          predictions.eval(),
                          rtol=0.01)
      loss = lr.regularized_loss(examples)
      self.assertAllClose(2175.0 / 270.0, loss.eval(), atol=0.01)


class SdcaWithHingeLossTest(SdcaModelTest):
  """SDCA optimizer test class for hinge loss."""

  def testSimple(self):
    # Setup test data
    example_protos = [
        make_example_proto(
            {'age': [0],
             'gender': [0]}, 0),
        make_example_proto(
            {'age': [1],
             'gender': [1]}, 1),
    ]
    example_weights = [1.0, 1.0]
    with self._single_threaded_test_session():
      examples = make_example_dict(example_protos, example_weights)
      variables = make_variable_dict(1, 1)
      options = dict(symmetric_l2_regularization=1.0,
                     symmetric_l1_regularization=0,
                     loss_type='hinge_loss')
      model = SdcaModel(examples, variables, options)
      tf.initialize_all_variables().run()

      # Before minimization, the weights default to zero. There is no loss due
      # to regularization, only unregularized loss which is 0.5 * (1+1) = 1.0.
      predictions = model.predictions(examples)
      self.assertAllClose([0.0, 0.0], predictions.eval())
      unregularized_loss = model.unregularized_loss(examples)
      regularized_loss = model.regularized_loss(examples)
      self.assertAllClose(1.0, unregularized_loss.eval())
      self.assertAllClose(1.0, regularized_loss.eval())

      # After minimization, the model separates perfectly the data points. There
      # are 4 sparse weights: 2 for age (say w1, w2) and 2 for gender (say w3
      # and w4). Solving the system w1 + w3 = 1.0, w2 + w4 = -1.0 and minimizing
      # wrt to \|\vec{w}\|_2, gives w1=w3=1/2 and w2=w4=-1/2. This gives 0.0
      # unregularized loss and 0.25 L2 loss.
      train_op = model.minimize()
      for _ in range(_MAX_ITERATIONS):
        train_op.run()

      binary_predictions = get_binary_predictions_for_hinge(predictions)
      self.assertAllEqual([-1.0, 1.0], predictions.eval())
      self.assertAllEqual([0, 1], binary_predictions.eval())
      self.assertAllClose(0.0, unregularized_loss.eval())
      self.assertAllClose(0.25, regularized_loss.eval(), atol=0.05)

  def testDenseFeaturesPerfectlySeparable(self):
    with self._single_threaded_test_session():
      examples, variables = make_dense_examples_and_variables_dicts(
          dense_features_values=[[1.0, 1.0], [1.0, -1.0]],
          weights=[1.0, 1.0],
          labels=[1.0, 0.0])
      options = dict(
          symmetric_l2_regularization=1.0,
          symmetric_l1_regularization=0,
          loss_type='hinge_loss')
      model = SdcaModel(examples, variables, options)
      tf.initialize_all_variables().run()
      predictions = model.predictions(examples)
      binary_predictions = get_binary_predictions_for_hinge(predictions)

      train_op = model.minimize()
      for _ in range(_MAX_ITERATIONS):
        train_op.run()

      self.assertAllClose([1.0, -1.0], predictions.eval(), atol=0.05)
      self.assertAllEqual([1, 0], binary_predictions.eval())

      # (1.0, 1.0) and (1.0, -1.0) are perfectly separable by x-axis (that is,
      # the SVM's functional margin >=1), so the unregularized loss is ~0.0.
      # There is only loss due to l2-regularization. For these datapoints, it
      # turns out that w_1~=0.0 and w_2~=1.0 which means that l2 loss is ~0.25.
      unregularized_loss = model.unregularized_loss(examples)
      regularized_loss = model.regularized_loss(examples)
      self.assertAllClose(0.0, unregularized_loss.eval(), atol=0.02)
      self.assertAllClose(0.25, regularized_loss.eval(), atol=0.02)

  def testDenseFeaturesSeparableWithinMargins(self):
    with self._single_threaded_test_session():
      examples, variables = make_dense_examples_and_variables_dicts(
          dense_features_values=[[[1.0, 0.5], [1.0, -0.5]]],
          weights=[1.0, 1.0],
          labels=[1.0, 0.0])
      options = dict(symmetric_l2_regularization=1.0,
                     symmetric_l1_regularization=0,
                     loss_type='hinge_loss')
      model = SdcaModel(examples, variables, options)
      tf.initialize_all_variables().run()
      predictions = model.predictions(examples)
      binary_predictions = get_binary_predictions_for_hinge(predictions)

      train_op = model.minimize()
      for _ in range(_MAX_ITERATIONS):
        train_op.run()

      # (1.0, 0.5) and (1.0, -0.5) are separable by x-axis but the datapoints
      # are within the margins so there is unregularized loss (1/2 per example).
      # For these datapoints, optimal weights are w_1~=0.0 and w_2~=1.0 which
      # gives an L2 loss of ~0.25.
      self.assertAllClose([0.5, -0.5], predictions.eval(), rtol=0.05)
      self.assertAllEqual([1, 0], binary_predictions.eval())
      unregularized_loss = model.unregularized_loss(examples)
      regularized_loss = model.regularized_loss(examples)
      self.assertAllClose(0.5, unregularized_loss.eval(), atol=0.02)
      self.assertAllClose(0.75, regularized_loss.eval(), atol=0.02)

  def testDenseFeaturesWeightedExamples(self):
    with self._single_threaded_test_session():
      examples, variables = make_dense_examples_and_variables_dicts(
          dense_features_values=[[[1.0], [1.0]], [[0.5], [-0.5]]],
          weights=[3.0, 1.0],
          labels=[1.0, 0.0])
      options = dict(symmetric_l2_regularization=1.0,
                     symmetric_l1_regularization=0,
                     loss_type='hinge_loss')
      model = SdcaModel(examples, variables, options)
      tf.initialize_all_variables().run()
      predictions = model.predictions(examples)
      binary_predictions = get_binary_predictions_for_hinge(predictions)
      train_op = model.minimize()
      for _ in range(_MAX_ITERATIONS):
        train_op.run()

      # Point (1.0, 0.5) has higher weight than (1.0, -0.5) so the model will
      # try to increase the margin from (1.0, 0.5). Due to regularization,
      # (1.0, -0.5) will be within the margin. For these points and example
      # weights, the optimal weights are w_1~=0.4 and w_2~=1.2 which give an L2
      # loss of 0.5 * 0.25 * 0.25 * 1.6 = 0.2. The binary predictions will be
      # correct, but the boundary will be much closer to the 2nd point than the
      # first one.
      self.assertAllClose([1.0, -0.2], predictions.eval(), atol=0.05)
      self.assertAllEqual([1, 0], binary_predictions.eval())
      unregularized_loss = model.unregularized_loss(examples)
      regularized_loss = model.regularized_loss(examples)
      self.assertAllClose(0.2, unregularized_loss.eval(), atol=0.02)
      self.assertAllClose(0.4, regularized_loss.eval(), atol=0.02)


class SdcaWithSmoothHingeLossTest(SdcaModelTest):
  """SDCA optimizer test class for smooth hinge loss."""

  def testSimple(self):
    # Setup test data
    example_protos = [
        make_example_proto({'age': [0],
                            'gender': [0]}, 0),
        make_example_proto({'age': [1],
                            'gender': [1]}, 1),
    ]
    example_weights = [1.0, 1.0]
    with self._single_threaded_test_session():
      examples = make_example_dict(example_protos, example_weights)
      variables = make_variable_dict(1, 1)
      options = dict(
          symmetric_l2_regularization=1.0,
          symmetric_l1_regularization=0,
          loss_type='smooth_hinge_loss')
      model = SdcaModel(examples, variables, options)
      tf.initialize_all_variables().run()

      # Before minimization, the weights default to zero. There is no loss due
      # to regularization, only unregularized loss which is 0.5 * (1+1) = 1.0.
      predictions = model.predictions(examples)
      self.assertAllClose([0.0, 0.0], predictions.eval())
      unregularized_loss = model.unregularized_loss(examples)
      regularized_loss = model.regularized_loss(examples)
      self.assertAllClose(1.0, unregularized_loss.eval())
      self.assertAllClose(1.0, regularized_loss.eval())

      # After minimization, the model separates perfectly the data points. There
      # are 4 sparse weights: 2 for age (say w1, w2) and 2 for gender (say w3
      # and w4). The minimization leads to w1=w3=1/3 and w2=w4=-1/3. This gives
      # an unregularized hinge loss of 0.33 and a 0.11 L2 loss
      train_op = model.minimize()
      for _ in range(_MAX_ITERATIONS):
        train_op.run()

      binary_predictions = get_binary_predictions_for_hinge(predictions)
      self.assertAllClose([-0.67, 0.67], predictions.eval(), atol=0.05)
      self.assertAllEqual([0, 1], binary_predictions.eval())
      self.assertAllClose(0.33, unregularized_loss.eval(), atol=0.02)
      self.assertAllClose(0.44, regularized_loss.eval(), atol=0.02)


class SparseFeatureColumnTest(SdcaModelTest):
  """Tests for SparseFeatureColumn.
  """

  def testBasic(self):
    expected_example_indices = [1, 1, 1, 2]
    expected_feature_indices = [0, 1, 2, 0]
    sfc = SparseFeatureColumn(expected_example_indices,
                              expected_feature_indices, None)
    self.assertTrue(isinstance(sfc.example_indices, tf.Tensor))
    self.assertTrue(isinstance(sfc.feature_indices, tf.Tensor))
    self.assertEqual(sfc.feature_values, None)
    with self._single_threaded_test_session():
      self.assertAllEqual(expected_example_indices, sfc.example_indices.eval())
      self.assertAllEqual(expected_feature_indices, sfc.feature_indices.eval())
    expected_feature_values = [1.0, 2.0, 3.0, 4.0]
    sfc = SparseFeatureColumn([1, 1, 1, 2], [0, 1, 2, 0],
                              expected_feature_values)
    with self._single_threaded_test_session():
      self.assertAllEqual(expected_feature_values, sfc.feature_values.eval())

class SdcaFprintTest(SdcaModelTest):
  """Tests for the SdcaFprint op.

  This is one way of enforcing the platform-agnostic nature of SdcaFprint.
  Basically we are checking against exact values and this test could be running
  across different platforms. Note that it is fine for expected values to change
  in the future, if the implementation of SdcaFprint changes (ie this is *not* a
  frozen test).
  """

  def testFprint(self):
    with self._single_threaded_test_session():
      in_data = tf.constant(['abc', 'very looooooong string', 'def'])
      out_data = _sdca_ops.sdca_fprint(in_data)
      self.assertAllEqual(
          [b'\x04l\x12\xd2\xaf\xb2\x809E\x9e\x02\x13',
           b'\x9f\x0f\x91P\x9aG.Ql\xf2Y\xf9',
           b'"0\xe00"\x18_\x08\x12?\xa0\x17'], out_data.eval())


class ShardedMutableHashTableTest(SdcaModelTest):
  """Tests for the _ShardedMutableHashTable class."""

  def testShardedMutableHashTable(self):
    for num_shards in [1, 3, 10]:
      with self._single_threaded_test_session():
        default_val = -1
        keys = tf.constant(['brain', 'salad', 'surgery'])
        values = tf.constant([0, 1, 2], tf.int64)
        table = _ShardedMutableHashTable(tf.string,
                                         tf.int64,
                                         default_val,
                                         num_shards=num_shards)
        self.assertAllEqual(0, table.size().eval())

        table.insert(keys, values).run()
        self.assertAllEqual(3, table.size().eval())

        input_string = tf.constant(['brain', 'salad', 'tank'])
        output = table.lookup(input_string)
        self.assertAllEqual([3], output.get_shape())

        result = output.eval()
        self.assertAllEqual([0, 1, -1], result)

  def testExportSharded(self):
    with self._single_threaded_test_session():
      default_val = -1
      num_shards = 2
      keys = tf.constant(['a1', 'b1', 'c2'])
      values = tf.constant([0, 1, 2], tf.int64)
      table = _ShardedMutableHashTable(
          tf.string, tf.int64, default_val, num_shards=num_shards)
      self.assertAllEqual(0, table.size().eval())

      table.insert(keys, values).run()
      self.assertAllEqual(3, table.size().eval())

      keys_list, values_list = table.export_sharded()
      self.assertAllEqual(num_shards, len(keys_list))
      self.assertAllEqual(num_shards, len(values_list))

      self.assertAllEqual(set([b'b1', b'c2']), set(keys_list[0].eval()))
      self.assertAllEqual([b'a1'], keys_list[1].eval())
      self.assertAllEqual(set([1, 2]), set(values_list[0].eval()))
      self.assertAllEqual([0], values_list[1].eval())


if __name__ == '__main__':
  googletest.main()
