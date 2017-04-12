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

"""Adam for TensorFlow."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
from tensorflow.python.training import optimizer
from tensorflow.python.training import training_ops


class AdamOptimizer(optimizer.Optimizer):
  """Optimizer that implements the Adam algorithm.

  See [Kingma et. al., 2014](http://arxiv.org/abs/1412.6980)
  ([pdf](http://arxiv.org/pdf/1412.6980.pdf)).
  """

  def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8,
               use_locking=False, name="Adam"):
    """Construct a new Adam optimizer.

    Initialization:

    ```
    m_0 <- 0 (Initialize initial 1st moment vector)
    v_0 <- 0 (Initialize initial 2nd moment vector)
    t <- 0 (Initialize timestep)
    ```

    The update rule for `variable` with gradient `g` uses an optimization
    described at the end of section2 of the paper:

    ```
    t <- t + 1
    lr_t <- learning_rate * sqrt(1 - beta2^t) / (1 - beta1^t)

    m_t <- beta1 * m_{t-1} + (1 - beta1) * g
    v_t <- beta2 * v_{t-1} + (1 - beta2) * g * g
    variable <- variable - lr_t * m_t / (sqrt(v_t) + epsilon)
    ```

    The default value of 1e-8 for epsilon might not be a good default in
    general. For example, when training an Inception network on ImageNet a
    current good choice is 1.0 or 0.1.

    Note that in dense implement of this algorithm, m_t, v_t and variable will
    update even if g is zero, but in sparse implement, m_t, v_t and variable
    will not update in iterations g is zero.

    Args:
      learning_rate: A Tensor or a floating point value.  The learning rate.
      beta1: A float value or a constant float tensor.
        The exponential decay rate for the 1st moment estimates.
      beta2: A float value or a constant float tensor.
        The exponential decay rate for the 2nd moment estimates.
      epsilon: A small constant for numerical stability.
      use_locking: If True use locks for update operations.
      name: Optional name for the operations created when applying gradients.
        Defaults to "Adam".
    """
    super(AdamOptimizer, self).__init__(use_locking, name)
    self._lr = learning_rate
    self._beta1 = beta1
    self._beta2 = beta2
    self._epsilon = epsilon

    # Tensor versions of the constructor arguments, created in _prepare().
    self._lr_t = None
    self._beta1_t = None
    self._beta2_t = None
    self._epsilon_t = None

    # Variables to accumulate the powers of the beta parameters.
    # Created in _create_slots when we know the variables to optimize.
    self._beta1_power = None
    self._beta2_power = None

    # Created in SparseApply if needed.
    self._updated_lr = None

  def _get_beta_accumulators(self):
    return self._beta1_power, self._beta2_power

  def _create_slots(self, var_list):
    # Create the beta1 and beta2 accumulators on the same device as the first
    # variable.
    if (self._beta1_power is None or
        self._beta1_power.graph is not var_list[0].graph):
      with ops.colocate_with(var_list[0]):
        self._beta1_power = variables.Variable(self._beta1,
                                               name="beta1_power",
                                               trainable=False)
        self._beta2_power = variables.Variable(self._beta2,
                                               name="beta2_power",
                                               trainable=False)
    # Create slots for the first and second moments.
    for v in var_list:
      self._zeros_slot(v, "m", self._name)
      self._zeros_slot(v, "v", self._name)

  def _prepare(self):
    self._lr_t = ops.convert_to_tensor(self._lr, name="learning_rate")
    self._beta1_t = ops.convert_to_tensor(self._beta1, name="beta1")
    self._beta2_t = ops.convert_to_tensor(self._beta2, name="beta2")
    self._epsilon_t = ops.convert_to_tensor(self._epsilon, name="epsilon")

  def _apply_dense(self, grad, var):
    m = self.get_slot(var, "m")
    v = self.get_slot(var, "v")
    return training_ops.apply_adam(
        var, m, v,
        math_ops.cast(self._beta1_power, var.dtype.base_dtype),
        math_ops.cast(self._beta2_power, var.dtype.base_dtype),
        math_ops.cast(self._lr_t, var.dtype.base_dtype),
        math_ops.cast(self._beta1_t, var.dtype.base_dtype),
        math_ops.cast(self._beta2_t, var.dtype.base_dtype),
        math_ops.cast(self._epsilon_t, var.dtype.base_dtype),
        grad, use_locking=self._use_locking).op

  def _resource_apply_dense(self, grad, var):
    m = self.get_slot(var, "m")
    v = self.get_slot(var, "v")
    return training_ops.resource_apply_adam(
        var.handle, m.handle, v.handle,
        math_ops.cast(self._beta1_power, grad.dtype.base_dtype),
        math_ops.cast(self._beta2_power, grad.dtype.base_dtype),
        math_ops.cast(self._lr_t, grad.dtype.base_dtype),
        math_ops.cast(self._beta1_t, grad.dtype.base_dtype),
        math_ops.cast(self._beta2_t, grad.dtype.base_dtype),
        math_ops.cast(self._epsilon_t, grad.dtype.base_dtype),
        grad, use_locking=self._use_locking)

  def _apply_sparse(self, grad, var):
    beta1_power = math_ops.cast(self._beta1_power, var.dtype.base_dtype)
    beta2_power = math_ops.cast(self._beta2_power, var.dtype.base_dtype)
    lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
    beta1_t = math_ops.cast(self._beta1_t, var.dtype.base_dtype)
    beta2_t = math_ops.cast(self._beta2_t, var.dtype.base_dtype)
    epsilon_t = math_ops.cast(self._epsilon_t, var.dtype.base_dtype)
    lr = (lr_t * math_ops.sqrt(1 - beta2_power) / (1 - beta1_power))
    # m_t = beta1 * m + (1 - beta1) * g_t
    m = self.get_slot(var, "m")
    m_scaled_g_values = grad.values * (1 - beta1_t)
    m_t = state_ops.assign(m, m * beta1_t,
                           use_locking=self._use_locking)
    m_t = state_ops.scatter_add(m_t, grad.indices, m_scaled_g_values,
                                use_locking=self._use_locking)
    # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
    v = self.get_slot(var, "v")
    v_scaled_g_values = (grad.values * grad.values) * (1 - beta2_t)
    v_t = state_ops.assign(v, v * beta2_t, use_locking=self._use_locking)
    v_t = state_ops.scatter_add(v_t, grad.indices, v_scaled_g_values,
                                use_locking=self._use_locking)
    v_sqrt = math_ops.sqrt(v_t)
    var_update = state_ops.assign_sub(var,
                                      lr * m_t / (v_sqrt + epsilon_t),
                                      use_locking=self._use_locking)
    return control_flow_ops.group(*[var_update, m_t, v_t])

  def _finish(self, update_ops, name_scope):
    # Update the power accumulators.
    with ops.control_dependencies(update_ops):
      with ops.colocate_with(self._beta1_power):
        update_beta1 = self._beta1_power.assign(
            self._beta1_power * self._beta1_t,
            use_locking=self._use_locking)
        update_beta2 = self._beta2_power.assign(
            self._beta2_power * self._beta2_t,
            use_locking=self._use_locking)
    return control_flow_ops.group(*update_ops + [update_beta1, update_beta2],
                                  name=name_scope)




class NadamOptimizer(AdamOptimizer):

  """Nadam and Radam optimizer extensions both feather Nesterov momentum
  (or a Nesterov-like interpolation factor) into Adam calculations.

  Both algorithms tend to be more stable than Adam with negligiable added 
  calculation (even though Nadam looks extravagent).

  Nadam's added Nesterov momentum term typically provides a 1-2% shift in momentum
  during the initial optimization steps. This is when optimizers are most prone 
  to develop instablity though so this can be quite important. This momentum 
  correction term is both theoretically known to reduce regret bounds as well as
  improve performance and stability in practice.

  [Dozat 2015]  "Incorporating Nesterov momentum into Adam",
  Stanford University, CS 229 Machine Learning, Tech. Rep.,
  2015. http://cs229.stanford.edu/proj2015/054_report.pdf

  Adapted from https://github.com/tdozat/Optimization/blob/master/tensorflow/

  """
  def __init__(self, learning_rate=0.0002, beta1=0.825, beta2=0.99685, epsilon=1e-8,
               use_locking=False, name="Nadam"):

    super(NadamOptimizer, self).__init__(use_locking, name)
    self._lr = learning_rate
    self._beta1 = beta1
    self._beta2 = beta2
    self._epsilon = epsilon

    # Tensor versions of the constructor arguments, created in _prepare().
    self._lr_t = None
    self._beta1_t = None
    self._beta2_t = None
    self._epsilon_t = None

    # Variables to accumulate the powers of the beta parameters.
    # Created in _create_slots when we know the variables to optimize.
    self._beta1_power = None
    self._beta2_power = None

    # Created in SparseApply if needed.
    self._updated_lr = None


  def _apply_dense(self, grad, var):
    m = self.get_slot(var, "m")
    v = self.get_slot(var, "v")
    return training_ops.apply_nadam(
        var, m, v,
        math_ops.cast(self._beta1_power, var.dtype.base_dtype),
        math_ops.cast(self._beta2_power, var.dtype.base_dtype),
        math_ops.cast(self._lr_t, var.dtype.base_dtype),
        math_ops.cast(self._beta1_t, var.dtype.base_dtype),
        math_ops.cast(self._beta2_t, var.dtype.base_dtype),
        math_ops.cast(self._epsilon_t, var.dtype.base_dtype),
        grad, use_locking=self._use_locking).op

  def _resource_apply_dense(self, grad, var):
    m = self.get_slot(var, "m")
    v = self.get_slot(var, "v")
    return training_ops.resource_apply_nadam(
        var, m.handle, v.handle,
        math_ops.cast(self._beta1_power, grad.dtype.base_dtype),
        math_ops.cast(self._beta2_power, grad.dtype.base_dtype),
        math_ops.cast(self._lr_t, grad.dtype.base_dtype),
        math_ops.cast(self._beta1_t, grad.dtype.base_dtype),
        math_ops.cast(self._beta2_t, grad.dtype.base_dtype),
        math_ops.cast(self._epsilon_t, grad.dtype.base_dtype),
        grad, use_locking=self._use_locking)

  def _apply_sparse(self, grad, var):
    beta1_power = math_ops.cast(self._beta1_power, var.dtype.base_dtype)
    beta2_power = math_ops.cast(self._beta2_power, var.dtype.base_dtype)
    lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
    beta1_t = math_ops.cast(self._beta1_t, var.dtype.base_dtype)
    beta2_t = math_ops.cast(self._beta2_t, var.dtype.base_dtype)
    epsilon_t = math_ops.cast(self._epsilon_t, var.dtype.base_dtype)
    tp = math_ops.sqrt(1 - beta2_power) / (1 - beta1_power)
    tp1 = math_ops.sqrt(1 - beta2_power*beta2_t) / (1 - beta1_power*beta1_t)

    # m_t = beta1 * m + (1 - beta1) * g_t
    m = self.get_slot(var, "m")
    m_t = state_ops.scatter_update(m, grad.indices,
                                   beta1_t * array_ops.gather(m, grad.indices) +
                                   (tp1 / tp) * (1 - beta1_t) * grad.values,
                                   use_locking=self._use_locking)

    # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
    v = self.get_slot(var, "v")
    v_t = state_ops.scatter_update(v, grad.indices,
                                   beta2_t * array_ops.gather(v, grad.indices) +
                                   (1 - beta2_t) * math_ops.square(grad.values),
                                   use_locking=self._use_locking)

    # variable -= learning_rate * m_t / (epsilon_t + sqrt(v_t))
    m_t_slice = array_ops.gather(m_t, grad.indices)
    v_t_slice = array_ops.gather(v_t, grad.indices)
    denominator_slice = math_ops.sqrt(v_t_slice) + epsilon_t
    var_update = state_ops.scatter_sub(var, grad.indices,
                                       lr_t * tp * m_t_slice / denominator_slice,
                                       use_locking=self._use_locking)
    return control_flow_ops.group(var_update, m_t, v_t)




class RadamOptimizer(AdamOptimizer):

  """
  Radam is a proposed mathematical simplification of the more theoretically sound
  but complex Nadam algorithm. This proposal comes from additional, more informal
  work by Dozat (to extend his results on Nadam):

  Details of Radam extension - https://github.com/tdozat/Optimization

  [Dozat 2015]  "Incorporating Nesterov momentum into Adam",
  Stanford University, CS 229 Machine Learning, Tech. Rep.,
  2015. http://cs229.stanford.edu/proj2015/054_report.pdf

  Interestingly, Radam turns out to be isomorphic to an Adam optimizer initialized
  with lower beta1 and a slightly higher learning rate.

  However, Radam's gamma hyperparameter is a more straightforward way for those
  who have a good learning rate for an Adam optimizer to incorporate a Nesterov-like
  momentum adjustment, than attempting to tune beta1 (or beta1+lr) by hand.

  """

  def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8,
               gamma=0.05, use_locking=False, name="Radam"):

    super(RadamOptimizer, self).__init__(use_locking, name)
    self._beta1 = beta1 - gamma
    self._lr = learning_rate * (self._beta1 / beta1)
    self._beta2 = beta2
    self._epsilon = epsilon

    # Tensor versions of the constructor arguments, created in _prepare().
    self._lr_t = None
    self._beta1_t = None
    self._beta2_t = None
    self._epsilon_t = None

    # Variables to accumulate the powers of the beta parameters.
    # Created in _create_slots when we know the variables to optimize.
    self._beta1_power = None
    self._beta2_power = None

    # Created in SparseApply if needed.
    self._updated_lr = None
