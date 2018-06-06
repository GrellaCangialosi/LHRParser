/* Copyright 2018-present LHRParser Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * -----------------------------------------------------------------------------*/

package com.grellacangialosi.lhrparser.encoders.dependentsencoder

import com.kotlinnlp.simplednn.core.functionalities.updatemethods.UpdateMethod
import com.kotlinnlp.simplednn.core.neuralnetwork.NetworkParameters
import com.kotlinnlp.simplednn.core.optimizer.Optimizer
import com.kotlinnlp.simplednn.core.optimizer.ParamsOptimizer

/**
 * @param updateMethod the update method helper (Learning Rate, ADAM, AdaGrad, ...)
 * @param model the model of this optimizer
 */
class DependentsEncoderOptimizer(
  private val model: DependentsEncoderModel,
  updateMethod: UpdateMethod<*>
) : Optimizer(
  updateMethod = updateMethod
) {

  /**
   * The Optimizer of the left encoder parameters.
   */
  private val leftRNNOptimizer: ParamsOptimizer<NetworkParameters> =
    ParamsOptimizer(params = this.model.leftRNN.model, updateMethod = this.updateMethod)

  /**
   * The Optimizer of the right encoder parameters.
   */
  private val rightRNNOptimizer: ParamsOptimizer<NetworkParameters> =
    ParamsOptimizer(params = this.model.rightRNN.model, updateMethod = this.updateMethod)

  /**
   * Update the parameters of the neural element associated to this optimizer.
   */
  override fun update() {
    this.leftRNNOptimizer.update()
    this.rightRNNOptimizer.update()
  }

  /**
   * Accumulate the given params errors into the accumulator.
   *
   * @param paramsErrors the parameters errors to accumulate
   * @param copy a Boolean indicating if the params errors can be used as reference or must be copied. Set copy = false
   *             to optimize the accumulation when the amount of the errors to accumulate is 1. (default = true)
   */
  fun accumulate(paramsErrors: DependentsEncoderParams, copy: Boolean = true) {

    this.leftRNNOptimizer.accumulate(paramsErrors.leftRNN, copy = copy)
    this.rightRNNOptimizer.accumulate(paramsErrors.rightRNN, copy = copy)
  }
}