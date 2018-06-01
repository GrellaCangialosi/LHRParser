/* Copyright 2018-present LHRParser Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * -----------------------------------------------------------------------------*/

package com.grellacangialosi.lhrparser.encoders.headsencoder

import com.kotlinnlp.simplednn.core.functionalities.updatemethods.UpdateMethod
import com.kotlinnlp.simplednn.core.neuralnetwork.NetworkParameters
import com.kotlinnlp.simplednn.core.optimizer.Optimizer
import com.kotlinnlp.simplednn.core.optimizer.ParamsOptimizer
import com.kotlinnlp.simplednn.deeplearning.birnn.BiRNNParameters

/**
 * @param updateMethod the update method helper (Learning Rate, ADAM, AdaGrad, ...)
 * @param model the model of this optimizer
 */
class HeadsEncoderOptimizer(
  private val model: HeadsEncoderModel,
  updateMethod: UpdateMethod<*>
) : Optimizer(
  updateMethod = updateMethod
) {

  /**
   * The Optimizer of the encoder parameters.
   */
  private val encoderOptimizer: ParamsOptimizer<BiRNNParameters> =
    ParamsOptimizer(params = this.model.biRNN.model, updateMethod = this.updateMethod)

  /**
   * The Optimizer of the outputEncoder parameters.
   */
  private val outputEncoderOptimizer: ParamsOptimizer<NetworkParameters> =
    ParamsOptimizer(params = this.model.outputNetwork.model, updateMethod = this.updateMethod)

  /**
   * Update the parameters of the neural element associated to this optimizer.
   */
  override fun update() {
    this.encoderOptimizer.update()
    this.outputEncoderOptimizer.update()
  }

  /**
   * Accumulate the given params errors into the accumulator.
   *
   * @param paramsErrors the parameters errors to accumulate
   * @param copy a Boolean indicating if the params errors can be used as reference or must be copied. Set copy = false
   *             to optimize the accumulation when the amount of the errors to accumulate is 1. (default = true)
   */
  fun accumulate(paramsErrors: HeadsEncoderParams, copy: Boolean = true) {

    this.outputEncoderOptimizer.accumulate(paramsErrors.feedforwardParameters, copy = copy)
    this.encoderOptimizer.accumulate(paramsErrors.biRNNParameters, copy = copy)
  }
}