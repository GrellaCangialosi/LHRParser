/* Copyright 2018-present LHRParser Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * -----------------------------------------------------------------------------*/

package com.grellacangialosi.lhrparser.labeler

import com.kotlinnlp.simplednn.core.functionalities.updatemethods.UpdateMethod
import com.kotlinnlp.simplednn.core.optimizer.Optimizer
import com.kotlinnlp.simplednn.core.optimizer.ParamsOptimizer
import com.kotlinnlp.simplednn.deeplearning.multitasknetwork.MultiTaskNetworkParameters

/**
 * @param updateMethod the update method helper (Learning Rate, ADAM, AdaGrad, ...)
 * @param model the model of this optimizer
 */
class DeprelAndPOSLabelerOptimizer(
  private val model: DeprelAndPOSLabelerModel,
  updateMethod: UpdateMethod<*>
) : Optimizer(
  updateMethod = updateMethod
) {

  /**
   * The optimizer of the network
   */
  private val optimizer: ParamsOptimizer<MultiTaskNetworkParameters> =
    ParamsOptimizer(params = this.model.multitaskNetworkModel.params, updateMethod = this.updateMethod)

  /**
   * Update the parameters of the neural element associated to this optimizer.
   */
  override fun update() {
    this.optimizer.update()
  }

  /**
   * Accumulate the given [paramsErrors] into the accumulator.
   *
   * @param paramsErrors the parameters errors to accumulate
   * @param copy a Boolean indicating if the [paramsErrors] can be used as reference or must be copied. Set copy = false
   *             to optimize the accumulation when the amount of the errors to accumulate is 1. (default = true)
   */
  fun accumulate(paramsErrors: DeprelAndPOSLabelerParams, copy: Boolean = true) {

    paramsErrors.multiTaskParams.forEach {
      this.optimizer.accumulate(paramsErrors = it, copy = copy)
    }
  }
}