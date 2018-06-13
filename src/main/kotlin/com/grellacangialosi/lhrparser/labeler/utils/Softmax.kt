/* Copyright 2018-present LHRParser Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * -----------------------------------------------------------------------------*/

package com.grellacangialosi.lhrparser.labeler.utils

import com.kotlinnlp.simplednn.core.functionalities.losses.SoftmaxCrossEntropyCalculator
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 *
 */
class Softmax : LossCriterion {

  /**
   * @param prediction a prediction array
   * @param goldIndex the index of the gold value
   *
   * @return the errors of the given prediction
   */
  override fun getPredictionErrors(prediction: DenseNDArray, goldIndex: Int): DenseNDArray =
    SoftmaxCrossEntropyCalculator().calculateErrors(output = prediction, goldIndex = goldIndex)
}