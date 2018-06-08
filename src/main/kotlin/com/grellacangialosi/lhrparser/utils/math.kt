/* Copyright 2018-present LHRParser Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * -----------------------------------------------------------------------------*/

package com.grellacangialosi.lhrparser.utils

import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory

/**
 * Get the index of the highest value without considering the one at the given [exceptIndex].
 *
 * @param exceptIndex the index to exclude
 *
 * @return the index of the maximum value or -1 if empty
 **/
fun DenseNDArray.indexOfHighestExcept(exceptIndex: Int): Int {

  var maxIndex: Int = -1
  var maxValue: Double? = null

  (0 until this.length).forEach { i ->

    if (i != exceptIndex) {
      val value = this[i]

      if (maxValue == null || value > maxValue!!) {
        maxValue = value
        maxIndex = i
      }
    }
  }

  return maxIndex
}

/**
 * Get the prediction errors using the Hinge Loss method.
 *
 * @param prediction a prediction array
 * @param goldIndex the index of the gold value
 *
 * @return the errors of the given prediction
 */
fun getErrorsByHingeLoss(prediction: DenseNDArray, goldIndex: Int): DenseNDArray {

  val errors: DenseNDArray = DenseNDArrayFactory.zeros(prediction.shape)

  val highestScoringIncorrectIndex: Int = prediction.indexOfHighestExcept(exceptIndex = goldIndex)

  val margin: Double = prediction[goldIndex] - prediction[highestScoringIncorrectIndex]

  if (margin < 1.0) {
    errors[goldIndex] = -1.0
    errors[highestScoringIncorrectIndex] = 1.0
  }

  return errors
}

/**
 * Get the prediction errors when the output is activated with the Softmax function.
 *
 * @param prediction a prediction array
 * @param goldIndex the index of the gold value
 *
 * @return the errors of the given Deprel prediction
 */
fun getErrorsBySoftmax(prediction: DenseNDArray, goldIndex: Int): DenseNDArray {

  val errors: DenseNDArray = prediction.copy()

  errors[goldIndex] = errors[goldIndex] - 1.0

  return errors
}
