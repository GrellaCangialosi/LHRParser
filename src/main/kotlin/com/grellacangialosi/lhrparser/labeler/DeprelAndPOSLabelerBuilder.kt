/* Copyright 2018-present LHRParser Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * -----------------------------------------------------------------------------*/

package com.grellacangialosi.lhrparser.labeler

import com.grellacangialosi.lhrparser.encoders.headsencoder.HeadsEncoder
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * A simple [HeadsEncoder] builder.
 *
 * @param model the encoder model
 * @param rootVector the vector that represents the root token
 */
class DeprelAndPOSLabelerBuilder(model: DeprelAndPOSLabelerModel, rootVector: DenseNDArray){

  /**
   * The context encoder.
   */
  private val labeler = DeprelAndPOSLabeler(model, rootVector)

  /**
   * @return the [labeler]
   */
  operator fun invoke(): DeprelAndPOSLabeler = this.labeler
}
