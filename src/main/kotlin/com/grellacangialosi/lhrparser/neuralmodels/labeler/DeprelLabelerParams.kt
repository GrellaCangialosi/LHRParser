/* Copyright 2018-present LHRParser Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * -----------------------------------------------------------------------------*/

package com.grellacangialosi.lhrparser.neuralmodels.labeler

import com.kotlinnlp.simplednn.core.neuralnetwork.NetworkParameters
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * The parameters of the [DeprelLabeler].
 *
 * @property params the parameters of the deprel classifier
 * @property distanceEmbeddings the list pair of id and value of a distance embedding
 */
data class DeprelLabelerParams(
  val params: NetworkParameters,
  val distanceEmbeddings: List<Pair<Int, DenseNDArray>>
)