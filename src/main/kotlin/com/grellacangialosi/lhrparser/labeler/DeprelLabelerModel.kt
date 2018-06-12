/* Copyright 2018-present LHRParser Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * -----------------------------------------------------------------------------*/

package com.grellacangialosi.lhrparser.labeler

import com.kotlinnlp.dependencytree.Deprel
import com.kotlinnlp.simplednn.core.functionalities.activations.Softmax
import com.kotlinnlp.simplednn.core.layers.LayerInterface
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.neuralnetwork.NeuralNetwork
import com.kotlinnlp.utils.DictionarySet
import java.io.Serializable

/**
 * The model of the [DeprelLabeler].
 *
 * @property tokenEncodingSize the size of the token encoding vectors
 * @property deprels the dictionary set of all possible deprels
 * @property trainingMode the training mode
 */
class DeprelLabelerModel(
  val tokenEncodingSize: Int,
  val deprels: DictionarySet<Deprel>,
  val trainingMode: LabelerTrainingMode
) : Serializable {

  /**
   * The Network model that predicts the Deprels
   */
  val networkModel: NeuralNetwork = NeuralNetwork(
    LayerInterface(sizes = listOf(this.tokenEncodingSize, this.tokenEncodingSize)), // [dependent, governor]
    LayerInterface(size = 2 * this.tokenEncodingSize, connectionType = LayerType.Connection.Concat, dropout = 0.2),
    LayerInterface(
      type = LayerType.Input.Dense,
      size = this.deprels.size,
      dropout = 0.0,
      connectionType = LayerType.Connection.Feedforward,
      activationFunction = when (this.trainingMode) {
        LabelerTrainingMode.Softmax -> Softmax()
        LabelerTrainingMode.HingeLoss -> null
      })
  )
}