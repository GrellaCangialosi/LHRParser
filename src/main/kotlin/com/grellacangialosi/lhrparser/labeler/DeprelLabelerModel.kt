/* Copyright 2018-present LHRParser Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * -----------------------------------------------------------------------------*/

package com.grellacangialosi.lhrparser.labeler

import com.grellacangialosi.lhrparser.labeler.utils.LossCriterionType
import com.kotlinnlp.dependencytree.Deprel
import com.kotlinnlp.simplednn.core.functionalities.activations.Softmax
import com.kotlinnlp.simplednn.core.functionalities.activations.Tanh
import com.kotlinnlp.simplednn.core.layers.LayerInterface
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.neuralnetwork.NeuralNetwork
import com.kotlinnlp.utils.DictionarySet
import java.io.Serializable

/**
 * The model of the [DeprelLabeler].
 *
 * @property contextEncodingSize the size of the token encoding vectors
 * @property deprels the dictionary set of all possible deprels
 * @property lossCriterionType the training mode
 */
class DeprelLabelerModel(
  val contextEncodingSize: Int,
  val deprels: DictionarySet<Deprel>,
  val lossCriterionType: LossCriterionType
) : Serializable {

  /**
   * The Network model that predicts the Deprels
   */
  val networkModel: NeuralNetwork = NeuralNetwork(
    // [contextDependent, contextGovernor]
    LayerInterface(sizes = listOf(this.contextEncodingSize, this.contextEncodingSize)),
    LayerInterface(
      size = 2 * this.contextEncodingSize,
      connectionType = LayerType.Connection.Concat,
      dropout = 0.2),
    LayerInterface(
      size = this.contextEncodingSize,
      connectionType = LayerType.Connection.LSTM,
      activationFunction = Tanh()),
    LayerInterface(
      type = LayerType.Input.Dense,
      size = this.deprels.size,
      dropout = 0.0,
      connectionType = LayerType.Connection.Feedforward,
      activationFunction = when (this.lossCriterionType) {
        LossCriterionType.Softmax -> Softmax()
        LossCriterionType.HingeLoss -> null
      })
  )
}