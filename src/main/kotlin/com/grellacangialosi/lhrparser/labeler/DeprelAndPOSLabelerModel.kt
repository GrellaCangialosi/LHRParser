/* Copyright 2018-present LHRParser Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * -----------------------------------------------------------------------------*/

package com.grellacangialosi.lhrparser.labeler

import com.google.common.collect.HashMultimap
import com.kotlinnlp.dependencytree.Deprel
import com.kotlinnlp.dependencytree.POSTag
import com.kotlinnlp.simplednn.core.functionalities.activations.Softmax
import com.kotlinnlp.simplednn.core.functionalities.activations.Tanh
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.deeplearning.multitasknetwork.MultiTaskNetworkConfig
import com.kotlinnlp.simplednn.deeplearning.multitasknetwork.MultiTaskNetworkModel
import com.kotlinnlp.simplednn.utils.DictionarySet
import java.io.Serializable

/**
 * The model of the [DeprelAndPOSLabeler].
 *
 * @property tokenEncodingSize the size of the token encoding vectors
 * @property deprels the dictionary set of all possible deprels
 * @property posTags the dictionary set of all possible POS tags
 * @property deprelPosTagCombinations the map of all the possible deprel-posTags combinations
 * @property predictPosTags whether to predict the POS tags together with the Deprels
 * @property trainingMode the training mode
 */
class DeprelAndPOSLabelerModel(
  val tokenEncodingSize: Int,
  val deprels: DictionarySet<Deprel>,
  val posTags: DictionarySet<POSTag>,
  val deprelPosTagCombinations: HashMultimap<Deprel, POSTag>,
  val predictPosTags: Boolean,
  val trainingMode: LabelerTrainingMode
) : Serializable {

  /**
   * The Multitask Network model that predicts the Deprels and the POS tags.
   */
  val multitaskNetworkModel: MultiTaskNetworkModel = this.buildMultitaskNetwork()

  /**
   * @return the MultitaskNetwork model
   */
  private fun buildMultitaskNetwork(): MultiTaskNetworkModel {

    val deprelsConfig = MultiTaskNetworkConfig(
      outputSize = this.deprels.size,
      outputActivation = when (this.trainingMode) {
        LabelerTrainingMode.Softmax -> Softmax()
        LabelerTrainingMode.HingeLoss -> null
      },
      outputMeProp = false)

    val posTagsConfig = MultiTaskNetworkConfig(
      outputSize = this.posTags.size,
      outputActivation = when (this.trainingMode) {
        LabelerTrainingMode.Softmax -> Softmax()
        LabelerTrainingMode.HingeLoss -> null
      },
      outputMeProp = false)

    return MultiTaskNetworkModel(
      inputType = LayerType.Input.Dense,
      inputSize = 2 * this.tokenEncodingSize, // [dependent, governor]
      inputDropout = 0.0,
      hiddenSize = 100,
      hiddenActivation = Tanh(),
      hiddenDropout = 0.0,
      hiddenMeProp = false,
      outputConfigurations = if (this.predictPosTags) listOf(deprelsConfig, posTagsConfig) else listOf(deprelsConfig)
    )
  }
}