/* Copyright 2018-present LHRParser Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * -----------------------------------------------------------------------------*/

package com.grellacangialosi.lhrparser.labeler

import com.kotlinnlp.dependencytree.DependencyTree
import com.kotlinnlp.neuralparser.language.Token
import com.kotlinnlp.dependencytree.Deprel
import com.kotlinnlp.dependencytree.POSTag
import com.kotlinnlp.simplednn.core.functionalities.losses.SoftmaxCrossEntropyCalculator
import com.kotlinnlp.simplednn.core.functionalities.losses.getErrorsByHingeLoss
import com.kotlinnlp.simplednn.deeplearning.multitasknetwork.MultiTaskNetwork
import com.kotlinnlp.simplednn.deeplearning.multitasknetwork.MultitaskNetworksPool
import com.kotlinnlp.simplednn.simplemath.concatVectorsV
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory

/**
 * @param model the model of this labeler
 * @param rootVector the vector that represents the root token
 */
class DeprelAndPOSLabeler(private val model: DeprelAndPOSLabelerModel, private val rootVector: DenseNDArray) {

  /**
   * The outcome of a single prediction of the labeler.
   *
   * @property deprels the deprels prediction
   * @property posTags the POS tags prediction (can be null)
   */
  data class Prediction(val deprels: DenseNDArray, val posTags: DenseNDArray?) {

    /**
     * @return the list containing the predictions of this outcome
     */
    fun toList(): List<DenseNDArray> = if (this.posTags != null)
      listOf(this.deprels, this.posTags)
    else
      listOf(this.deprels)
  }

  /**
   * The pool of feed-forward multitask networks that classify deprels and POS tags.
   */
  private val networksPool = MultitaskNetworksPool<DenseNDArray>(this.model.multitaskNetworkModel)

  /**
   * The list of multitask networks of the [networksPool] used for the last prediction.
   */
  private val usedNetworks = mutableListOf<MultiTaskNetwork<DenseNDArray>>()

  /**
   * The tokens heads used for the last predictions done.
   */
  private lateinit var lastTokensHeads: Array<Int?>

  /**
   * The last predictions done.
   */
  private lateinit var lastPredictions: List<Prediction>

  /**
   * Assign the Deprel and POS tag labels.
   *
   * @param tokens a list of [Token]
   * @param tokensVectors the vectors that represent each token
   * @param dependencyTree the dependency tree
   */
  fun assignLabels(tokens: List<Token>, tokensVectors: List<DenseNDArray>, dependencyTree: DependencyTree) {

    val labelerPredictions = this.predict(
      tokens = tokens,
      tokensHeads = dependencyTree.heads,
      tokensVectors = tokensVectors)

    labelerPredictions.forEachIndexed { tokenId, prediction ->

      if (prediction.posTags == null) {

        dependencyTree.setDeprel(tokenId, this.getDeprel(prediction.deprels.argMaxIndex()))

      } else {

        dependencyTree.setDeprel(tokenId, this.getDeprel(prediction.deprels.argMaxIndex()))
        dependencyTree.setPosTag(tokenId, this.getPosTag(prediction.posTags.argMaxIndex()))
      }
    }
  }

  /**
   * Predict the deprel and the POS tag for each token.
   *
   * @param tokens a list of [Token]
   * @param tokensHeads the array containing the head id of each token (the null value represents the root)
   * @param tokensVectors the vectors that represent each token
   *
   * @return a list of predictions, one for each token
   */
  fun predict(tokens: List<Token>, tokensHeads: Array<Int?>, tokensVectors: List<DenseNDArray>): List<Prediction> {

    this.initialize()

    this.lastTokensHeads = tokensHeads

    this.lastPredictions = this
      .extractFeatures(tokens = tokens, tokensVectors = tokensVectors, tokensHeads = tokensHeads)
      .map {

        val outputList: List<DenseNDArray> = this.getNetwork().forward(it)

        Prediction(deprels = outputList[0], posTags = if (this.model.predictPosTags) outputList[1] else null)
      }

    return this.lastPredictions
  }

  /**
   * Propagate the errors through the neural components of the labeler. Errors are calculated comparing the last
   * predictions done with the given gold deprels and POS tags.
   *
   * @param goldDeprels the list of gold deprels
   * @param goldPosTags the list of gold POS tags
   */
  fun backward(goldDeprels: Array<Deprel?>,
               goldPosTags: Array<POSTag?>) {

    this.getPredictionsErrors(goldDeprels = goldDeprels, goldPosTags = goldPosTags).forEachIndexed { i, errors ->

      this.usedNetworks[i].backward(outputErrorsList = errors.toList(), propagateToInput = true)
    }
  }

  /**
   * @return the input errors and the root errors
   */
  fun getInputErrors(): Pair<List<DenseNDArray>, DenseNDArray> {

    val contextVectorsSize: Int = this.model.multitaskNetworkModel.inputSize / 2 // [dependent, governor] as input
    val errors = List(size = this.usedNetworks.size, init = { DenseNDArrayFactory.zeros(Shape(contextVectorsSize)) })
    lateinit var rootErrors: DenseNDArray

    this.usedNetworks
      .map { it.getInputErrors(copy = false) }
      .forEachIndexed { dependentId, inputErrors ->

        val splitErrors: List<DenseNDArray> = inputErrors.splitV(contextVectorsSize)
        val governorId: Int? = this.lastTokensHeads[dependentId]

        errors[dependentId].assignSum(splitErrors[0])

        if (governorId != null) {
          errors[governorId].assignSum(splitErrors[1])
        } else {
          rootErrors = splitErrors[1]
        }
      }

    return Pair(errors, rootErrors)
  }

  /**
   * @param copy a Boolean indicating whether the returned errors must be a copy or a reference
   *
   * @return the errors of the [DeprelAndPOSLabeler] parameters
   */
  fun getParamsErrors(copy: Boolean = true) = DeprelAndPOSLabelerParams(
    multiTaskParams = this.usedNetworks.map { it.getParamsErrors(copy = copy) } )

  /**
   * Initialize the labeler for the next sentence.
   */
  private fun initialize() {
    this.usedNetworks.clear()
    this.networksPool.releaseAll()
  }

  /**
   * @return a multitask network from the [networksPool]
   */
  private fun getNetwork(): MultiTaskNetwork<DenseNDArray> {
    this.usedNetworks.add(this.networksPool.getItem())
    return this.usedNetworks.last()
  }

  /**
   * Extract for each token the features to predict its dependency relation.
   *
   * @param tokens a list of [Token]
   * @param tokensHeads the array containing the head id of each token (the null value represents the root)
   * @param tokensVectors the vectors that represent each token
   *
   * @return a list of features
   */
  private fun extractFeatures(tokens: List<Token>,
                              tokensHeads: Array<Int?>,
                              tokensVectors: List<DenseNDArray>): List<DenseNDArray> {

    val features = mutableListOf<DenseNDArray>()

    tokens.map { it.id }.zip(tokensHeads).forEach { (dependentId, headId) ->

      val encodedHead: DenseNDArray = headId?.let { tokensVectors[it] } ?: this.rootVector

      features.add(concatVectorsV(tokensVectors[dependentId], encodedHead))
    }

    return features
  }

  /**
   * Return the errors of the last predictions done, respect to a gold dependency tree.
   *
   * @param goldDeprels the list of gold deprels
   * @param goldPosTags the list of gold POS tags
   *
   * @return a list of predictions errors
   */
  private fun getPredictionsErrors(goldDeprels: Array<Deprel?>, goldPosTags: Array<POSTag?>): List<Prediction> {

    val errorsList = mutableListOf<Prediction>()

    this.lastPredictions.forEachIndexed { tokenId, prediction ->

      val goldDeprel: Deprel = goldDeprels[tokenId]!!
      val goldDeprelIndex: Int = this.model.deprels.getId(goldDeprel)!!

      val goldPosTag: POSTag? = goldPosTags[tokenId]
      val goldPosTagIndex: Int? = if (goldPosTag != null) this.model.posTags.getId(goldPosTag) else null

      errorsList.add(
        Prediction(
          deprels = this.getPredictionErrors(prediction = prediction.deprels, goldIndex = goldDeprelIndex),
          posTags = if (prediction.posTags != null)
            this.getPredictionErrors(prediction = prediction.posTags, goldIndex = goldPosTagIndex!!)
          else
            null
        )
      )
    }

    return errorsList
  }

  /**
   * @param prediction a prediction array
   * @param goldIndex the index of the gold value
   *
   * @return the errors of the given prediction
   */
  private fun getPredictionErrors(prediction: DenseNDArray, goldIndex: Int): DenseNDArray =
    when (this.model.trainingMode) {
      LabelerTrainingMode.Softmax ->
        SoftmaxCrossEntropyCalculator().calculateErrors(output = prediction, goldIndex = goldIndex)

      LabelerTrainingMode.HingeLoss ->
        getErrorsByHingeLoss(prediction = prediction, goldIndex = goldIndex)
    }

  /**
   * @param deprelId a deprel id
   *
   * @return the [Deprel] of the given [deprelId]
   */
  private fun getDeprel(deprelId: Int) = this.model.deprels.getElement(deprelId)!!

  /**
   * @param posTagId a POS tag id
   *
   * @return the [POSTag] of the given [posTagId]
   */
  private fun getPosTag(posTagId: Int): POSTag = this.model.posTags.getElement(posTagId)!!
}
