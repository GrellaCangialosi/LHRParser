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
import com.kotlinnlp.simplednn.core.functionalities.losses.SoftmaxCrossEntropyCalculator
import com.kotlinnlp.simplednn.core.functionalities.losses.getErrorsByHingeLoss
import com.kotlinnlp.simplednn.core.neuralprocessor.batchfeedforward.BatchFeedforwardProcessor
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory

/**
 * @param model the model of this labeler
 * @param rootVector the vector that represents the root token
 */
class DeprelLabeler(private val model: DeprelLabelerModel, private val rootVector: DenseNDArray) {

  /**
   * The outcome of a single prediction of the labeler.
   *
   * @property deprels the deprels prediction
   */
  data class Prediction(val deprels: DenseNDArray)

  /**
   * The processor that classify the deprels.
   */
  private val processor = BatchFeedforwardProcessor<DenseNDArray>(this.model.networkModel)

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

      dependencyTree.setDeprel(tokenId, this.getDeprel(prediction.deprels.argMaxIndex()))
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

    val outputList: List<DenseNDArray> = this.processor.forward(ArrayList(
      this.extractFeatures(tokens = tokens, tokensVectors = tokensVectors, tokensHeads = tokensHeads).map { it.toList() }))

    this.lastPredictions = outputList.map { Prediction(deprels = it) }

    return this.lastPredictions
  }

  /**
   * Propagate the errors through the neural components of the labeler. Errors are calculated comparing the last
   * predictions done with the given gold deprels and POS tags.
   *
   * @param goldDeprels the list of gold deprels
   */
  fun backward(goldDeprels: Array<Deprel?>) {

    this.processor.backward(outputErrors = this.getPredictionsErrors(goldDeprels = goldDeprels), propagateToInput = true)
  }

  /**
   * @return the input errors and the root errors
   */
  fun getInputErrors(): Pair<List<DenseNDArray>, DenseNDArray> {

    val inputErrors: List<List<DenseNDArray>> = this.processor.getInputsErrors(copy = false)
    val contextVectorsSize: Int = this.model.networkModel.inputsSize[0] // [dependent, governor] as input
    val errors = List(size = inputErrors.size, init = { DenseNDArrayFactory.zeros(Shape(contextVectorsSize)) })
    lateinit var rootErrors: DenseNDArray

    inputErrors.forEachIndexed { dependentId, vectorErrors ->

        val governorId: Int? = this.lastTokensHeads[dependentId]

        errors[dependentId].assignSum(vectorErrors[0])

        if (governorId != null) {
          errors[governorId].assignSum(vectorErrors[1])
        } else {
          rootErrors = vectorErrors[1]
        }
      }

    return Pair(errors, rootErrors)
  }

  /**
   * @param copy a Boolean indicating whether the returned errors must be a copy or a reference
   *
   * @return the errors of the [DeprelLabeler] parameters
   */
  fun getParamsErrors(copy: Boolean = true) = DeprelLabelerParams(
    params = this.processor.getParamsErrors(copy = copy))

  /**
   * Initialize the labeler for the next sentence.
   */
  private fun initialize() { }

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
                              tokensVectors: List<DenseNDArray>): List<List<DenseNDArray>> {

    val features = mutableListOf<List<DenseNDArray>>()

    tokens.map { it.id }.zip(tokensHeads).forEach { (dependentId, headId) ->

      val encodedHead: DenseNDArray = headId?.let { tokensVectors[it] } ?: this.rootVector

      features.add(listOf(tokensVectors[dependentId], encodedHead))
    }

    return features
  }

  /**
   * Return the errors of the last predictions done, respect to a gold dependency tree.
   *
   * @param goldDeprels the list of gold deprels
   *
   * @return a list of predictions errors
   */
  private fun getPredictionsErrors(goldDeprels: Array<Deprel?>): List<DenseNDArray> {

    val errorsList = mutableListOf<DenseNDArray>()

    this.lastPredictions.forEachIndexed { tokenId, prediction ->

      val goldDeprel: Deprel = goldDeprels[tokenId]!!
      val goldDeprelIndex: Int = this.model.deprels.getId(goldDeprel)!!

      errorsList.add(
        this.getPredictionErrors(prediction = prediction.deprels, goldIndex = goldDeprelIndex)
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
}
