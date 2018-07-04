/* Copyright 2018-present LHRParser Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * -----------------------------------------------------------------------------*/

package com.grellacangialosi.lhrparser.labeler

import com.grellacangialosi.lhrparser.LatentSyntacticStructure
import com.grellacangialosi.lhrparser.labeler.utils.LossCriterion
import com.kotlinnlp.dependencytree.DependencyTree
import com.kotlinnlp.dependencytree.Deprel
import com.kotlinnlp.simplednn.core.neuralprocessor.batchfeedforward.BatchFeedforwardProcessor
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory

/**
 * @param model the model of this labeler
 */
class DeprelLabeler(private val model: DeprelLabelerModel) {

  /**
   * The input errors of this labeler.
   */
  class InputErrors (
    val rootErrors: DenseNDArray,
    val contextErrors: List<DenseNDArray>)

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
   * The dependency tree used for the last predictions done.
   */
  private lateinit var dependencyTree: DependencyTree

  /**
   * The last predictions done.
   */
  private lateinit var lastPredictions: List<Prediction>

  /**
   * Predict the deprel and the POS tag for each token.
   *
   * @param lss the latent syntactic structure
   * @param dependencyTree the dependency tree
   *
   * @return a list of predictions, one for each token
   */
  fun predict(lss: LatentSyntacticStructure, dependencyTree: DependencyTree): List<Prediction> {

    this.initialize()

    this.dependencyTree = dependencyTree

    val features = FeaturesExtractor(lss, dependencyTree, this.model.paddingVector).extract()

    val outputList: List<DenseNDArray> = this.processor.forward(ArrayList(features))

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
  fun getInputErrors(): InputErrors {

    val inputErrors: List<List<DenseNDArray>> = this.processor.getInputsErrors(copy = false)

    val contextErrors = List(size = inputErrors.size, init = {
      DenseNDArrayFactory.zeros(Shape(this.model.contextEncodingSize))
    })

    val rootErrors: DenseNDArray = DenseNDArrayFactory.zeros(Shape(this.model.contextEncodingSize))

    inputErrors.forEachIndexed { tokenId, (depErrors, govErrors) ->

      val depVector: DenseNDArray = contextErrors[tokenId]
      val govVector: DenseNDArray = this.dependencyTree.heads[tokenId].let {
        if (it == null) rootErrors else contextErrors[it]
      }

      depVector.assignSum(depErrors)
      govVector.assignSum(govErrors)
    }

    return InputErrors(rootErrors = rootErrors, contextErrors = contextErrors)
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

      errorsList.add(LossCriterion(this.model.lossCriterionType).getPredictionErrors(
        prediction = prediction.deprels, goldIndex = goldDeprelIndex))
    }

    return errorsList
  }

  /**
   * @param index a prediction index
   *
   * @return the [Deprel] corrisponding to the given [index]
   */
  fun getDeprel(index: Int) = this.model.deprels.getElement(index)!!
}
