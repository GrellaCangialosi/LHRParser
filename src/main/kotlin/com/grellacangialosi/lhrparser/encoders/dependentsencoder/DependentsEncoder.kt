/* Copyright 2018-present LHRParser Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * -----------------------------------------------------------------------------*/

package com.grellacangialosi.lhrparser.encoders.dependentsencoder

import com.kotlinnlp.dependencytree.DependencyTree
import com.kotlinnlp.simplednn.core.functionalities.losses.MSECalculator
import com.kotlinnlp.simplednn.core.neuralnetwork.NetworkParameters
import com.kotlinnlp.simplednn.core.neuralprocessor.recurrent.RecurrentNeuralProcessor
import com.kotlinnlp.simplednn.core.neuralprocessor.recurrent.RecurrentNeuralProcessorsPool
import com.kotlinnlp.simplednn.core.optimizer.ParamsErrorsAccumulator
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * Encoder that generates the Latent Dependents Representation.
 *
 * @param model the model of this encoder
 */
class DependentsEncoder(private val model: DependentsEncoderModel) {

  /**
   *
   */
  private inner class Element(
    private val tokenVector: DenseNDArray,
    private val leftDependents: List<DenseNDArray>,
    private val rightDependents: List<DenseNDArray>) {

    /**
     * The [RecurrentNeuralProcessor] to encode the left-children.
     */
    internal val leftProcessor: RecurrentNeuralProcessor<DenseNDArray>
      = this@DependentsEncoder.leftProcessorsPool.getItem()

    /**
     * The [RecurrentNeuralProcessor] to encode the right-children.
     */
    internal val rightProcessor: RecurrentNeuralProcessor<DenseNDArray>
      = this@DependentsEncoder.rightProcessorsPool.getItem()

    /**
     *
     */
    fun learn() {
      this.leftProcessor.learn((listOf(this.tokenVector) + this.leftDependents))
      this.rightProcessor.learn((listOf(this.tokenVector) + this.rightDependents))
    }

    /**
     *
     */
    fun getInputErrors(): DenseNDArray =
      this.leftProcessor.getInputErrors(0).sum(this.rightProcessor.getInputErrors(0))

    /**
     *
     */
    fun getParamsErrors() = Pair(
      this.leftProcessor.getParamsErrors(copy = false),
      this.rightProcessor.getParamsErrors(copy = false))

    /**
     *
     */
    private fun RecurrentNeuralProcessor<DenseNDArray>.learn(inputArrays: List<DenseNDArray>) {

      this.forward(inputArrays)

      this.backward(
        MSECalculator().calculateErrors(
          outputSequence = this.getOutputSequence(copy = false),
          outputGoldSequence = inputArrays),
        propagateToInput = true)
    }
  }

  /**
   * The pool of processors for the left RNN.
   */
  private val leftProcessorsPool = RecurrentNeuralProcessorsPool<DenseNDArray>(this.model.leftRNN)

  /**
   * The pool of processors for the right RNN.
   */
  private val rightProcessorsPool = RecurrentNeuralProcessorsPool<DenseNDArray>(this.model.rightRNN)

  /**
   *
   */
  private val elements = mutableListOf<Element>()

  /**
   * @param tokensVectors the vectors that represent each token
   *
   * @return the latent heads representation
   */
  fun learn(tokensVectors: List<DenseNDArray>, dependencyTree: DependencyTree) {

    this.elements.clear()
    this.leftProcessorsPool.releaseAll()
    this.rightProcessorsPool.releaseAll()

    tokensVectors.forEachIndexed { tokenIndex, tokenVector ->

      val element = Element(
        tokenVector = tokenVector,
        leftDependents = dependencyTree.leftDependents[tokenIndex].reversed().map { tokensVectors[it] },
        rightDependents = dependencyTree.rightDependents[tokenIndex].map { tokensVectors[it] })

      element.learn()

      this.elements.add(element)
    }
  }

  /**
   * @param copy whether to return by value or by reference
   *
   * @return the input errors
   */
  fun getInputErrors(copy: Boolean = true): List<DenseNDArray> = this.elements.map { it.getInputErrors() }

  /**
   * @param copy a Boolean indicating whether the returned errors must be a copy or a reference
   *
   * @return the errors of the [DependentsEncoder] parameters
   */
  fun getParamsErrors(copy: Boolean = true): DependentsEncoderParams {

    val leftAccumulator = ParamsErrorsAccumulator<NetworkParameters>()
    val rightAccumulator = ParamsErrorsAccumulator<NetworkParameters>()

    this.elements.forEach { element ->
      val (leftParamsErrors, rightParamsErrors) = element.getParamsErrors()

      leftAccumulator.accumulate(leftParamsErrors)
      rightAccumulator.accumulate(rightParamsErrors)
    }

    leftAccumulator.averageErrors()
    rightAccumulator.averageErrors()

    return DependentsEncoderParams(
      leftRNN = leftAccumulator.getParamsErrors(copy = false),
      rightRNN = rightAccumulator.getParamsErrors(copy = false))
  }
}
