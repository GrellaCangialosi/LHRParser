/* Copyright 2018-present LHRParser Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * -----------------------------------------------------------------------------*/

package com.grellacangialosi.lhrparser.encoders.dependentsencoder

import com.kotlinnlp.simplednn.core.neuralprocessor.recurrent.RecurrentNeuralProcessor
import com.kotlinnlp.simplednn.core.neuralprocessor.recurrent.RecurrentNeuralProcessorsPool
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
  private inner class Element {

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
    fun forward(): Array<DenseNDArray> {
      TODO()
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
   * @param tokensVectors the vectors that represent each token
   *
   * @return the latent heads representation
   */
  fun encode(tokensVectors: Array<DenseNDArray>): Array<DenseNDArray> {


    TODO()
  }

  /**
   * @param errors the errors of the current encoding
   */
  fun backward(errors: Array<DenseNDArray>) {
    TODO()
  }

  /**
   * @param copy whether to return by value or by reference
   *
   * @return the input errors
   */
  fun getInputErrors(copy: Boolean = true): Array<DenseNDArray> = TODO()

  /**
   * @param copy a Boolean indicating whether the returned errors must be a copy or a reference
   *
   * @return the errors of the [DependentsEncoder] parameters
   */
  fun getParamsErrors(copy: Boolean = true) {
    TODO()
  }
}
