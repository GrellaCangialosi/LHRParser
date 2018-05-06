/* Copyright 2018-present LHRParser Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * -----------------------------------------------------------------------------*/

package com.grellacangialosi.lhrparser.encoders.headsencoder

import com.kotlinnlp.simplednn.encoders.sequenceencoder.SequenceFeedforwardEncoder
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.neuralparser.language.Token
import com.kotlinnlp.simplednn.encoders.birnn.BiRNNEncoder
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory

/**
 * Encoder that generates the Latent Heads Representation.
 *
 * @param model the model of this encoder
 */
class HeadsEncoder(private val model: HeadsEncoderModel) {

  /**
   * A BiRNN Encoder that encodes the latent heads.
   */
  private val encoder = BiRNNEncoder<DenseNDArray>(this.model.biRNN)

  /**
   * The Feedforward Encoder that reduces the size of the output of the [encoder].
   */
  private val outputEncoder = SequenceFeedforwardEncoder<DenseNDArray>(this.model.outputNetwork)

  /**
   * @param tokensVectors the vectors that represent each token
   *
   * @return the latent heads representation
   */
  fun encode(tokensVectors: Array<DenseNDArray>): Array<DenseNDArray> =
    this.outputEncoder.encode(this.encoder.encode(tokensVectors))

  /**
   * @param errors the errors of the current encoding
   */
  fun backward(errors: Array<DenseNDArray>) {

    this.outputEncoder.backward(errors, propagateToInput = true)
    return this.encoder.backward(this.outputEncoder.getInputSequenceErrors(copy = false), propagateToInput = true)
  }

  /**
   * @param copy whether to return by value or by reference
   *
   * @return the input errors
   */
  fun getInputErrors(copy: Boolean = true): Array<DenseNDArray> = this.encoder.getInputSequenceErrors(copy = copy)

  /**
   * @param copy a Boolean indicating whether the returned errors must be a copy or a reference
   *
   * @return the errors of the [HeadsEncoder] parameters
   */
  fun getParamsErrors(copy: Boolean = true) = HeadsEncoderParams(
    biRNNParameters = this.encoder.getParamsErrors(copy = copy),
    feedforwardParameters = this.outputEncoder.getParamsErrors(copy = copy)
  )

  /**
   * Get the list of RAN importance scores of each token of the last sentence parsed.
   * The importance scores are dense arrays (with size equal to the number of tokens), containing the importance of
   * each token respect to a given one. The score of a token respect to itself is always -1.
   *
   * This method should be called only after the parsing of a sentence.
   * It is required that the networks structures contain only a RAN layer.
   *
   * @param tokens the tokens of the last parsed sentence
   *
   * @return the list of importance scores of the last parsed sentence
   */
  @Suppress("unused")
  fun getRANImportanceScores(tokens: List<Token>): List<DenseNDArray> {

    val tokensSize: Int = tokens.size

    return tokens.zip(this.encoder.getRANImportanceScores()).map { (token, scores) ->
      DenseNDArrayFactory.arrayOf(DoubleArray(
        size = tokensSize,
        init = { i ->

          val leftScores: DenseNDArray? = scores.first
          val rightScores: DenseNDArray? = scores.second

          when {
            i < token.id -> leftScores!![i]
            i > token.id -> rightScores!![tokensSize - i - 1] // reversed order
            else -> -1.0
          }
        }
      ))
    }
  }
}
