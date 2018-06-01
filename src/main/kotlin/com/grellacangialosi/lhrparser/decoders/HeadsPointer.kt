/* Copyright 2018-present LHRParser Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * -----------------------------------------------------------------------------*/

package com.grellacangialosi.lhrparser.decoders

import com.grellacangialosi.lhrparser.LatentSyntacticStructure
import com.grellacangialosi.lhrparser.utils.ArcScores
import com.grellacangialosi.lhrparser.utils.ArcScores.Companion.rootId
import com.kotlinnlp.simplednn.core.functionalities.losses.SoftmaxCrossEntropyCalculator
import com.kotlinnlp.simplednn.deeplearning.attention.pointernetwork.PointerNetworkProcessor
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory

/**
 * The HeadsPointer decoder.
 *
 * @param networkProcessor the pointer network
 */
class HeadsPointer(private val networkProcessor: PointerNetworkProcessor) : LSSDecoder {

  /**
   * Calculate the arc-scores.
   *
   * @param lss the latent syntactic structure to decode
   *
   * @return the computed scores
   */
  override fun decode(lss: LatentSyntacticStructure): ArcScores {

    val scores = mutableMapOf<Int, MutableMap<Int, Double>>()

    this.networkProcessor.setInputSequence(lss.contextVectors.toList())

    lss.latentHeads.forEachIndexed { tokenIndex, latentHead ->

      val prediction: DenseNDArray = this.networkProcessor.forward(latentHead)

      scores[tokenIndex] = mutableMapOf()

      (0 until prediction.length).forEach { headIndex ->

        if (tokenIndex == headIndex) {
          scores.getValue(tokenIndex)[rootId] = prediction[headIndex]
        } else {
          scores.getValue(tokenIndex)[headIndex] = prediction[headIndex]
        }
      }
    }

    return ArcScores(scores)
  }

  /**
   * @param lss the latent syntactic structure
   * @param goldHeads the heads of the gold dependency tree
   */
  fun learn(lss: LatentSyntacticStructure, goldHeads: Array<Int?>) {

    this.networkProcessor.setInputSequence(lss.contextVectors.toList())
    this.networkProcessor.backward(this.calculateErrors(lss.latentHeads, goldHeads))
  }

  /**
   * @return the params errors
   */
  fun getParamsErrors(copy: Boolean = true) = this.networkProcessor.getParamsErrors(copy = copy)

  /**
   * @return the errors of the latent heads
   */
  fun getLatentHeadsErrors(): Array<DenseNDArray> = this.networkProcessor.getInputErrors().inputVectorsErrors.toTypedArray()

  /**
   * @return the errors of the context vectors
   */
  fun getContextVectorsErrors(): Array<DenseNDArray> = this.networkProcessor.getInputErrors().inputSequenceErrors.toTypedArray()

  /**
   * @param latentHeads the latent heads
   * @param goldHeads the heads of the gold dependency tree
   */
  private fun calculateErrors(latentHeads: List<DenseNDArray>, goldHeads: Array<Int?>): List<DenseNDArray> {

    return latentHeads.mapIndexed { index, latentHead ->

      val prediction: DenseNDArray = this.networkProcessor.forward(latentHead)

      val expectedValues = DenseNDArrayFactory.oneHotEncoder(
        length = this.networkProcessor.inputSequenceSize,
        oneAt = goldHeads[index] ?: index) // self-root

      SoftmaxCrossEntropyCalculator().calculateErrors(
        output = prediction,
        outputGold = expectedValues)
    }
  }
}