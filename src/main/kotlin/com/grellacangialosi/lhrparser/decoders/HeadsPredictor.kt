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
import com.kotlinnlp.simplednn.core.functionalities.activations.*
import com.kotlinnlp.simplednn.core.functionalities.initializers.GlorotInitializer
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.adam.ADAMMethod
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.neuralnetwork.NetworkParameters
import com.kotlinnlp.simplednn.core.neuralnetwork.NeuralNetwork
import com.kotlinnlp.simplednn.core.neuralnetwork.preset.FeedforwardNeuralNetwork
import com.kotlinnlp.simplednn.core.neuralprocessor.feedforward.FeedforwardNeuralProcessor
import com.kotlinnlp.simplednn.core.optimizer.ParamsOptimizer
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory

/**
 * The HeadsPredictor decoder.
 */
class HeadsPredictor : LSSDecoder {

  /**
   *
   */
  private lateinit var network: NeuralNetwork

  /**
   *
   */
  private lateinit var processor: FeedforwardNeuralProcessor<DenseNDArray>

  /**
   *
   */
  private lateinit var optimizer: ParamsOptimizer<NetworkParameters>

  /**
   * Calculate the arc-scores.
   *
   * @param lss the latent syntactic structure to decode
   *
   * @return the computed scores
   */
  override fun decode(lss: LatentSyntacticStructure): ArcScores {

    this.train(lss)

    val scores = mutableMapOf<Int, MutableMap<Int, Double>>()

    lss.latentHeads.forEachIndexed { tokenIndex, latentHead ->

      val prediction: DenseNDArray = this.processor.forward(latentHead, useDropout = false)

      scores[tokenIndex] = mutableMapOf()

      (0 until prediction.length).forEach { headIndex ->

        if (headIndex == lss.contextVectors.lastIndex + 1) {
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
   */
  private fun train(lss: LatentSyntacticStructure) {

    val inputSize = lss.contextVectors.first().length

    this.network = FeedforwardNeuralNetwork(
      inputType = LayerType.Input.Dense,
      inputSize = inputSize,
      hiddenSize = 50,
      hiddenActivation = Tanh(),
      outputSize = lss.size + 1, // + root
      outputActivation = Softmax(),
      weightsInitializer =  GlorotInitializer(),
      biasesInitializer = null)

    this.optimizer = ParamsOptimizer(
      params = this.network.model,
      updateMethod = ADAMMethod(stepSize = 0.001, beta1 = 0.9, beta2 = 0.999))

    this.processor = FeedforwardNeuralProcessor(this.network)

    (0 .. 100).forEach {

      this.optimizer.newBatch()

      this.learn(lss.virtualRoot, goldIndex = lss.contextVectors.lastIndex + 1)

      lss.contextVectors.forEachIndexed { index, vector ->
        this.learn(vector, goldIndex = index)
      }

      this.optimizer.update()
    }
  }


  /**
   * @param inputVector the input vector
   * @param goldIndex the gold index
   */
  private fun learn(inputVector: DenseNDArray, goldIndex: Int) {

    val prediction = this.processor.forward(inputVector, useDropout = true)
    val argmaxIndex = prediction.argMaxIndex()

    if (argmaxIndex != goldIndex || prediction[argmaxIndex] < 0.99) {

      this.optimizer.newExample()

      val expectedValues = DenseNDArrayFactory.oneHotEncoder(this.network.outputSize, oneAt = goldIndex)

      val errors = SoftmaxCrossEntropyCalculator().calculateErrors(
        output = prediction,
        outputGold = expectedValues)

      this.propagateErrors(errors)
    }
  }

  /**
   * @param errors the errors to propagate
   */
  private fun propagateErrors(errors: DenseNDArray) {
    this.processor.backward(errors)
    this.optimizer.accumulate(this.processor.getParamsErrors(copy = false))
  }
}