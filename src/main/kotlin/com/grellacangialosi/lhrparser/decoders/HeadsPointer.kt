package com.grellacangialosi.lhrparser.decoders

import com.kotlinnlp.simplednn.core.functionalities.losses.SoftmaxCrossEntropyCalculator
import com.kotlinnlp.simplednn.deeplearning.pointernetwork.PointerNetwork
import com.kotlinnlp.simplednn.deeplearning.pointernetwork.PointerNetworkModel
import com.kotlinnlp.simplednn.deeplearning.pointernetwork.PointerNetworkParameters
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory

/**
 * The HeadsPointer decoder.
 *
 * @param model the model
 */
class HeadsPointer(model: PointerNetworkModel) {

  /**
   * The pointer network.
   * TODO: fix usage
   */
  val network = PointerNetwork(model)

  /**
   * @param tokensEncodings tokens encodings
   * @param goldHeads the heads of the gold dependency tree
   */
  fun learn(tokensEncodings: Array<DenseNDArray>, goldHeads: Array<Int?>) {

    this.network.setInputSequence(tokensEncodings.toList())

    val errorsList = tokensEncodings.mapIndexed { index, tokenEncoding ->

      val prediction = this.network.forward(tokenEncoding)

      val expectedValues = DenseNDArrayFactory.oneHotEncoder(
        length = tokensEncodings.size,
        oneAt = goldHeads[index] ?: index) // self-root

      SoftmaxCrossEntropyCalculator().calculateErrors(output = prediction, outputGold = expectedValues)
    }

    this.network.backward(errorsList)
  }

  /**
   * @return the params errors
   */
  fun getParamsErrors(copy: Boolean = true): PointerNetworkParameters = this.network.getParamsErrors(copy = copy)

  /**
   * @return the input errors
   */
  fun getInputErrors(): Array<DenseNDArray> {

    val inputErrors = this.network.getInputErrors()

    require(inputErrors.decodingSequenceErrors.size == inputErrors.inputSequenceErrors.size )

    return inputErrors.decodingSequenceErrors.zip(inputErrors.inputSequenceErrors).map {
      it.first.sum(it.second)
    }.toTypedArray()
  }
}