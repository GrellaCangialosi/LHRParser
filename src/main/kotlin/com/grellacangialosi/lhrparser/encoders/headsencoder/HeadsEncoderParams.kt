package com.grellacangialosi.lhrparser.encoders.headsencoder

import com.kotlinnlp.simplednn.core.neuralnetwork.NetworkParameters
import com.kotlinnlp.simplednn.deeplearning.birnn.BiRNNParameters

/**
 * @property biRNNParameters the params of the BiRNN of the [HeadsEncoder]
 * @property feedforwardParameters the param of the feedforward output of the [HeadsEncoder]
 */
class HeadsEncoderParams(val biRNNParameters: BiRNNParameters, val feedforwardParameters: NetworkParameters)