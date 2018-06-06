/* Copyright 2018-present LHRParser Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * -----------------------------------------------------------------------------*/

package com.grellacangialosi.lhrparser.encoders.dependentsencoder

import com.kotlinnlp.simplednn.core.neuralnetwork.NetworkParameters

/**
 * @property leftToRight the params of the left rnn of the [DependentsEncoder]
 * @property rightToLeft the params of the right rnn of the [DependentsEncoder]
 */
class DependentsEncoderParams(val leftToRight: NetworkParameters,
                              val rightToLeft: NetworkParameters)