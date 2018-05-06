/* Copyright 2018-present LHRParser Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * -----------------------------------------------------------------------------*/

package com.grellacangialosi.lhrparser.encoders.contextencoder

/**
 * A simple [ContextEncoder] builder.
 *
 * @param model the encoder model
 */
class ContextEncoderBuilder(model: ContextEncoderModel){

  /**
   * The context encoder.
   */
  private val contextEncoder = ContextEncoder(model)

  /**
   * @return the [contextEncoder]
   */
  operator fun invoke(): ContextEncoder = this.contextEncoder
}
