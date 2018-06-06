/* Copyright 2018-present LHRParser Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * -----------------------------------------------------------------------------*/

package com.grellacangialosi.lhrparser.encoders.dependentsencoder

/**
 * A simple [DependentsEncoder] builder.
 *
 * @param model the encoder model
 */
class DependentsEncoderBuilder(model: DependentsEncoderModel){

  /**
   * The context encoder.
   */
  private val dependentsEncoder = DependentsEncoder(model)

  /**
   * @return the [dependentsEncoder]
   */
  operator fun invoke(): DependentsEncoder = this.dependentsEncoder
}
