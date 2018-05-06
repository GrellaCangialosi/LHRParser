/* Copyright 2018-present LHRParser Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * -----------------------------------------------------------------------------*/

package com.grellacangialosi.lhrparser.labeler

/**
 * The training mode of the labeler.
 *
 * @property Softmax activate the output with a Softmax function and calculate the errors as mean errors
 * @property HingeLoss don't activate the output and calculate the errors with the hinge loss method
 */
enum class LabelerTrainingMode { Softmax, HingeLoss }
