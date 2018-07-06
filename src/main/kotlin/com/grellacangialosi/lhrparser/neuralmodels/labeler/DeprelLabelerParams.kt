/* Copyright 2018-present LHRParser Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * -----------------------------------------------------------------------------*/

package com.grellacangialosi.lhrparser.neuralmodels.labeler

import com.kotlinnlp.simplednn.core.neuralnetwork.NetworkParameters

/**
 * @property params the parameters of the [DeprelLabeler]
 */
class DeprelLabelerParams(val params: NetworkParameters)