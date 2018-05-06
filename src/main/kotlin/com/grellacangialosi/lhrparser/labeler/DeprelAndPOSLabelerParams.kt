/* Copyright 2018-present LHRParser Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * -----------------------------------------------------------------------------*/

package com.grellacangialosi.lhrparser.labeler

import com.kotlinnlp.simplednn.deeplearning.multitasknetwork.MultiTaskNetworkParameters

/**
 * @property multiTaskParams the parameters of the [DeprelAndPOSLabeler]
 */
class DeprelAndPOSLabelerParams(val multiTaskParams: List<MultiTaskNetworkParameters>)