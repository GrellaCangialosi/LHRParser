/* Copyright 2018-present LHRParser Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * -----------------------------------------------------------------------------*/

package com.grellacangialosi.lhrparser.labeler.utils

import com.kotlinnlp.dependencytree.DependencyTree

/**
 * @param id the token id
 *
 * @return the left most child (can be null)
 */
fun DependencyTree.leftMostChild(id: Int): Int? =
  if (this.leftDependents[id].isEmpty()) null else this.leftDependents[id].first()

/**
 * @param id the token id
 *
 * @return the right most child (can be null)
 */
fun DependencyTree.rightMostChild(id: Int): Int? =
  if (this.rightDependents[id].isEmpty()) null else this.rightDependents[id].last()