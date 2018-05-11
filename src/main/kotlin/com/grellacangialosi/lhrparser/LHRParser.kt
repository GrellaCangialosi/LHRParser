/* Copyright 2018-present LHRParser Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * -----------------------------------------------------------------------------*/

package com.grellacangialosi.lhrparser

import com.grellacangialosi.lhrparser.decoders.HeadsDecoder
import com.grellacangialosi.lhrparser.utils.ArcScores
import com.kotlinnlp.dependencytree.DependencyTree
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.grellacangialosi.lhrparser.encoders.contextencoder.ContextEncoder
import com.grellacangialosi.lhrparser.encoders.contextencoder.ContextEncoderBuilder
import com.grellacangialosi.lhrparser.encoders.headsencoder.HeadsEncoder
import com.grellacangialosi.lhrparser.encoders.headsencoder.HeadsEncoderBuilder
import com.grellacangialosi.lhrparser.labeler.DeprelAndPOSLabeler
import com.grellacangialosi.lhrparser.labeler.DeprelAndPOSLabelerBuilder
import com.grellacangialosi.lhrparser.utils.ArcScores.Companion.rootId
import com.grellacangialosi.lhrparser.utils.CyclesFixer
import com.kotlinnlp.neuralparser.NeuralParser
import com.kotlinnlp.neuralparser.language.Sentence
import com.kotlinnlp.simplednn.attention.pointernetwork.PointerNetwork
import com.kotlinnlp.tokensencoder.TokensEncoderFactory

/**
 * The Latent Head Representation (LHR) Parser.
 *
 * Implemented as described in the following publication:
 *   [Non-Projective Dependency Parsing via Latent Heads Representation (LHR)](https://arxiv.org/abs/1802.02116)
 *
 * @property model the parser model
 */
class LHRParser(override val model: LHRModel) : NeuralParser<LHRModel> {

  /**
   * A more convenient access to the embeddings values of the virtual root.
   */
  private val virtualRoot: DenseNDArray get() = this.model.rootEmbedding.array.values

  /**
   * The builder of the tokens encoder.
   */
  private val tokensEncoderBuilder = TokensEncoderFactory(this.model.tokensEncoderModel, trainingMode = false)

  /**
   * The builder of the [ContextEncoder].
   */
  private val contextEncoderBuilder = ContextEncoderBuilder(this.model.contextEncoderModel)

  /**
   * The builder of the [HeadsEncoder].
   */
  private val headsEncoderBuilder = HeadsEncoderBuilder(this.model.headsEncoderModel)

  /**
   * The builder of the labeler.
   */
  private val deprelAndPOSLabelerBuilder: DeprelAndPOSLabelerBuilder? = this.model.labelerModel?.let {
    DeprelAndPOSLabelerBuilder(model = it, rootVector = this.virtualRoot)
  }

  /**
   * The pointer network
   */
  private val pointerNetwork = PointerNetwork(this.model.pointerNetworkModel) // TODO: to check

  /**
   * Parse a sentence, returning its dependency tree.
   * The dependency tree is obtained by decoding a latent syntactic structure.
   * If the labeler is available, the dependency tree can contains deprel and posTag annotations.
   *
   * @param sentence a [Sentence]
   *
   * @return the dependency tree predicted for the given [sentence]
   */
  override fun parse(sentence: Sentence): DependencyTree {

    val encoder: LSSEncoder = this.buildEncoder()
    val lss = encoder.encode(sentence.tokens)

    val scores: ArcScores = HeadsDecoder().decode(lss)
    //val scores2: ArcScores = HeadsPointer(this.pointerNetwork).decode(lss) // TODO: to try

    return this.buildDependencyTree(lss, scores)
  }

  /**
   * @return a LSSEncoder
   */
  fun buildEncoder() = LSSEncoder(
    tokensEncoder = this.tokensEncoderBuilder.invoke(),
    contextEncoder = this.contextEncoderBuilder.invoke(),
    headsEncoder = this.headsEncoderBuilder.invoke(),
    virtualRoot = this.virtualRoot)

  /**
   * @param lss the latent syntactic structure
   * @param scores the attachment scores
   *
   * @return the annotated dependency tree (without cycles)
   */
  private fun buildDependencyTree(lss: LatentSyntacticStructure, scores: ArcScores): DependencyTree {

    val dependencyTree = DependencyTree(lss.tokens.size)

    this.assignHeads(dependencyTree, scores)
    this.fixCycles(dependencyTree, scores)
    this.assignLabels(dependencyTree, lss)

    return dependencyTree
  }

  /**
   * @param dependencyTree the dependency tree to annotate with the heads
   * @param scores the attachment scores
   */
  fun assignHeads(dependencyTree: DependencyTree, scores: ArcScores) {

    val (topId: Int, topScore: Double) = scores.findHighestScoringTop()

    dependencyTree.setAttachmentScore(dependent = topId, score = topScore)

    scores.filterNot { it.key == topId }.forEach { depId, _ ->

      val (govId: Int, score: Double) = scores.findHighestScoringHead(dependentId = depId, except = listOf(rootId))!!

      dependencyTree.setArc(
        dependent = depId,
        governor = govId,
        allowCycle = true,
        score = score)
    }
  }

  /**
   * @param dependencyTree the dependency tree to fix
   * @param scores the attachment scores
   */
  private fun fixCycles(dependencyTree: DependencyTree, scores: ArcScores) {
    CyclesFixer(dependencyTree, scores).fixCycles()
  }

  /**
   * @param dependencyTree the dependency tree to annotate with pos-tags and deprels
   * @param lss the latent syntactic structure
   */
  private fun assignLabels(dependencyTree: DependencyTree, lss: LatentSyntacticStructure) {

    val labeler: DeprelAndPOSLabeler? = this@LHRParser.deprelAndPOSLabelerBuilder?.invoke()

    labeler?.assignLabels(
      tokens = lss.tokens,
      tokensVectors = lss.contextVectors,
      dependencyTree = dependencyTree)
  }
}
