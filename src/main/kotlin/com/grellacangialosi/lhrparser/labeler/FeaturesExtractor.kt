package com.grellacangialosi.lhrparser.labeler

import com.grellacangialosi.lhrparser.LatentSyntacticStructure
import com.kotlinnlp.dependencytree.DependencyTree
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * The features extractor of the [DeprelLabeler].
 *
 * @param lss the latent syntactic structure
 * @param dependencyTree the dependency tree
 * @param paddingVector the vector to use in case of null item
 */
class FeaturesExtractor(
  private val lss: LatentSyntacticStructure,
  private val dependencyTree: DependencyTree,
  private val paddingVector: DenseNDArray
) {

  /**
   * @return a list of features
   */
  fun extract(): List<List<DenseNDArray>> {

    val features = mutableListOf<List<DenseNDArray>>()

    this.lss.tokens.map { it.id }.zip(this.dependencyTree.heads).forEach { (dependentId, headId) ->

      val depLeftMostChildId: Int? = if (this.dependencyTree.leftDependents[dependentId].isNotEmpty())
        this.dependencyTree.leftDependents[dependentId].first()
      else
        null

      val depRightMostChildId: Int? = if (this.dependencyTree.rightDependents[dependentId].isNotEmpty())
        this.dependencyTree.rightDependents[dependentId].last()
      else
        null

      features.add(listOf(
        this.lss.contextVectors[dependentId],
        headId?.let { this.lss.contextVectors[it] } ?: this.lss.virtualRoot,
        depLeftMostChildId?.let { this.lss.contextVectors[it] } ?: this.paddingVector,
        depRightMostChildId?.let { this.lss.contextVectors[it] } ?: this.paddingVector
      ))
    }

    return features
  }
}