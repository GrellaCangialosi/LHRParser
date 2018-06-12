/* Copyright 2018-present LHRParser Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.grellacangialosi.lhrparser

import com.grellacangialosi.lhrparser.decoders.HeadsPointer
import com.grellacangialosi.lhrparser.encoders.contextencoder.ContextEncoder
import com.grellacangialosi.lhrparser.encoders.contextencoder.ContextEncoderBuilder
import com.grellacangialosi.lhrparser.encoders.contextencoder.ContextEncoderOptimizer
import com.grellacangialosi.lhrparser.encoders.dependentsencoder.DependentsEncoderBuilder
import com.grellacangialosi.lhrparser.encoders.dependentsencoder.DependentsEncoderOptimizer
import com.grellacangialosi.lhrparser.encoders.headsencoder.HeadsEncoder
import com.grellacangialosi.lhrparser.encoders.headsencoder.HeadsEncoderBuilder
import com.grellacangialosi.lhrparser.encoders.headsencoder.HeadsEncoderOptimizer
import com.grellacangialosi.lhrparser.labeler.DeprelLabeler
import com.grellacangialosi.lhrparser.labeler.DeprelLabelerBuilder
import com.grellacangialosi.lhrparser.labeler.DeprelLabelerOptimizer
import com.grellacangialosi.lhrparser.utils.ArcScores
import com.kotlinnlp.dependencytree.DependencyTree
import com.kotlinnlp.dependencytree.POSTag
import com.kotlinnlp.neuralparser.helpers.Trainer
import com.kotlinnlp.neuralparser.helpers.Validator
import com.kotlinnlp.neuralparser.language.Sentence
import com.kotlinnlp.simplednn.deeplearning.attention.pointernetwork.PointerNetworkProcessor
import com.kotlinnlp.simplednn.core.functionalities.losses.MSECalculator
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.UpdateMethod
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.adam.ADAMMethod
import com.kotlinnlp.simplednn.core.optimizer.ParamsOptimizer
import com.kotlinnlp.simplednn.simplemath.assignSum
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.utils.scheduling.BatchScheduling
import com.kotlinnlp.simplednn.utils.scheduling.EpochScheduling
import com.kotlinnlp.simplednn.utils.scheduling.ExampleScheduling
import com.kotlinnlp.tokensencoder.TokensEncoder
import com.kotlinnlp.tokensencoder.TokensEncoderBuilder
import com.kotlinnlp.tokensencoder.TokensEncoderFactory
import com.kotlinnlp.tokensencoder.TokensEncoderOptimizerFactory

/**
 * The training helper.
 *
 * @param parser a neural parser
 * @param batchSize the size of the batches of sentences
 * @param epochs the number of training epochs
 * @param validator the validation helper (if it is null no validation is done after each epoch)
 * @param modelFilename the name of the file in which to save the best trained model
 * @param updateMethod the update method shared to all the parameters of the parser (Learning Rate, ADAM, AdaGrad, ...)
 * @param lhrErrorsOptions the settings for calculating the latent heads errors
 * @param verbose a Boolean indicating if the verbose mode is enabled (default = true)
 */
class LHRTrainer(
  private val parser: LHRParser,
  private val batchSize: Int,
  private val epochs: Int,
  validator: Validator?,
  modelFilename: String,
  private val updateMethod: UpdateMethod<*> = ADAMMethod(stepSize = 0.001, beta1 = 0.9, beta2 = 0.999),
  private val lhrErrorsOptions: LHRErrorsOptions,
  verbose: Boolean = true
) : Trainer(
  neuralParser = parser,
  batchSize = batchSize,
  epochs = epochs,
  validator = validator,
  modelFilename = modelFilename,
  minRelevantErrorsCountToUpdate = 1,
  verbose = verbose
) {

  /**
   * @property skipPunctuationErrors whether to do not consider punctuation errors
   */
  data class LHRErrorsOptions(val skipPunctuationErrors: Boolean)

  /**
   * A more convenient access to the embeddings values of the virtual root.
   */
  private val virtualRoot: DenseNDArray get() = this.parser.model.rootEmbedding.array.values

  /**
   * The builder of the [ContextEncoder].
   */
  private val contextEncoderBuilder = ContextEncoderBuilder(this.parser.model.contextEncoderModel)

  /**
   * The builder of the [HeadsEncoder].
   */
  private val headsEncoderBuilder = HeadsEncoderBuilder(this.parser.model.headsEncoderModel)

  /**
   * The builder of the tokens encoder.
   */
  private val tokensEncoderBuilder: TokensEncoderBuilder = TokensEncoderFactory(
    this.parser.model.tokensEncoderModel, trainingMode = true)

  /**
   * The builder of the labeler.
   */
  private val deprelLabelerBuilder: DeprelLabelerBuilder? = this.parser.model.labelerModel?.let {
    DeprelLabelerBuilder(model = it, rootVector = this.virtualRoot)
  }

  /**
   * The optimizer of the latent heads encoder.
   */
  private val headsEncoderOptimizer = HeadsEncoderOptimizer(
    model = this.parser.model.headsEncoderModel, updateMethod = this.updateMethod)

  /**
   * The optimizer of the context encoder.
   */
  private val contextEncoderOptimizer = ContextEncoderOptimizer(
    model = this.parser.model.contextEncoderModel, updateMethod = this.updateMethod)

  /**
   * The optimizer of the labeler (can be null).
   */
  private val deprelLabelerOptimizer: DeprelLabelerOptimizer? = this.parser.model.labelerModel?.let {
    DeprelLabelerOptimizer(model = this.parser.model.labelerModel, updateMethod = this.updateMethod)
  }

  /**
   * The optimizer of the tokens encoder.
   */
  private val tokensEncoderOptimizer = TokensEncoderOptimizerFactory(
    model = this.parser.model.tokensEncoderModel, updateMethod = this.updateMethod)

  /**
   * The pointer network.
   */
  private val pointerNetwork = PointerNetworkProcessor(this.parser.model.pointerNetworkModel)

  /**
   *
   */
  private val dependentsEncoder = DependentsEncoderBuilder(this.parser.model.dependentsEncoderModel).invoke()

  /**
   *
   */
  private val dependentsEncoderOptimizer =
    DependentsEncoderOptimizer(this.parser.model.dependentsEncoderModel, updateMethod = this.updateMethod)

  /**
   * The heads pointer optimizer.
   * TODO: fix usage
   */
  private val headsPointerOptimizer = ParamsOptimizer(
    params = this.parser.model.pointerNetworkModel.params,
    updateMethod = updateMethod)

  /**
   * Group the optimizers all together.
   */
  private val optimizers = listOf(
    this.headsEncoderOptimizer,
    this.contextEncoderOptimizer,
    this.deprelLabelerOptimizer,
    this.tokensEncoderOptimizer,
    this.headsPointerOptimizer,
    this.dependentsEncoderOptimizer)

  /**
   * @return a string representation of the configuration of this Trainer
   */
  override fun toString(): String = """
    %-33s : %s
    %-33s : %s
    %-33s : %s
  """.trimIndent().format(
    "Epochs", this.epochs,
    "Batch size", this.batchSize,
    "Skip punctuation errors", this.lhrErrorsOptions.skipPunctuationErrors
  )

  /**
   * Beat the occurrence of a new batch.
   */
  override fun newBatch() {

    if (this.updateMethod is BatchScheduling) {
      this.updateMethod.newBatch()
    }
  }

  /**
   * Beat the occurrence of a new epoch.
   */
  override fun newEpoch() {

    if (this.updateMethod is EpochScheduling) {
      this.updateMethod.newEpoch()
    }
  }

  /**
   * Update the model parameters.
   */
  override fun update() {
    this.optimizers.forEach { it?.update() }
  }

  /**
   * @return the count of the relevant errors
   */
  override fun getRelevantErrorsCount(): Int = 1

  /**
   * Method to call before learning a new sentence.
   */
  private fun beforeSentenceLearning() {

    if (this.updateMethod is ExampleScheduling) {
      this.updateMethod.newExample()
    }
  }

  /**
   * Method to call after learning a sentence.
   */
  private fun afterSentenceLearning() {
    // TODO("Delete it?")
  }

  /**
   * Train the Transition System with the given [sentence].
   *
   * @param sentence a sentence
   * @param goldPOSSentence an optional sentence with gold annotated POS in its dependency tree
   */
  override fun trainSentence(sentence: Sentence, goldPOSSentence: Sentence?) {

    val goldTree: DependencyTree = checkNotNull(sentence.dependencyTree) {
      "The gold dependency tree of a sentence cannot be null during the training."
    }

    val goldPOSTree: DependencyTree? = goldPOSSentence?.let { checkNotNull(it.dependencyTree) {
      "The gold dependency tree of a gold POS sentence cannot be null during the training."
    } }

    this.beforeSentenceLearning()

    this.learn(
      sentence = sentence,
      goldTree = goldTree,
      goldPosTags = goldPOSTree?.posTags)

    this.afterSentenceLearning()
  }

  /**
   * Learn the latent syntactic structure of a sentence.
   *
   * @param sentence a sentence
   * @param goldTree the gold dependency tree of the [sentence]
   * @param goldPosTags optional gold POS tags to use instead of the ones of the [goldTree] (default = null)
   *
   * @return the loss
   */
  private fun learn(sentence: Sentence,
                    goldTree: DependencyTree,
                    goldPosTags: Array<POSTag?>? = null) {

    val encoder: LSSEncoder = this.buildEncoder()
    val lss = encoder.encode(sentence.tokens)

    val errors = MSECalculator().calculateErrors(
      outputSequence = lss.latentHeads,
      outputGoldSequence = this.getExpectedLatentHeads(lss, goldTree.heads))

    val labeler: DeprelLabeler? = this.deprelLabelerBuilder?.invoke()

    labeler?.predict( // important to calculate the right errors
      tokens = sentence.tokens,
      tokensHeads = goldTree.heads,
      tokensVectors = lss.contextVectors)

    val headsPointer: HeadsPointer? = null // HeadsPointer(this.pointerNetwork) // TODO: fix

    // TODO: to refactor
    headsPointer?.let {
      it.learn(lss, goldTree.heads)
      this.headsPointerOptimizer.accumulate(it.getParamsErrors(copy = false))
    }

    //this.dependentsEncoder.learn(lss.contextVectors, goldTree)
    //this.dependentsEncoderOptimizer.accumulate(this.dependentsEncoder.getParamsErrors())

    this.propagateErrors(
      errors = errors,
      goldTree = goldTree,
      goldPosTags = goldPosTags ?: goldTree.posTags,
      encoder = encoder,
      headsPointer = headsPointer,
      labeler = labeler)
  }


  /**
   * @param scores the attachment scores
   *
   * @return a dependency tree
   */
  private fun buildDependencyTree(scores: ArcScores): DependencyTree {
    val dependencyTree = DependencyTree(scores.size)
    this.parser.assignHeads(dependencyTree, scores)
    return dependencyTree
  }

  /**
   * Return a list containing the expected latent heads, one for each token of the sentence.
   *
   * @param lss the latent syntactic structure
   * @param goldHeads the gold heads ids
   *
   * @return the expected latent heads
   */
  private fun getExpectedLatentHeads(lss: LatentSyntacticStructure,
                                     goldHeads: Array<Int?>): List<DenseNDArray> {

    return lss.tokens.zip(goldHeads).map { (token, goldHeadId) ->

      when {

        goldHeadId == null -> lss.virtualRoot

        this.lhrErrorsOptions.skipPunctuationErrors && token.isPunctuation ->
          lss.latentHeads[token.id] // this means no errors

        else ->
          lss.contextVectors[goldHeadId]
      }
    }
  }

  /**
   * @return a LSSEncoder
   */
  private fun buildEncoder() = LSSEncoder(
    tokensEncoder = this.tokensEncoderBuilder.invoke(),
    contextEncoder = this.contextEncoderBuilder.invoke(),
    headsEncoder = this.headsEncoderBuilder.invoke(),
    virtualRoot = this.virtualRoot)

  /**
   * Propagate the errors through the encoders.
   *
   * @param errors the latent heads errors
   * @param goldTree the gold dependency tree
   * @param goldPosTags the gold pos-tags
   * @param encoder the encoder of the latent syntactic structure
   * @param headsPointer the heads pointer
   * @param labeler the labeler
   */
  private fun propagateErrors(
    errors: List<DenseNDArray>,
    goldTree: DependencyTree,
    goldPosTags: Array<POSTag?>?,
    encoder: LSSEncoder,
    headsPointer: HeadsPointer?,
    labeler: DeprelLabeler?){

    headsPointer?.let {
      errors.assignSum(headsPointer.getLatentHeadsErrors()) // // TODO: to refactor
    }

    val contextErrors = encoder.headsEncoder.propagateErrors(errors)

    headsPointer?.let {
      contextErrors.assignSum(headsPointer.getContextVectorsErrors()) // TODO: to refactor
    }

    //contextErrors.assignSum(this.dependentsEncoder.getInputErrors().toTypedArray())

    labeler?.propagateErrors(goldTree)?.let { labelerInputErrors ->

      val (extContextErrors: List<DenseNDArray>, rootErrors: DenseNDArray) = labelerInputErrors

      contextErrors.assignSum(extContextErrors)

      this.propagateRootErrors(rootErrors)
    }

    encoder.tokensEncoder.propagateErrors(encoder.contextEncoder.propagateErrors(contextErrors))
  }

  /**
   * Propagate the [outputErrors] through the heads encoder, accumulates the resulting parameters errors in the
   * [headsEncoderOptimizer] and returns the input errors.
   *
   * @param outputErrors the output errors
   *
   * @return the input errors
   */
  private fun HeadsEncoder.propagateErrors(outputErrors: List<DenseNDArray>): List<DenseNDArray> {

    this.backward(outputErrors)
    this@LHRTrainer.headsEncoderOptimizer.accumulate(this.getParamsErrors(copy = false))
    return this.getInputErrors(copy = false)
  }

  /**
   * Propagate the [outputErrors] through the context encoder, accumulates the resulting parameters errors in the
   * [contextEncoderOptimizer] and returns the input errors.
   *
   * @param outputErrors the output errors
   *
   * @return the input errors
   */
  private fun ContextEncoder.propagateErrors(outputErrors: List<DenseNDArray>): List<DenseNDArray> {

    this.backward(outputErrors)
    this@LHRTrainer.contextEncoderOptimizer.accumulate(this.getParamsErrors(copy = false))
    return this.getInputErrors(copy = false)
  }

  /**
   * Calculate the labeler errors respect to the [goldTree], accumulates the resulting parameters
   * errors in the [deprelLabelerOptimizer] and returns the input errors.
   *
   * @param goldTree the gold dependency tree
   */
  private fun DeprelLabeler.propagateErrors(goldTree: DependencyTree): Pair<List<DenseNDArray>, DenseNDArray> {

    this.backward(goldDeprels = goldTree.deprels)
    this@LHRTrainer.deprelLabelerOptimizer!!.accumulate(this.getParamsErrors(copy = false))
    return this.getInputErrors()
  }

  /**
   * Propagate the [outputErrors] through the tokens encoder, accumulates the resulting parameters errors in the
   * [tokensEncoderOptimizer] and returns the input errors.
   *
   * @param outputErrors the output errors
   */
  private fun TokensEncoder.propagateErrors(outputErrors: List<DenseNDArray>) {

    this.backward(outputErrors)
    this@LHRTrainer.tokensEncoderOptimizer.accumulate(this.getParamsErrors(copy = false))
  }

  /**
   * Propagate the [errors] through the virtual root embedding.
   *
   * @param errors the errors
   */
  private fun propagateRootErrors(errors: DenseNDArray) {
    this.parser.model.rootEmbedding.let { this.updateMethod.update(array = it.array, errors = errors) }
  }
}