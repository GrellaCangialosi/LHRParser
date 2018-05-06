/* Copyright 2018-present LHRParser Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * -----------------------------------------------------------------------------*/

package utils

import com.xenomachina.argparser.ArgParser
import com.xenomachina.argparser.InvalidArgumentException
import com.xenomachina.argparser.default

/**
 * The interpreter of command line arguments for the training script.
 *
 * @param args the array of command line arguments
 */
class TrainingArgs(args: Array<String>) {

  /**
   * The type of tokens encoding.
   *
   * TODO: AMBIGUOUS_POS
   */
  enum class TokensEncodingType {
    WORD_AND_POS_EMBEDDINGS, WORD_AND_EXT_AND_POS_EMBEDDINGS, MORPHO_FEATURES
  }

  /**
   * The parser of the string arguments.
   */
  private val parser = ArgParser(args)

  /**
   * The language code
   */
  val langCode: String by parser.storing(
    "-l",
    "--language",
    help="the language code"
  )

  /**
   * The number of training epochs (default = 30).
   */
  val epochs: Int by parser.storing(
    "-e",
    "--epochs",
    help="the number of training epochs (default = 30)"
  ) { toInt() }.default(30)

  /**
   * The size of the batches of sentences (default = 1).
   */
  val batchSize: Int by parser.storing(
    "-b",
    "--batch-size",
    help="the size of the batches of sentences (default = 1)"
  ) { toInt() }.default(1)

  /**
   * The maximum number of sentences to load for training (default unlimited)
   */
  val maxSentences: Int? by parser.storing(
    "-s",
    "--max-sentences",
    help="the maximum number of sentences to load for training (default unlimited)"
  ) { toInt() }.default { null }

  /**
   * The file path of the training set.
   */
  val trainingSetPath: String by parser.storing(
    "-t",
    "--training-set",
    help="the file path of the training set"
  )

  /**
   * The file path of the gold-POS training set.
   */
  val goldPosSetPath: String? by parser.storing(
    "-p",
    "--pos-set",
    help="the file path of the gold-POS training set"
  ).default { null }

  /**
   * The file path of the validation set.
   */
  val validationSetPath: String by parser.storing(
    "-v",
    "--validation-set",
    help="the file path of the validation set"
  )

  /**
   * The path of the file in which to save the serialized model.
   */
  val modelPath: String by parser.storing(
    "-m",
    "--model-path",
    help="the path of the file in which to save the serialized model"
  )

  /**
   * The file path of the pre-trained word embeddings.
   */
  val embeddingsPath: String? by parser.storing(
    "-w",
    "--trained-word-emb-path",
    help="the file path of the pre-trained word embeddings"
  ).default { null }

  /**
   * The number of stacked BiRNNs of the context encoder (default 2).
   */
  val numOfContextLayers: Int by parser.storing(
    "-c",
    "--context-layers",
    help="the number of stacked BiRNNs of the context encoder (default 2)"
  ){ toInt() }
    .default(2)
    .addValidator { if (value < 1) throw InvalidArgumentException( "The number of context-layers must >= 1") }

  /**
   * The size of the word embedding vectors.
   */
  val wordEmbeddingSize: Int by parser.storing(
    "--word-emb-size",
    help="the size of the word embedding vectors (default 150)"
  ){ toInt() }.default(150)

  /**
   * The word embeddings dropout coefficient.
   */
  val wordDropoutCoefficient: Double by parser.storing(
    "--word-dropout",
    help="the word embeddings dropout coefficient (default 0.25)"
  ){ toDouble() }.default(0.25)

  /**
   * The size of the part-of-speech embedding vectors.
   */
  val posEmbeddingSize: Int by parser.storing(
    "--pos-emb-size",
    help="the size of the part-of-speech embedding vectors (default 50)"
  ){ toInt() }.default(50)

  /**
   * The part-of-speech embeddings dropout coefficient.
   */
  val posDropoutCoefficient: Double by parser.storing(
    "--pos-dropout",
    help="the part-of-speech embeddings dropout coefficient (default 0.0)"
  ){ toDouble() }.default(0.0)

  /**
   * Whether to skip non-projective sentences.
   */
  val skipNonProjective: Boolean by parser.flagging(
    "--skip-non-projective",
    help="whether to skip non-projective sentences"
  )

  /**
   * Whether to do not consider punctuation errors.
   */
  val skipPunctuationErrors: Boolean by parser.flagging(
    "--skip-punct-err",
    help="whether to do not consider punctuation errors"
  )

  /**
   * Whether to use the labeler.
   */
  val useLabeler: Boolean by parser.flagging(
    "--labeler",
    help="whether to use the labeler"
  )

  /**
   * Whether to do not predict the POS tags.
   */
  val noPosPrediction: Boolean by parser.flagging(
    "--no-pos",
    help="whether to do not predict the POS tags"
  )

  /**
   * Whether to propagate only if errors are relevant.
   */
  val onlyRelevantErrors: Boolean by parser.flagging(
    "--only-relevant-err",
    help="whether to propagate only if errors are relevant"
  )

  /**
   * Whether to propagate only if errors are relevant.
   */
  val relaxReconstructionErrors: Boolean by parser.flagging(
    "--relax-errors",
    help="whether to ignore the latent heads errors if the attachment is already correct"
  )

  /**
   * The file path of the serialized morphology dictionary.
   */
  val morphoDictionaryPath: String? by parser.storing(
    "-d",
    "--dictionary",
    help="the file path of the serialized morphology dictionary"
  ).default { null }

  /**
   * The file path of the lexicon dictionary.
   */
  val lexiconDictionaryPath: String? by parser.storing(
    "-x",
    "--lexicon",
    help="the file path of the lexicon dictionary"
  ).default { null }

  /**
   * The type of morphology encoding.
   */
  val tokensEncodingType: TokensEncodingType by parser.mapping(
    "--tokens-word-pos-emb" to TokensEncodingType.WORD_AND_POS_EMBEDDINGS,
    "--tokens-word-ext-pos-emb" to TokensEncodingType.WORD_AND_EXT_AND_POS_EMBEDDINGS,
    "--tokens-morpho" to TokensEncodingType.MORPHO_FEATURES,
    help = "the type of morphology encoding (default --tokens-word-pos-emb)"
  ).default { TokensEncodingType.WORD_AND_POS_EMBEDDINGS }

  /**
   * Whether to do not show details about the training.
   */
  val quiet: Boolean by parser.flagging(
    "-q",
    "--quiet",
    help="whether to do not show details about the training "
  )

  /**
   * Force parsing all arguments (only read ones are parsed by default).
   */
  init {
    parser.force()
  }
}
