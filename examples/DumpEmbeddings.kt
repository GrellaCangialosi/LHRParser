/* Copyright 2018-present LHRParser Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * -----------------------------------------------------------------------------*/

import com.grellacangialosi.lhrparser.LHRModel
import com.kotlinnlp.tokensencoder.embeddings.dictionary.EmbeddingsEncoderByDictionaryModel
import com.kotlinnlp.tokensencoder.ensamble.concat.ConcatTokensEncoderModel
import com.xenomachina.argparser.mainBody
import java.io.File
import java.io.FileInputStream

/**
 * First argument: model filename
 * Second argument: embeddings output filename
 */
fun main(args: Array<String>) = mainBody {

  println("Loading model from '${args[0]}'.")
  val model = LHRModel.load(FileInputStream(File(args[0])))

  println("\n-- MODEL")
  println(model)
  println()

  require (model.tokensEncoderModel is ConcatTokensEncoderModel)

  (model.tokensEncoderModel as ConcatTokensEncoderModel).models
    .filter { it is EmbeddingsEncoderByDictionaryModel }
    .forEachIndexed { i, m ->

      val out = "${args[1]}.$i"

      println("Dump embeddings $out...")

      (m as EmbeddingsEncoderByDictionaryModel).embeddingsMap.dump(filename = out, digits = 6)
    }

  println("Done.")
}
