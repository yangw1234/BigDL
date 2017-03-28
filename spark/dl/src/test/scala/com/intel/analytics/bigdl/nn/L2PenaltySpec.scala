/*
 * Copyright 2016 The BigDL Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.intel.analytics.bigdl.nn

import com.intel.analytics.bigdl.tensor.Tensor
import org.scalatest.{FlatSpec, Matchers}

import scala.util.Random


@com.intel.analytics.bigdl.tags.Parallel
class L2PenaltySpec extends FlatSpec with Matchers {

  "A L2Penalty" should "generate correct output and grad" in {
    val weight = 1
    val l2 = L2Penalty[Double](weight)

    val input = Tensor[Double](2, 7).apply1(_ => Random.nextDouble())

    val output = l2.forward(input)
    val loss: Double = l2.loss
    val gradOutput = Tensor[Double](input.size(): _*)
    val grad = l2.backward(input, gradOutput)

    val l2Norm2 = input.clone().apply1(x => x*x).sum()
    val trueGrad = input.clone().apply1(_ * weight * 2)

    output should be (input)
    loss should be (l2Norm2 +- 1e-9)
    grad should be (trueGrad)

  }
}
