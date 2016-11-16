/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.intel.analytics.sparkdl.torch

import com.intel.analytics.sparkdl.nn.Mul
import com.intel.analytics.sparkdl.tensor.Tensor
import com.intel.analytics.sparkdl.utils.RandomGenerator._
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

import scala.util.Random

class MulSpec extends FlatSpec with BeforeAndAfter with Matchers{

  before {
    if (!TH.hasTorch()) {
      cancel("Torch is not installed")
    }
  }

  "A Mul Module " should "generate correct output and grad" in {
    val inputN = 5
    val seed = 100
    RNG.setSeed(seed)
    val module = new Mul[Double]()
    val input = Tensor[Double](1, 5).apply1(e => Random.nextDouble())
    val gradOutput = Tensor[Double](5).apply1(e => Random.nextDouble())

    val code = "torch.manualSeed(" + seed + ")\n" +
      "module = nn.Mul()\n" +
      "module:reset()\n" +
      "gradWeight = module.gradWeight\n" +
      "output = module:forward(input)\n" +
      "gradInput = module:backward(input, gradOutput)\n"

    val (luaTime, torchResult) = TH.run(code, Map("input" -> input, "gradOutput" -> gradOutput),
      Array("output", "gradInput", "gradWeight"))

    val luaOutput1 = torchResult("output").asInstanceOf[Tensor[Double]]
    val luaOutput2 = torchResult("gradInput").asInstanceOf[Tensor[Double]]
    val luaWeight = torchResult("gradWeight").asInstanceOf[Tensor[Double]]

    val start = System.nanoTime()
    module.reset()
    val output = module.forward(input)
    val gradInput = module.backward(input, gradOutput)
    val end = System.nanoTime()
    val scalaTime = end - start

    luaOutput1 should be(output)
    luaOutput2 should be(gradInput)

    println("Test case : Mul, Torch : " + luaTime + " s, Scala : " + scalaTime / 1e9 + " s")
  }
}
