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

import com.intel.analytics.sparkdl.nn.NarrowTable
import com.intel.analytics.sparkdl.tensor.Tensor
import com.intel.analytics.sparkdl.utils.RandomGenerator._
import com.intel.analytics.sparkdl.utils.T
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

import scala.collection.mutable.HashMap
import scala.util.Random

class NarrowTableSpec extends FlatSpec with BeforeAndAfter with Matchers{
  before {
    if (!TH.hasTorch()) {
      cancel("Torch is not installed")
    }
  }

  "A NarrowTable Module " should "generate correct output and grad" in {
    val seed = 100
    RNG.setSeed(seed)
    val module = new NarrowTable[Double](1)

    val input = T()
    input(1.toDouble) = Tensor[Double](2, 3).apply1(e => Random.nextDouble())
    input(2.toDouble) = Tensor[Double](2, 1).apply1(e => Random.nextDouble())
    input(3.toDouble) = Tensor[Double](2, 2).apply1(e => Random.nextDouble())

    val gradOutput = T()
    gradOutput(1.toDouble) = Tensor[Double](5, 3).apply1(e => Random.nextDouble())
    gradOutput(2.toDouble) = Tensor[Double](2, 5).apply1(e => Random.nextDouble())

    val code = "torch.manualSeed(" + seed + ")\n" +
      "module = nn.NarrowTable(1)\n" +
      "output = module:forward(input)\n" +
      "gradInput = module:backward(input, gradOutput)\n"

    val (luaTime, torchResult) = TH.run(code, Map("input" -> input, "gradOutput" -> gradOutput),
      Array("output", "gradInput"))

    val luaOutput1 = torchResult("output").asInstanceOf[HashMap[Double, Tensor[Double]]]
    val luaOutput2 = torchResult("gradInput").asInstanceOf[HashMap[Double, Tensor[Double]]]

    val start = System.nanoTime()
    val output = module.forward(input)
    val gradInput = module.backward(input, gradOutput)
    val end = System.nanoTime()
    val scalaTime = end - start

    val outputData = output.getState()
    val gradInputData = gradInput.getState()

    luaOutput1.size should be(outputData.size)
    luaOutput2.size should be(gradInputData.size)

    val tmp = output.length()

    var i = 1
    while (i <= luaOutput1.size) {
      val val1 = luaOutput1.get(i.toDouble).getOrElse(null)
      val val2 = outputData.get(i.toDouble).getOrElse(null)
      val1 should be(val2)
      i += 1
    }

    i = 1
    while (i <= luaOutput2.size) {
      val val1 = luaOutput2.get(i.toDouble).getOrElse(null)
      val val2 = gradInputData.get(i.toDouble).getOrElse(null)
      val1 should be(val2)
      i += 1
    }

    println("Test case : NarrowTable, Torch : " + luaTime +
      " s, Scala : " + scalaTime / 1e9 + " s")
  }
}
