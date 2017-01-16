/*
 * Licensed to Intel Corporation under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * Intel Corporation licenses this file to You under the Apache License, Version 2.0
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

package com.intel.analytics.bigdl.models

import com.intel.analytics.bigdl.nn.GradientChecker
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.RandomGenerator._
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

import scala.util.Random

class ModelGraientCheckSpec extends FlatSpec with BeforeAndAfter with Matchers {

  private val checkModel = true

  "GoogleNet_v1 model in batch mode" should "be good in gradient check for input" in {
    val seed = 100
    RNG.setSeed(seed)
    val start = System.nanoTime()
    val input = Tensor[Double](4, 3, 224, 224).apply1(e => Random.nextDouble())
    val model = GoogleNet_v1(1000)
    model.zeroGradParameters()

    val checker = new GradientChecker(1e-4).setType(checkModel)
    checker.checkLayer(model, input, 1e-2) should be(true)
    val scalaTime = System.nanoTime() - start
    println("Test Scala time : " + scalaTime / 1e9 + " s")
  }

  "GoogleNet_v1 model in batch mode" should "be good in gradient check for weight" in {
    val seed = 100
    RNG.setSeed(seed)
    val start = System.nanoTime()
    val input = Tensor[Double](4, 3, 224, 224).apply1(e => Random.nextDouble())
    val model = GoogleNet_v1(1000)
    model.zeroGradParameters()

    val checker = new GradientChecker(1e-4).setType(checkModel)
    checker.checkWeight(model, input, 1e-2) should be(true)
    val scalaTime = System.nanoTime() - start
    println("Test Scala time : " + scalaTime / 1e9 + " s")
  }

  "GoogleNet_v2 model in batch mode" should "be good in gradient check for input" in {
    val seed = 100
    RNG.setSeed(seed)
    val start = System.nanoTime()
    val input = Tensor[Double](4, 3, 224, 224).apply1(e => Random.nextDouble())
    val model = GoogleNet_v2(1000)
    model.zeroGradParameters()

    val checker = new GradientChecker(1e-4).setType(checkModel)
    checker.checkLayer(model, input, 1e-2) should be(true)
    val scalaTime = System.nanoTime() - start
    println("Test Scala time : " + scalaTime / 1e9 + " s")
  }

  "GoogleNet_v2 model in batch mode" should "be good in gradient check for weight" in {
    val seed = 100
    RNG.setSeed(seed)
    val start = System.nanoTime()
    val input = Tensor[Double](4, 3, 224, 224).apply1(e => Random.nextDouble())
    val model = GoogleNet_v2.applyNoBn(1000)
    model.zeroGradParameters()

    val checker = new GradientChecker(1e-4).setType(checkModel)
    checker.checkWeight(model, input, 1e-2) should be(true)
    val scalaTime = System.nanoTime() - start
    println("Test Scala time : " + scalaTime / 1e9 + " s")
  }

  "AlexNet-OWT model in batch mode" should "be good in gradient check for input" in {
    val seed = 100
    RNG.setSeed(seed)
    val start = System.nanoTime()
    val input = Tensor[Double](8, 3, 224, 224).apply1(e => Random.nextDouble())
    val model = AlexNet_OWT(1000, false, true)
    model.zeroGradParameters()

    val checker = new GradientChecker(1e-4).setType(checkModel)
    checker.checkLayer[Double](model, input, 1e-2) should be(true)
    val scalaTime = System.nanoTime() - start
    println("Test Scala time : " + scalaTime / 1e9 + " s")
  }

  "AlexNet-OWT model in batch mode" should "be good in gradient check for weight" in {
    val seed = 100
    RNG.setSeed(seed)
    val start = System.nanoTime()
    val input = Tensor[Double](8, 3, 224, 224).apply1(e => Random.nextDouble())
    val model = AlexNet_OWT(1000, false, true)
    model.zeroGradParameters()

    val checker = new GradientChecker(1e-4).setType(checkModel)
    checker.checkWeight[Double](model, input, 1e-2) should be(true)
    val scalaTime = System.nanoTime() - start
    println("Test Scala time : " + scalaTime / 1e9 + " s")
  }

  "AlexNet model in batch mode" should "be good in graident check for input" in {
    val seed = 100
    RNG.setSeed(seed)
    val start = System.nanoTime()
    val input = Tensor[Double](8, 3, 227, 227).apply1(e => Random.nextDouble())
    val model = AlexNet(1000)
    model.zeroGradParameters()

    val checker = new GradientChecker(1e-4).setType(checkModel)
    checker.checkLayer[Double](model, input, 1e-2) should be(true)
    val scalaTime = System.nanoTime() - start
    println("Test Scala time : " + scalaTime / 1e9 + " s")
  }

  "AlexNet model in batch mode" should "be good in graident check for weight" in {
    val seed = 100
    RNG.setSeed(seed)
    val start = System.nanoTime()
    val input = Tensor[Double](8, 3, 227, 227).apply1(e => Random.nextDouble())
    val model = AlexNet(1000)
    model.zeroGradParameters()

    val checker = new GradientChecker(1e-4).setType(checkModel)
    checker.checkWeight[Double](model, input, 1e-2) should be(true)
    val scalaTime = System.nanoTime() - start
    println("Test Scala time : " + scalaTime / 1e9 + " s")
  }

  "VggLike model in batch mode" should "be good in gradient check for input" in {
    val seed = 100
    RNG.setSeed(seed)
    val start = System.nanoTime()
    val input = Tensor[Double](8, 3, 32, 32).apply1(e => Random.nextDouble())
    val model = VggLike(10)
    model.zeroGradParameters()

    val checker = new GradientChecker(1e-4).setType(checkModel)
    checker.checkLayer[Double](model, input, 1e-2) should be(true)
    val scalaTime = System.nanoTime() - start
    println("Test Scala time : " + scalaTime / 1e9 + " s")
  }

  "VggLike model in batch mode" should "be good in gradient check for weight" in {
    val seed = 100
    RNG.setSeed(seed)
    val start = System.nanoTime()
    val input = Tensor[Double](8, 3, 32, 32).apply1(e => Random.nextDouble())
    val model = VggLike(10)
    model.zeroGradParameters()

    val checker = new GradientChecker(1e-4).setType(checkModel)
    checker.checkWeight[Double](model, input, 1e-2) should be(true)
    val scalaTime = System.nanoTime() - start
    println("Test Scala time : " + scalaTime / 1e9 + " s")
  }

  "LeNet model in batch mode" should "be good in gradient check for input" in {
    val seed = 100
    RNG.setSeed(seed)
    val start = System.nanoTime()
    val input = Tensor[Double](8, 1, 28, 28).apply1(e => Random.nextDouble())
    val model = LeNet5(10)
    model.zeroGradParameters()

    val checker = new GradientChecker(1e-4).setType(checkModel)
    checker.checkLayer[Double](model, input, 1e-2) should be(true)
    val scalaTime = System.nanoTime() - start
    println("Test Scala time : " + scalaTime / 1e9 + " s")
  }

  "LeNet model in batch mode" should "be good in gradient check for weight" in {
    val seed = 100
    RNG.setSeed(seed)
    val start = System.nanoTime()
    val input = Tensor[Double](8, 1, 28, 28).apply1(e => Random.nextDouble())
    val model = LeNet5(10)
    model.zeroGradParameters()

    val checker = new GradientChecker(1e-4).setType(checkModel)
    checker.checkWeight(model, input, 1e-2) should be(true)
    val scalaTime = System.nanoTime() - start
    println("Test Scala time : " + scalaTime / 1e9 + " s")
  }

  "CNN model in batch mode" should "be good in gradient check for input" in {
    val seed = 100
    RNG.setSeed(seed)
    val start = System.nanoTime()
    val input = Tensor[Double](8, 1, 28, 28).apply1(e => Random.nextDouble())
    val model = SimpleCNN(10)
    model.zeroGradParameters()

    val checker = new GradientChecker(1e-4).setType(checkModel)
    checker.checkLayer[Double](model, input, 1e-2) should be(true)
    val scalaTime = System.nanoTime() - start
    println("Test Scala time : " + scalaTime / 1e9 + " s")
  }

  "CNN model in batch mode" should "be good in gradient check for weight" in {
    val seed = 100
    RNG.setSeed(seed)
    val start = System.nanoTime()
    val input = Tensor[Double](8, 1, 28, 28).apply1(e => Random.nextDouble())
    val model = SimpleCNN(10)
    model.zeroGradParameters()

    val checker = new GradientChecker(1e-4).setType(checkModel)
    checker.checkWeight(model, input, 1e-2) should be(true)
    val scalaTime = System.nanoTime() - start
    println("Test Scala time : " + scalaTime / 1e9 + " s")
  }
}
