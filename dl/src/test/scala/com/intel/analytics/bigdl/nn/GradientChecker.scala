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

package com.intel.analytics.bigdl.nn

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

class GradientChecker(stepSize: Double, threshold: Double) {

  def checkLayer[T: ClassTag](
    layer: Module[Tensor[T], Tensor[T], T],
    input: Tensor[T],
    epsilon: Double = 0.001,
    num: Int = -1)
    (implicit ev: TensorNumeric[T]): Boolean = {
    val gradOutput = lossAndGradient(layer.updateOutput(input))._2
    val computedGrad = layer.updateGradInput(input, gradOutput)
    computedGrad.resize(Array(computedGrad.nElement()))

    val perturbation = Tensor[T]()
    perturbation.set(input)
    perturbation.resize(input.nElement())
    var result = true
    var i = 1
    while (i <= 10) { //input.nElement()) { //ev.min(100,input.nElement()) {
      val curValue = perturbation.valueAt(i)
      perturbation.setValue(i, ev.fromType(ev.toType[Double](curValue) + stepSize))
      val positiveLoss = lossAndGradient(layer.updateOutput(input))._1
      perturbation.setValue(i, ev.fromType(ev.toType[Double](curValue) - stepSize))
      val negativeLoss = lossAndGradient(layer.updateOutput(input))._1
      val estimatedGradient = (positiveLoss - negativeLoss) / stepSize / 2.0

      val tmp = math.abs(estimatedGradient - ev.toType[Double](computedGrad.valueAt(i)))
      result = result & (math.abs(estimatedGradient -
        ev.toType[Double](computedGrad.valueAt(i))) < epsilon)
      perturbation.setValue(i, curValue)
      i += 1
    }

    result
  }

  def checkWeight[T: ClassTag](
    layer: Module[Tensor[T], Tensor[T], T],
    input: Tensor[T],
    epsilon: Double = 0.001,
    num: Int = -1)
  (implicit ev: TensorNumeric[T]): Boolean = {
    val gradOutput = lossAndGradient(layer.updateOutput(input))._2
    val computedGrad = layer.updateGradInput(input, gradOutput)
    computedGrad.resize(Array(computedGrad.nElement()))

    val weights = layer.getParameters()._1

    val perturbation = Tensor[T]()
    perturbation.set(weights)
    perturbation.resize(weights.nElement())
    var result = true
    var i = 1
    println(weights.nElement())
    while (i <= 100){ //weights.nElement()) { //input.nElement()) { //ev.min(100,input.nElement()) {
    val curValue = perturbation.valueAt(i)
      perturbation.setValue(i, ev.fromType(ev.toType[Double](curValue) + stepSize))
      val positiveLoss = lossAndGradient(layer.updateOutput(input))._1
      perturbation.setValue(i, ev.fromType(ev.toType[Double](curValue) - stepSize))
      val negativeLoss = lossAndGradient(layer.updateOutput(input))._1
      val estimatedGradient = (positiveLoss - negativeLoss) / stepSize / 2.0

      result = result & (math.abs(estimatedGradient -
        ev.toType[Double](computedGrad.valueAt(i))) < epsilon)
      perturbation.setValue(i, curValue)
      i += 1
    }

    result
  }

  def lossAndGradient[T: ClassTag](output: Tensor[T])(
    implicit ev: TensorNumeric[T]): (Double, Tensor[T]) = {
    val gradOutput = Tensor[T]().resizeAs(output).copy(output)
    var loss = 0.0
    gradOutput.apply1(a => {
      val aDouble = ev.toType[Double](a)
      loss += 0.5 * aDouble * aDouble
      a
    })
    (loss, gradOutput)
  }
}
