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
package com.intel.analytics.sparkdl.nn

import com.intel.analytics.sparkdl.tensor.Tensor
import com.intel.analytics.sparkdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.sparkdl.utils.T

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

/**
 * a weighted sum of other criterions each applied to the same input and target;
 */
class MultiCriterion[T: ClassTag](implicit ev: TensorNumeric[T]) extends TensorCriterion[T] {

  var gradInput = Tensor[T]()
  @transient
  private var weights = new ArrayBuffer[Double]
  private val criterions = T()

  def add(criterion: TensorCriterion[T], weight: Double = 1): Unit = {
    criterions.insert(criterions.length() + 1, criterion)
    weights.append(weight)
  }
  override def updateOutput(input: Tensor[T], target: Tensor[T]): T = {
    var i = 1
    while (i <= criterions.length) {
      val tmp1 = criterions[TensorCriterion[T]](i).updateOutput(input, target)
      output = ev.plus(output, ev.times(ev.fromType(weights(i-1)), tmp1))
      i +=1
    }
    output
  }

  override def updateGradInput(input: Tensor[T], target: Tensor[T]): Tensor[T] = {
    gradInput = Utils.recursiveResizeAs[T](gradInput, input).toTensor()
    Utils.recursiveFill[T](gradInput, 0)
    var i = 1
    while (i <= criterions.length) {
      Utils.recursiveAdd(gradInput, weights(i - 1),
        criterions[TensorCriterion[T]](i).updateGradInput(input, target))
      i += 1
    }
    gradInput
  }

  override def toString(): String = {
    s"nn.MultiCriterion"
  }
}
