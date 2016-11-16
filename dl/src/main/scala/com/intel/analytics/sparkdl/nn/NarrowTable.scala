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

import com.intel.analytics.sparkdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.sparkdl.utils.Table
import com.intel.analytics.sparkdl.tensor.Tensor

import scala.reflect.ClassTag

/**
 * Creates a module that takes a table as input and outputs the subtable starting at index
 * offset having length elements (defaults to 1 element). The elements can be either
 * a table or a Tensor.
 * @param offset
 * @param length
 */
class NarrowTable[T: ClassTag](offset: Int, length: Int = 1)
 (implicit ev: TensorNumeric[T]) extends Module[Table, Table, T]{

  override def updateOutput(input: Table): Table = {
    var i = 1
    while (i <= length) {
      output.insert(i, input(offset + i -1))
      i += 1
    }
    output
  }

  override def updateGradInput(input: Table, gradOutput: Table): Table = {
    var i = 1
    while (i <= gradOutput.length()) {
      gradInput.insert(offset + i - 1, gradOutput(i))
      i += 1
    }

    i = 1
    while (i <= input.length()) {
      if (!gradInput.contains(i)) gradInput(i) = Tensor[T]()
      if ((i < offset) || (i >= (offset + length))) {
        gradInput(i) = Utils.recursiveResizeAs(gradInput(i), input(i))
        Utils.recursiveFill(gradInput(i), 0)
      }
      i += 1
    }
    gradInput
  }

  override def toString(): String = {
    s"nn.NarrowTable($offset, $length)"
  }
}
