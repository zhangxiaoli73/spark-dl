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

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.numeric.NumericDouble
import com.intel.analytics.bigdl.utils.{T, Table}

/**
 * no Dropout
 */
object GoogleNet_v1 {
  def apply(classNum: Int): Module[Double] = {
    val feature1 = Sequential()
    feature1.add(SpatialConvolution(3, 64, 7, 7, 2, 2, 3, 3, 1, true).setInitMethod(Xavier)
      .setName("conv1/7x7_s2"))
    feature1.add(ReLU(true).setName("conv1/relu_7x7"))
    feature1.add(SpatialMaxPooling(3, 3, 2, 2).ceil().setName("pool1/3x3_s2"))
    feature1.add(SpatialCrossMapLRN(5, 0.0001, 0.75).setName("pool1/norm1"))
    feature1.add(SpatialConvolution(64, 64, 1, 1, 1, 1).setInitMethod(Xavier)
      .setName("conv2/3x3_reduce"))
    feature1.add(ReLU(true).setName("conv2/relu_3x3_reduce"))
    feature1.add(SpatialConvolution(64, 192, 3, 3, 1, 1, 1, 1).setInitMethod(Xavier)
      .setName("conv2/3x3"))
    feature1.add(ReLU(true).setName("conv2/relu_3x3"))
    feature1.add(SpatialCrossMapLRN(5, 0.0001, 0.75). setName("conv2/norm2"))
    feature1.add(SpatialMaxPooling(3, 3, 2, 2).ceil().setName("pool2/3x3_s2"))
    feature1.add(Inception_v1(192, T(T(64), T(96, 128), T(16, 32), T(32)), "inception_3a/"))
    feature1.add(Inception_v1(256, T(T(128), T(128, 192), T(32, 96), T(64)), "inception_3b/"))
    feature1.add(SpatialMaxPooling(3, 3, 2, 2).ceil().setName("pool3/3x3_s2"))
    feature1.add(Inception_v1(480, T(T(192), T(96, 208), T(16, 48), T(64)), "inception_4a/"))

    val output1 = Sequential()
    output1.add(SpatialAveragePooling(5, 5, 3, 3).ceil().setName("loss1/ave_pool"))
    output1.add(SpatialConvolution(512, 128, 1, 1, 1, 1).setName("loss1/conv"))
    output1.add(ReLU(true).setName("loss1/relu_conv"))
    output1.add(View(128 * 4 * 4).setNumInputDims(3))
    output1.add(Linear(128 * 4 * 4, 1024).setName("loss1/fc"))
    output1.add(ReLU(true).setName("loss1/relu_fc"))
    // output1.add(Dropout(0.7).setName("loss1/drop_fc"))
    output1.add(Linear(1024, classNum).setName("loss1/classifier"))
    output1.add(LogSoftMax().setName("loss1/loss"))

    val feature2 = Sequential()
    feature2.add(Inception_v1(512, T(T(160), T(112, 224), T(24, 64), T(64)), "inception_4b/"))
    feature2.add(Inception_v1(512, T(T(128), T(128, 256), T(24, 64), T(64)), "inception_4c/"))
    feature2.add(Inception_v1(512, T(T(112), T(144, 288), T(32, 64), T(64)), "inception_4d/"))

    val output2 = Sequential()
    output2.add(SpatialAveragePooling(5, 5, 3, 3).setName("loss2/ave_pool"))
    output2.add(SpatialConvolution(528, 128, 1, 1, 1, 1).setName("loss2/conv"))
    output2.add(ReLU(true).setName("loss2/relu_conv"))
    output2.add(View(128 * 4 * 4).setNumInputDims(3))
    output2.add(Linear(128 * 4 * 4, 1024).setName("loss2/fc"))
    output2.add(ReLU(true).setName("loss2/relu_fc"))
    // output2.add(Dropout(0.7).setName("loss2/drop_fc"))
    output2.add(Linear(1024, classNum).setName("loss2/classifier"))
    output2.add(LogSoftMax().setName("loss2/loss"))

    val output3 = Sequential()
    output3.add(Inception_v1(528, T(T(256), T(160, 320), T(32, 128), T(128)), "inception_4e/"))
    output3.add(SpatialMaxPooling(3, 3, 2, 2).ceil().setName("pool4/3x3_s2"))
    output3.add(Inception_v1(832, T(T(256), T(160, 320), T(32, 128), T(128)), "inception_5a/"))
    output3.add(Inception_v1(832, T(T(384), T(192, 384), T(48, 128), T(128)), "inception_5b/"))
    output3.add(SpatialAveragePooling(7, 7, 1, 1).setName("pool5/7x7_s1"))
    // output3.add(Dropout(0.4).setName("pool5/drop_7x7_s1"))
    output3.add(View(1024).setNumInputDims(3))
    output3.add(Linear(1024, classNum).setInitMethod(Xavier).setName("loss3/classifier"))
    output3.add(LogSoftMax().setName("loss3/loss3"))

    val split2 = Concat(2).setName("split2")
    split2.add(output3)
    split2.add(output2)

    val mainBranch = Sequential()
    mainBranch.add(feature2)
    mainBranch.add(split2)

    val split1 = Concat(2).setName("split1")
    split1.add(mainBranch)
    split1.add(output1)

    val model = Sequential()

    model.add(feature1)
    model.add(split1)

    model.reset()
    model
  }
}

object Inception_v1 {
  def apply(inputSize: Int, config: Table, namePrefix : String = "") : Module[Double] = {
    val concat = Concat(2)
    val conv1 = Sequential()
    conv1.add(SpatialConvolution(inputSize,
      config[Table](1)(1), 1, 1, 1, 1).setInitMethod(Xavier).setName(namePrefix + "1x1"))
    conv1.add(ReLU(true).setName(namePrefix + "relu_1x1"))
    concat.add(conv1)
    val conv3 = Sequential()
    conv3.add(SpatialConvolution(inputSize,
      config[Table](2)(1), 1, 1, 1, 1).setInitMethod(Xavier).setName(namePrefix + "3x3_reduce"))
    conv3.add(ReLU(true).setName(namePrefix + "relu_3x3_reduce"))
    conv3.add(SpatialConvolution(config[Table](2)(1),
      config[Table](2)(2), 3, 3, 1, 1, 1, 1).setInitMethod(Xavier).setName(namePrefix + "3x3"))
    conv3.add(ReLU(true).setName(namePrefix + "relu_3x3"))
    concat.add(conv3)
    val conv5 = Sequential()
    conv5.add(SpatialConvolution(inputSize,
      config[Table](3)(1), 1, 1, 1, 1).setInitMethod(Xavier).setName(namePrefix + "5x5_reduce"))
    conv5.add(ReLU(true).setName(namePrefix + "relu_5x5_reduce"))
    conv5.add(SpatialConvolution(config[Table](3)(1),
      config[Table](3)(2), 5, 5, 1, 1, 2, 2).setInitMethod(Xavier).setName(namePrefix + "5x5"))
    conv5.add(ReLU(true).setName(namePrefix + "relu_5x5"))
    concat.add(conv5)
    val pool = Sequential()
    pool.add(SpatialMaxPooling(3, 3, 1, 1, 1, 1).ceil().setName(namePrefix + "pool"))
    pool.add(SpatialConvolution(inputSize,
      config[Table](4)(1), 1, 1, 1, 1).setInitMethod(Xavier).setName(namePrefix + "pool_proj"))
    pool.add(ReLU(true).setName(namePrefix + "relu_pool_proj"))
    concat.add(pool).setName(namePrefix + "output")
    concat
  }
}

object GoogleNet_v2 {
  def apply(classNum: Int): Module[Double] = {
    val features1 = Sequential()
    features1.add(SpatialConvolution(3, 64, 7, 7, 2, 2, 3, 3, 1, true)
      .setName("conv1/7x7_s2"))
    features1.add(SpatialBatchNormalization(64, 1e-3).setInit().setName("conv1/7x7_s2/bn"))
    features1.add(ReLU(true).setName("conv1/7x7_s2/bn/sc/relu"))
    features1.add(SpatialMaxPooling(3, 3, 2, 2).ceil().setName("pool1/3x3_s2"))
    features1.add(SpatialConvolution(64, 64, 1, 1).setName("conv2/3x3_reduce"))
    features1.add(SpatialBatchNormalization(64, 1e-3).setInit().setName("conv2/3x3_reduce/bn"))
    features1.add(ReLU(true).setName("conv2/3x3_reduce/bn/sc/relu"))
    features1.add(SpatialConvolution(64, 192, 3, 3, 1, 1, 1, 1).setName("conv2/3x3"))
    features1.add(SpatialBatchNormalization(192, 1e-3).setInit().setName("conv2/3x3/bn"))
    features1.add(ReLU(true).setName("conv2/3x3/bn/sc/relu"))
    features1.add(SpatialMaxPooling(3, 3, 2, 2).ceil().setName("pool2/3x3_s2"))
    features1.add(Inception_v2(192, T(T(64), T(64, 64), T(64, 96), T("avg", 32)), "inception_3a/"))
    features1.add(Inception_v2(256, T(T(64), T(64, 96), T(64, 96), T("avg", 64)), "inception_3b/"))
    features1.add(Inception_v2(320, T(T(0), T(128, 160), T(64, 96), T("max", 0)), "inception_3c/"))

    val output1 = Sequential()
    output1.add(SpatialAveragePooling(5, 5, 3, 3).ceil().setName("pool3/5x5_s3"))
    output1.add(SpatialConvolution(576, 128, 1, 1, 1, 1).setName("loss1/conv"))
    output1.add(SpatialBatchNormalization(128, 1e-3).setInit().setName("loss1/conv/bn"))
    output1.add(ReLU(true).setName("loss1/conv/bn/sc/relu"))
    output1.add(View(128 * 4 * 4).setNumInputDims(3))
    output1.add(Linear(128 * 4 * 4, 1024).setName("loss1/fc"))
    output1.add(ReLU(true).setName("loss1/fc/bn/sc/relu"))
    output1.add(Linear(1024, classNum).setName("loss1/classifier"))
    output1.add(LogSoftMax().setName("loss1/loss"))


    val features2 = Sequential()
    features2
      .add(Inception_v2(576, T(T(224), T(64, 96), T(96, 128), T("avg", 128)), "inception_4a/"))
      .add(Inception_v2(576, T(T(192), T(96, 128), T(96, 128), T("avg", 128)), "inception_4b/"))
      .add(Inception_v2(576, T(T(160), T(128, 160), T(128, 160), T("avg", 96)), "inception_4c/"))
      .add(Inception_v2(576, T(T(96), T(128, 192), T(160, 192), T("avg", 96)), "inception_4d/"))
      .add(Inception_v2(576, T(T(0), T(128, 192), T(192, 256), T("max", 0)), "inception_4e/"))

    val output2 = Sequential()
    output2.add(SpatialAveragePooling(5, 5, 3, 3).ceil().setName("pool4/5x5_s3"))
    output2.add(SpatialConvolution(1024, 128, 1, 1, 1, 1).setName("loss2/conv"))
    output2.add(SpatialBatchNormalization(128, 1e-3).setInit().setName("loss2/conv/bn"))
    output2.add(ReLU(true).setName("loss2/conv/bn/sc/relu"))
    output2.add(View(128 * 2 * 2).setNumInputDims(3))
    output2.add(Linear(128 * 2 * 2, 1024).setName("loss2/fc"))
    output2.add(ReLU(true).setName("loss2/fc/bn/sc/relu"))
    output2.add(Linear(1024, classNum).setName("loss2/classifier"))
    output2.add(LogSoftMax().setName("loss2/loss"))

    val output3 = Sequential()
    output3.add(Inception_v2(1024, T(T(352), T(192, 320), T(160, 224), T("avg", 128)),
      "inception_5a/"))
    output3.add(Inception_v2(1024, T(T(352), T(192, 320), T(192, 224), T("max", 128)),
      "inception_5b/"))
    output3.add(SpatialAveragePooling(7, 7, 1, 1).ceil().setName("pool5/7x7_s1"))
    output3.add(View(1024).setNumInputDims(3))
    output3.add(Linear(1024, classNum).setName("loss3/classifier"))
    output3.add(LogSoftMax().setName("loss3/loss"))

    val split2 = Concat(2)
    split2.add(output3)
    split2.add(output2)

    val mainBranch = Sequential()
    mainBranch.add(features2)
    mainBranch.add(split2)

    val split1 = Concat(2)
    split1.add(mainBranch)
    split1.add(output1)

    val model = Sequential()

    model.add(features1)
    model.add(split1)

    model.reset()
    model
  }

  def applyNoBn(classNum: Int): Module[Double] = {
    val features1 = Sequential()
    features1.add(SpatialConvolution(3, 64, 7, 7, 2, 2, 3, 3, 1, true)
      .setName("conv1/7x7_s2"))
    features1.add(ReLU(true).setName("conv1/7x7_s2/bn/sc/relu"))
    features1.add(SpatialMaxPooling(3, 3, 2, 2).ceil().setName("pool1/3x3_s2"))
    features1.add(SpatialConvolution(64, 64, 1, 1).setName("conv2/3x3_reduce"))
    features1.add(ReLU(true).setName("conv2/3x3_reduce/bn/sc/relu"))
    features1.add(SpatialConvolution(64, 192, 3, 3, 1, 1, 1, 1).setName("conv2/3x3"))
    features1.add(ReLU(true).setName("conv2/3x3/bn/sc/relu"))
    features1.add(SpatialMaxPooling(3, 3, 2, 2).ceil().setName("pool2/3x3_s2"))
    features1.add(Inception_v2.applyNoBn(192, T(T(64), T(64, 64), T(64, 96), T("avg", 32)),
      "inception_3a/"))
    features1.add(Inception_v2.applyNoBn(256, T(T(64), T(64, 96), T(64, 96), T("avg", 64)),
      "inception_3b/"))
    features1.add(Inception_v2.applyNoBn(320, T(T(0), T(128, 160), T(64, 96), T("max", 0)),
      "inception_3c/"))

    val output1 = Sequential()
    output1.add(SpatialAveragePooling(5, 5, 3, 3).ceil().setName("pool3/5x5_s3"))
    output1.add(SpatialConvolution(576, 128, 1, 1, 1, 1).setName("loss1/conv"))
    output1.add(ReLU(true).setName("loss1/conv/bn/sc/relu"))
    output1.add(View(128 * 4 * 4).setNumInputDims(3))
    output1.add(Linear(128 * 4 * 4, 1024).setName("loss1/fc"))
    output1.add(ReLU(true).setName("loss1/fc/bn/sc/relu"))
    output1.add(Linear(1024, classNum).setName("loss1/classifier"))
    output1.add(LogSoftMax().setName("loss1/loss"))


    val features2 = Sequential()
    features2
      .add(Inception_v2.applyNoBn(576, T(T(224), T(64, 96), T(96, 128), T("avg", 128)),
        "inception_4a/"))
      .add(Inception_v2.applyNoBn(576, T(T(192), T(96, 128), T(96, 128), T("avg", 128)),
        "inception_4b/"))
      .add(Inception_v2.applyNoBn(576, T(T(160), T(128, 160), T(128, 160), T("avg", 96)),
        "inception_4c/"))
      .add(Inception_v2.applyNoBn(576, T(T(96), T(128, 192), T(160, 192), T("avg", 96)),
        "inception_4d/"))
      .add(Inception_v2.applyNoBn(576, T(T(0), T(128, 192), T(192, 256), T("max", 0)),
        "inception_4e/"))

    val output2 = Sequential()
    output2.add(SpatialAveragePooling(5, 5, 3, 3).ceil().setName("pool4/5x5_s3"))
    output2.add(SpatialConvolution(1024, 128, 1, 1, 1, 1).setName("loss2/conv"))
    output2.add(ReLU(true).setName("loss2/conv/bn/sc/relu"))
    output2.add(View(128 * 2 * 2).setNumInputDims(3))
    output2.add(Linear(128 * 2 * 2, 1024).setName("loss2/fc"))
    output2.add(ReLU(true).setName("loss2/fc/bn/sc/relu"))
    output2.add(Linear(1024, classNum).setName("loss2/classifier"))
    output2.add(LogSoftMax().setName("loss2/loss"))

    val output3 = Sequential()
    output3.add(Inception_v2.applyNoBn(1024, T(T(352), T(192, 320), T(160, 224), T("avg", 128)),
      "inception_5a/"))
    output3.add(Inception_v2.applyNoBn(1024, T(T(352), T(192, 320), T(192, 224), T("max", 128)),
      "inception_5b/"))
    output3.add(SpatialAveragePooling(7, 7, 1, 1).ceil().setName("pool5/7x7_s1"))
    output3.add(View(1024).setNumInputDims(3))
    output3.add(Linear(1024, classNum).setName("loss3/classifier"))
    output3.add(LogSoftMax().setName("loss3/loss"))

    val split2 = Concat(2)
    split2.add(output3)
    split2.add(output2)

    val mainBranch = Sequential()
    mainBranch.add(features2)
    mainBranch.add(split2)

    val split1 = Concat(2)
    split1.add(mainBranch)
    split1.add(output1)

    val model = Sequential()

    model.add(features1)
    model.add(split1)

    model.reset()
    model
  }
}

object Inception_v2 {
  def apply(inputSize: Int, config: Table, namePrefix : String): Module[Double] = {
    val concat = Concat(2)
    if (config[Table](1)[Int](1) != 0) {
      val conv1 = Sequential()
      conv1.add(SpatialConvolution(inputSize, config[Table](1)(1), 1, 1, 1, 1)
        .setName(namePrefix + "1x1"))
      conv1.add(SpatialBatchNormalization(config[Table](1)(1), 1e-3).setInit()
        .setName(namePrefix + "1x1/bn"))
      conv1.add(ReLU(true).setName(namePrefix + "1x1/bn/sc/relu"))
      concat.add(conv1)
    }

    val conv3 = Sequential()
    conv3.add(SpatialConvolution(inputSize, config[Table](2)(1), 1, 1, 1, 1)
      .setName(namePrefix + "3x3_reduce"))
    conv3.add(SpatialBatchNormalization(config[Table](2)(1), 1e-3).setInit()
      .setName(namePrefix + "3x3_reduce/bn"))
    conv3.add(ReLU(true). setName(namePrefix + "3x3_reduce/bn/sc/relu"))
    if(config[Table](4)[String](1) == "max" && config[Table](4)[Int](2) == 0) {
      conv3.add(SpatialConvolution(config[Table](2)(1),
        config[Table](2)(2), 3, 3, 2, 2, 1, 1).setName(namePrefix + "3x3"))
    } else {
      conv3.add(SpatialConvolution(config[Table](2)(1),
        config[Table](2)(2), 3, 3, 1, 1, 1, 1).setName(namePrefix + "3x3"))
    }
    conv3.add(SpatialBatchNormalization(config[Table](2)(2), 1e-3).setInit()
      .setName(namePrefix + "3x3/bn"))
    conv3.add(ReLU(true).setName(namePrefix + "3x3/bn/sc/relu"))
    concat.add(conv3)

    val conv3xx = Sequential()
    conv3xx.add(SpatialConvolution(inputSize, config[Table](3)(1), 1, 1, 1, 1)
      .setName(namePrefix + "double3x3_reduce"))
    conv3xx.add(SpatialBatchNormalization(config[Table](3)(1), 1e-3).setInit()
      .setName(namePrefix + "double3x3_reduce/bn"))
    conv3xx.add(ReLU(true).setName(namePrefix + "double3x3_reduce/bn/sc/relu"))

    conv3xx.add(SpatialConvolution(config[Table](3)(1),
      config[Table](3)(2), 3, 3, 1, 1, 1, 1).setName(namePrefix + "double3x3a"))
    conv3xx.add(SpatialBatchNormalization(config[Table](3)(2), 1e-3).setInit()
      .setName(namePrefix + "double3x3a/bn"))
    conv3xx.add(ReLU(true).setName(namePrefix + "double3x3a/bn/sc/relu"))

    if(config[Table](4)[String](1) == "max" && config[Table](4)[Int](2) == 0) {
      conv3xx.add(SpatialConvolution(config[Table](3)(2),
        config[Table](3)(2), 3, 3, 2, 2, 1, 1).setName(namePrefix + "double3x3b"))
    } else {
      conv3xx.add(SpatialConvolution(config[Table](3)(2),
        config[Table](3)(2), 3, 3, 1, 1, 1, 1).setName(namePrefix + "double3x3b"))
    }
    conv3xx.add(SpatialBatchNormalization(config[Table](3)(2), 1e-3).setInit()
      .setName(namePrefix + "double3x3b/bn"))
    conv3xx.add(ReLU(true).setName(namePrefix + "double3x3b/bn/sc/relu"))
    concat.add(conv3xx)

    val pool = Sequential()
    config[Table](4)[String](1) match {
      case "max" =>
        if (config[Table](4)[Int](2) != 0) {
          pool.add(SpatialMaxPooling(3, 3, 1, 1, 1, 1).ceil().setName(namePrefix + "pool"))
        } else {
          pool.add(SpatialMaxPooling(3, 3, 2, 2).ceil().setName(namePrefix + "pool"))
        }
      case "avg" => pool.add(SpatialAveragePooling(3, 3, 1, 1, 1, 1).ceil()
        .setName(namePrefix + "pool"))
      case _ => throw new IllegalArgumentException
    }

    if (config[Table](4)[Int](2) != 0) {
      pool.add(SpatialConvolution(inputSize, config[Table](4)[Int](2), 1, 1, 1, 1)
        .setName(namePrefix + "pool_proj"))
      pool.add(SpatialBatchNormalization(config[Table](4)(2), 1e-3).setInit()
        .setName(namePrefix + "pool_proj/bn"))
      pool.add(ReLU(true).setName(namePrefix + "pool_proj/bn/sc/relu"))
    }
    concat.add(pool)
    concat.setName(namePrefix + "output")
  }

  def applyNoBn(inputSize: Int, config: Table, namePrefix : String): Module[Double] = {
    val concat = Concat(2)
    if (config[Table](1)[Int](1) != 0) {
      val conv1 = Sequential()
      conv1.add(SpatialConvolution(inputSize, config[Table](1)(1), 1, 1, 1, 1)
        .setName(namePrefix + "1x1"))
      conv1.add(ReLU(true).setName(namePrefix + "1x1/bn/sc/relu"))
      concat.add(conv1)
    }

    val conv3 = Sequential()
    conv3.add(SpatialConvolution(inputSize, config[Table](2)(1), 1, 1, 1, 1)
      .setName(namePrefix + "3x3_reduce"))
    conv3.add(ReLU(true). setName(namePrefix + "3x3_reduce/bn/sc/relu"))
    if(config[Table](4)[String](1) == "max" && config[Table](4)[Int](2) == 0) {
      conv3.add(SpatialConvolution(config[Table](2)(1),
        config[Table](2)(2), 3, 3, 2, 2, 1, 1).setName(namePrefix + "3x3"))
    } else {
      conv3.add(SpatialConvolution(config[Table](2)(1),
        config[Table](2)(2), 3, 3, 1, 1, 1, 1).setName(namePrefix + "3x3"))
    }
    conv3.add(ReLU(true).setName(namePrefix + "3x3/bn/sc/relu"))
    concat.add(conv3)

    val conv3xx = Sequential()
    conv3xx.add(SpatialConvolution(inputSize, config[Table](3)(1), 1, 1, 1, 1)
      .setName(namePrefix + "double3x3_reduce"))
    conv3xx.add(ReLU(true).setName(namePrefix + "double3x3_reduce/bn/sc/relu"))

    conv3xx.add(SpatialConvolution(config[Table](3)(1),
      config[Table](3)(2), 3, 3, 1, 1, 1, 1).setName(namePrefix + "double3x3a"))
    conv3xx.add(ReLU(true).setName(namePrefix + "double3x3a/bn/sc/relu"))

    if(config[Table](4)[String](1) == "max" && config[Table](4)[Int](2) == 0) {
      conv3xx.add(SpatialConvolution(config[Table](3)(2),
        config[Table](3)(2), 3, 3, 2, 2, 1, 1).setName(namePrefix + "double3x3b"))
    } else {
      conv3xx.add(SpatialConvolution(config[Table](3)(2),
        config[Table](3)(2), 3, 3, 1, 1, 1, 1).setName(namePrefix + "double3x3b"))
    }
    conv3xx.add(ReLU(true).setName(namePrefix + "double3x3b/bn/sc/relu"))
    concat.add(conv3xx)

    val pool = Sequential()
    config[Table](4)[String](1) match {
      case "max" =>
        if (config[Table](4)[Int](2) != 0) {
          pool.add(SpatialMaxPooling(3, 3, 1, 1, 1, 1).ceil().setName(namePrefix + "pool"))
        } else {
          pool.add(SpatialMaxPooling(3, 3, 2, 2).ceil().setName(namePrefix + "pool"))
        }
      case "avg" => pool.add(SpatialAveragePooling(3, 3, 1, 1, 1, 1).ceil()
        .setName(namePrefix + "pool"))
      case _ => throw new IllegalArgumentException
    }

    if (config[Table](4)[Int](2) != 0) {
      pool.add(SpatialConvolution(inputSize, config[Table](4)[Int](2), 1, 1, 1, 1)
        .setName(namePrefix + "pool_proj"))
      pool.add(ReLU(true).setName(namePrefix + "pool_proj/bn/sc/relu"))
    }
    concat.add(pool)
    concat.setName(namePrefix + "output")
  }
}
