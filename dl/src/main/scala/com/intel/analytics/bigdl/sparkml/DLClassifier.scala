
package org.apache.spark.ml

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import org.apache.spark.ml.param._
import org.apache.spark.ml.param.shared.{HasInputCol, HasOutputCol}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.mllib.linalg.DenseVector
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Row}

class DLClassifier(override val uid: String) extends Transformer with HasInputCol with HasOutputCol with DataParams{

  var tensorBuffer: Tensor[Float] = null

  def this() = this(Identifiable.randomUID("DLClassifier"))

  def setInputCol(value: String): this.type = set(inputCol, value)

  def setOutputCol(value: String): this.type = set(outputCol, value)

  private def predict(features: Tensor[Float]): Array[Float] = {
    val result = $(modelTrain).forward(features).toTensor[Float]
    require(result.dim() == 2)

    var i = 0
    val res = new Array[Float](result.size(1))
    while (i < result.size(1)) {
      val maxVal = result.select(1, i + 1).max(1)._2
      res(i) = maxVal(Array(1))
      i += 1
    }
    res
  }

  private def toBatch(input: DataFrame): Tensor[Float] = {
    if (null == tensorBuffer) tensorBuffer = Tensor[Float]($(batchSize))
    tensorBuffer.zero()

    var i = 1
    input.select($(inputCol)).collect()
      .foreach {
        case Row(feature: Any) =>
          {
            tensorBuffer.select(1, i).copy(Tensor(Storage(feature.
              asInstanceOf[DenseVector].values.map(_.toFloat))))
            i += 1
          }
      }
    tensorBuffer
  }


  override def transform(dataset: DataFrame): DataFrame = {
    require(null != $(modelTrain), "model for predict must not be null")
    require(null != $(batchSize), "batchSize for predict must not be null")

    val predictUDF = udf {
      val res = predict(toBatch(dataset))
      var i = -1
      (features: Any) => {
        i += 1
        res(i)
      }
    }
    dataset.withColumn($(outputCol), predictUDF(col($(inputCol))))
  }

  override def transformSchema(schema: StructType): StructType = schema

  override def copy(extra: ParamMap): DLClassifier = {
    copyValues(new DLClassifier(uid), extra)
  }
}

trait DataParams extends Params {
  final val modelTrain = new Param[Module[Float]](this, "module factory", "network model")
  final val batchSize = new Param[Array[Int]](this, "batch size", "batch size for input")

  final def getModel = $(modelTrain)
  final def getBatchSize = $(batchSize)
}
