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
package com.intel.analytics.bigdl.utils.tf

import com.intel.analytics.bigdl.dataset.{DistributedDataSet, MiniBatch}
import com.intel.analytics.bigdl.nn.CrossEntropyCriterion
import com.intel.analytics.bigdl.optim.{SGD, Trigger}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Engine
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

import scala.collection.mutable

class SessionSpec extends FlatSpec with Matchers with BeforeAndAfter {
  Logger.getLogger("org").setLevel(Level.WARN)
  Logger.getLogger("akka").setLevel(Level.WARN)


  var sc: SparkContext = null

  var dataSet: DistributedDataSet[MiniBatch[Float]] = null

  before {
    val conf = Engine.createSparkConf()
    conf.set("spark.master", "local[1]")
    conf.set("spark.app.name", "SessionSpec")
    sc = new SparkContext(conf)
    Engine.init
    Engine.model.setPoolSize(1)
  }

  after {
    if (sc != null) {
      sc.stop()
    }
  }

  "Session" should "be able to run basic model" in {

    import scala.collection.JavaConverters._

    val path = "/home/yang/sources/bigdl/pyspark/graph/model.pb"

    val nodes = TensorflowLoader.parse(path)

    val session = new BigDLSessionImpl[Float](nodes.asScala,
      new mutable.HashMap[String, (Tensor[Float], Tensor[Float])])

    val data = new Array[Tensor[Float]](100)
    val label = new Array[Tensor[Float]](100)
    for (i <- Range(0, 100)) {
      val t = Tensor[Float](Array(784))
      val l = Tensor[Float](Array(1)).setValue(1, 2)
      data.update(i, t)
      label.update(i, l)
    }

    val optim = new SGD[Float](0.001)
    val criterion = CrossEntropyCriterion[Float]()
    val endWhen = Trigger.maxEpoch(5)

    session.train(Seq("BiasAdd"), data.toSeq, label.toSeq, optim, criterion, 16, endWhen)
  }

  "Session " should "be able to construct local data" in {
    import scala.collection.JavaConverters._

    val path = "/tmp/tfmodel/graph.pbtxt"

    val nodes = TensorflowLoader.parseTxt(path)

    val session = new BigDLSessionImpl[Float](nodes.asScala,
      new mutable.HashMap[String, (Tensor[Float], Tensor[Float])])

    val result = session.constructLocalData(Seq("parallel_read/filenames/RandomShuffle"))

    println(result)
  }

}
