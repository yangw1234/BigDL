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

import java.nio.{ByteOrder, DoubleBuffer, FloatBuffer}

import com.intel.analytics.bigdl.Criterion
import com.intel.analytics.bigdl.dataset.{DataSet, Sample}
import com.intel.analytics.bigdl.nn.{ClassNLLCriterion, Graph, Linear}
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.optim.{OptimMethod, Optimizer, SGD, Trigger}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.utils._
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.tensorflow.framework.{GraphDef, NodeDef}

import scala.collection.mutable
import scala.reflect.ClassTag

abstract class Session[T: ClassTag] {

  def train(trainOp: String,
            optimizer: String,
            batchSize: Int,
            endWhen: Trigger): Unit

  def run(endPoints: Array[String],
          batchSize: Int): RDD[Array[Tensor[T]]]

  def train(outputs: Seq[String],
            data: Seq[Tensor[T]],
            label: Seq[Tensor[T]],
            optMethod: OptimMethod[T],
            criterion: Criterion[T],
            batchSize: Int,
            endWhen: Trigger): Graph[T]

  def train(modelOutputs: Seq[String],
            labels: Seq[String],
            optMethod: OptimMethod[T],
            criterion: Criterion[T],
            endWhen: Trigger): Graph[T]


  // def setGraph(graphDef: GraphDef): Unit
}

class BigDLSessionImpl[T: ClassTag](
       graph: Seq[NodeDef],
       context: mutable.HashMap[String, (Tensor[T], Tensor[T])])
                         (implicit ev: TensorNumeric[T]) extends Session[T] {
  import scala.collection.JavaConverters._

  val sc = SparkContext.getOrCreate()
  Engine.init


  private val noInputOp = Set("Const", "VariableV2", "NoOp")

  private val inputOp = Set("ReaderReadV2", "QueueDequeueV2", "QueueDequeueManyV2", "Placeholder")

  private val enqueueOp = Set("QueueEnqueueV2")

  private val readerOps = Set("TFRecordReaderV2")

  private val (wholeTFGraph, _) = TensorflowLoader.buildTFGraph(graph.asJava, null)

  private val name2Node = wholeTFGraph.
    DFS.filter(_.element != null).map(node => (node.element.getName, node)).toMap

  private def tableToSeq(table: Table): Seq[Tensor[T]] = {
    for (i <- 0 until table.length()) yield {
      table(i).asInstanceOf[Tensor[T]]
    }
  }

  private def seqToTable(tensors: Seq[Tensor[T]]): Table = {
    val table = new Table()
    for (tensor <- tensors) {
      table.insert(tensor)
    }
    table
  }

  private def handleReaderNode(node: Node[NodeDef]): RDD[Table] = {
    require(node.prevNodes.length == 2, "require ReaderReadV2 only has two inputs")
    val readerNode = node.prevNodes.head
    val queueNode = node.prevNodes(1)
    val enqueueNodes = queueNode.nextNodes.filter(n => enqueueOp(n.element.getOp))
    val filesSeq = enqueueNodes.map { enqueueNode =>
      val inputs = enqueueNode.prevNodes.filter(_ != queueNode).map(_.element.getName)
      constructLocalData(inputs)
    }.reduce { (outerSeq1, outerSeq2) =>
      outerSeq1.zip(outerSeq2).map { case (seq1, seq2) =>
        seq1.add(seq2)
      }
    }
    readerNode.element.getOp match {
      case "TFRecordReaderV2" => readTFRecord(filesSeq)
    }

  }

  private def readTFRecord(filesTensor: Seq[Table]): RDD[Table] = {
    throw new Exception()
//    val result = filesTensor.map { t =>
//        require(t.length() == 1 && t(1).isInstanceOf[Tensor[String]],
//          "Reader can only read one file at a time")
//        val file = t(1).asInstanceOf[Tensor[String]].apply(Array(1))
//        file
//    }.flatMap{ file =>
//      val iter = new TFRecordIterator(new java.io.File(file))
//      iter
//    }.map { record =>
//      T(Tensor[String](Array(record), Array(1)))
//    }
//    sc.parallelize(result, numSlices = 4)
  }

  private def handleLocalDequeue(node: Node[NodeDef]): Seq[Table] = {
    require(node.prevNodes.length == 1, "require QueueDequeueV2 only has one input")
    val queueNode = node.prevNodes.head
    val enqueueNodes = queueNode.nextNodes.filter(n => enqueueOp(n.element.getOp))
    val dataSeq = enqueueNodes.map { enqueueNode =>
      val inputs = enqueueNode.prevNodes.filter(_ != queueNode).map(_.element.getName)
      constructLocalData(inputs)
    }.reduce { (outerSeq1, outerSeq2) =>
      outerSeq1.zip(outerSeq2).map { case (seq1, seq2) =>
        seq1.add(seq2)
      }
    }
    dataSeq
  }

  private def handleDistriDequeue(node: Node[NodeDef]): RDD[Table] = {
    require(node.prevNodes.length == 1, "require QueueDequeueV2 only has one input")
    val queueNode = node.prevNodes.head
    val enqueueNodes = queueNode.nextNodes.filter(n => enqueueOp(n.element.getOp))
    val rdd = enqueueNodes.map { enqueueNode =>
      val inputs = enqueueNode.prevNodes.filter(_ != queueNode).map(_.element.getName)
      constructDistributeData(inputs)
    }.reduce { (rdd1, rdd2) =>
      rdd1.zip(rdd2).map { case (seq1, seq2) =>
        seq1.add(seq2)
      }
    }
    rdd
  }

  private def handleDistriDequeueManyNode(node: Node[NodeDef]): RDD[Table] = {
    require(node.prevNodes.length == 2, "require QueueDequeueManyV2 only has two input")
    val queueNode = node.prevNodes.head
    val enqueueNodes = queueNode.nextNodes.filter(n => enqueueOp(n.element.getOp))
    // get previous rdd
    val rdd = enqueueNodes.map { enqueueNode =>
      val inputs = enqueueNode.prevNodes.filter(_ != queueNode).map(_.element.getName)
      constructDistributeData(inputs)
    }.reduce { (rdd1, rdd2) =>
      rdd1.zip(rdd2).map { case (seq1, seq2) =>
        seq1.add(seq2)
      }
    }

    // get batch size
    val batchSizeNode = node.prevNodes(1)
    require(batchSizeNode.element.getOp == "Const", "batchsize must be a const")

    val batchSize = batchSizeNode.element.getAttrMap.get("value").getI.toInt

    val batchRdd = rdd.mapPartitions { iter =>

      new Iterator[Table] {
        override def hasNext: Boolean = iter.hasNext

        override def next(): Table = {
          require(iter.hasNext, "Call next() on a empty iterator")
          val batch = for (_ <- 0 until batchSize if iter.hasNext) yield {
            iter.next()
          }
          pack(batch)
        }
      }

    }
    batchRdd
  }

  private def pack(tables: Seq[Table], dimension: Int = 1): Table = {
    val batch = tables.map(tableToSeq)
    val firstSeq = batch.head
    val sizes = firstSeq.map { tensor =>
      val nDim = tensor.nDimension()
      val size: Array[Int] = new Array[Int](nDim + 1)
      var i = 1
      while(i <= nDim + 1) {
        if (i < dimension) {
          size(i-1) = tensor.size(i)
        } else if (i == dimension) {
          size(i-1) = batch.length
        } else {
          size(i-1) = tensor.size(i - 1)
        }
        i = i + 1
      }
      size
    }

    val results = sizes.map { size =>
      Tensor[T](size)
    }

    for ((seq, index) <- batch.zipWithIndex) {
      results.zip(seq).foreach { case (result, tensor) =>
        result.narrow(dimension, index + 1, 1).copy(tensor)
      }
    }
    seqToTable(results)
  }

  private def constructLocalData(endPoints: Seq[String]): Seq[Table] = {
    val isInputOp = (n: NodeDef) => inputOp(n.getOp)
    val (tfGraph, inputs) = TensorflowLoader.buildTFGraph(graph.asJava, endPoints, isInputOp)

    val inputNodes = inputs.map(name2Node)
    val inputDataSeq = inputNodes.map { node => // this is the input op
      node.element.getOp match {
          // only support Dequeue before reader
        case "QueueDequeueV2" => handleLocalDequeue(node)
      }
    }

    val reducedInputSeq = inputDataSeq.reduce { (outerSeq1, outerSeq2) =>
      outerSeq1.zip(outerSeq2).map { case (seq1, seq2) =>
        seq1.add(seq2)
      }
    }

    val transformer = TensorflowLoader.buildBigDLModel(
      tfGraph,
      inputNodes.map(_.element.getName),
      endPoints,
      ByteOrder.LITTLE_ENDIAN,
      "",
      Some(context)
    ).asInstanceOf[Graph[T]]

    reducedInputSeq.map { tensors =>
      val output = transformer.forward(tensors)
      output.asInstanceOf[Table]
    }
  }

  private def constructDistributeData(endPoints: Seq[String]): RDD[Table] = {
    val isInputOp = (n: NodeDef) => inputOp(n.getOp)
    val (tfGraph, inputs) = TensorflowLoader.buildTFGraph(graph.asJava, endPoints, isInputOp)

    val inputNodes = inputs.map(name2Node)

    val inputRdds = inputNodes.map { node => // this is the input op
      node.element.getOp match {
        case "ReaderReadV2" => handleReaderNode(node)
        case "QueueDequeueV2" => handleDistriDequeue(node)
        case "QueueDequeueManyV2" => handleDistriDequeueManyNode(node)
      }

    }
    val inputRdd = inputRdds.reduce { (rdd1, rdd2) =>
      rdd1.zip(rdd2).map { case (seq1, seq2) =>
        seq1.add(seq2)
      }
    }

    val transformer = TensorflowLoader.buildBigDLModel(
      tfGraph,
      inputNodes.map(_.element.getName),
      endPoints,
      ByteOrder.LITTLE_ENDIAN,
      "",
      Some(context)
    ).asInstanceOf[Graph[T]]

    inputRdd.map { tensors =>
      val output = transformer.forward(tensors)
      output.asInstanceOf[Table]
    }
  }

  private def constructModel(endPoints: Seq[String]): (Graph[T], Node[NodeDef]) = {
    val isInputOp = (n: NodeDef) => inputOp(n.getOp)
    val (tfGraph, inputs) = TensorflowLoader.buildTFGraph(graph.asJava, endPoints, isInputOp)

    val inputNodes = inputs.map(name2Node)

    require(inputNodes.length == 1, "Only support one model input")

    val model = TensorflowLoader.buildBigDLModel(
      tfGraph,
      inputNodes.map(_.element.getName),
      endPoints,
      ByteOrder.LITTLE_ENDIAN,
      "",
      Some(context)
    ).asInstanceOf[Graph[T]]
    (model, inputNodes.head)
  }

  override def train(trainOp: String, optimizer: String,
   batchSize: Int, endWhen: Trigger): Unit = {
    throw new NotImplementedError()
  }

  override def train(outputs: Seq[String],
                     data: Seq[Tensor[T]],
                     label: Seq[Tensor[T]],
                     optMethod: OptimMethod[T],
                     criterion: Criterion[T],
                     batchSize: Int, endWhen: Trigger): Graph[T] = {

    val samples = data.zip(label).map { elem =>
      Sample(elem._1, elem._2)
    }

    val coreNum = Engine.coreNumber()
    val rdd = sc.parallelize(samples, coreNum)

    val (model, input) = constructModel(outputs)

    require(input.element.getOp == "Placeholder",
      "only support Placeholder as input when in-memory input data is provided")

    val opt = Optimizer(
      model,
      rdd,
      criterion,
      batchSize
    )
    val optMethod = new SGD[T]()
    opt.setOptimMethod(optMethod).setEndWhen(endWhen)
      .optimize()
    model
  }

  override def train(modelOutputs: Seq[String],
                     labels: Seq[String],
                     optMethod: OptimMethod[T],
                     criterion: Criterion[T],
                     endWhen: Trigger): Graph[T] = {
    val (model, modelInput) = constructModel(modelOutputs)

    val (transformerForLabel, labelInput) = constructModel(labels)

    require(modelInput == labelInput, "data and label should come from the same queue")

    val data = constructDistributeData(Seq(input.element.getName) ++ labels)
  }

  override def run(endPoints: Array[String], batchSize: Int): RDD[Array[Tensor[T]]] = {
    throw new NotImplementedError()
  }

}
