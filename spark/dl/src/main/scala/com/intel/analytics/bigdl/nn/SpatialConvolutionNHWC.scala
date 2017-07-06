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

package com.intel.analytics.bigdl.nn

import com.intel.analytics.bigdl.nn.abstractnn.{Initializable, TensorModule}
import com.intel.analytics.bigdl.optim.Regularizer
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor._
import com.intel.analytics.bigdl.utils._
import com.intel.analytics.bigdl.utils.RandomGenerator._

import scala.concurrent.duration.Duration
import scala.concurrent.{Await, Future}
import scala.reflect.ClassTag

/**
 * Applies a 2D convolution over an input image composed of several input planes.
 * The input tensor in forward(input) is expected to be
 * a 3D tensor (nInputPlane x height x width).
 *
 * @param wRegularizer: instance of [[Regularizer]]
 *                    (eg. L1 or L2 regularization), applied to the input weights matrices.
 * @param bRegularizer: instance of [[Regularizer]]
 *                    applied to the bias.
 */

class SpatialConvolutionNHWC[T: ClassTag](
  val nInputPlane: Int, // The number of expected input planes in the image given into forward()
  val nOutputPlane: Int, // The number of output planes the convolution layer will produce.
  val kernelW: Int, // The kernel width of the convolution
  val kernelH: Int, // The kernel height of the convolution
  val strideW: Int = 1, // The step of the convolution in the width dimension.
  val strideH: Int = 1, // The step of the convolution in the height dimension
  val padW: Int = 0, // The additional zeros added per width to the input planes.
  val padH: Int = 0, // The additional zeros added per height to the input planes.
  // val nGroup: Int = 1, // Kernel group number
  val propagateBack: Boolean = true, // propagate gradient back
  var wRegularizer: Regularizer[T] = null,
  var bRegularizer: Regularizer[T] = null,
  val initWeight: Tensor[T] = null,
  val initBias: Tensor[T] = null,
  val initGradWeight: Tensor[T] = null,
  val initGradBias: Tensor[T] = null
)(implicit ev: TensorNumeric[T]) extends TensorModule[T] with Initializable {

  // require(nInputPlane % nGroup == 0, "Number of input channels should be multiples of group.")
  // require(nOutputPlane % nGroup == 0, "Number of output channels should be multiples of group.")

  val weight: Tensor[T] = if (initWeight != null) {
    initWeight
  } else {
    // Tensor[T](nGroup, nOutputPlane / nGroup, nInputPlane / nGroup, kernelH, kernelW)
    Tensor[T](kernelH, kernelW, nInputPlane, nOutputPlane)
  }

  val bias: Tensor[T] = if (initBias != null) initBias else Tensor[T](nOutputPlane)

  val gradWeight: Tensor[T] = if (initGradWeight != null) {
    initGradWeight
  } else {
    Tensor[T](kernelH, kernelW, nInputPlane, nOutputPlane)
  }

  val gradBias: Tensor[T] = if (initGradBias != null) initGradBias else Tensor[T](nOutputPlane)

  var fInput = Tensor[T]()
  var fGradInput = Tensor[T]()
  protected val ones = Tensor[T]()
  protected val onesBatch = Tensor[T]()
  protected val onesBias = Tensor[T]()
  protected var weightMM: Tensor[T] = null
  protected val gradientBiasMT: Tensor[T] = Tensor[T]()
  protected var gradWeightMM: Tensor[T] = null
  @transient
  protected var gradWeightMMInBatch: Tensor[T] = null
  protected val _1x1 = if (kernelH == 1 && kernelW == 1 && strideW == 1 && strideH == 1
    && padH == 0 && padW == 0) {
    true
  } else {
    false
  }

  {
    val stdv = 1.0 / math.sqrt(kernelW * kernelH * nInputPlane)
    val wInit: InitializationMethod = RandomUniform(-stdv, stdv)
    val bInit: InitializationMethod = RandomUniform(-stdv, stdv)
    setInitMethod(wInit, bInit)
  }

  protected var im2colTime = 0L
  protected var col2imTime = 0L

  def getIm2ColTime(): Double = im2colTime

  def getCol2ImgTime(): Double = col2imTime

  @transient
  protected var results: Array[Future[Unit]] = null

  override def reset(): Unit = {
    if (initWeight == null) {
      weightInitMethod.init(weight, VariableFormat.GP_OUT_IN_KW_KH)
    }
    if (initBias == null) {
      biasInitMethod.init(bias, VariableFormat.ONE_D)
    }
    zeroGradParameters()
  }

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    require(input.dim() == 3 || input.dim() == 4,
      "SpatialConvolution: " + ErrorInfo.constrainInputAs3DOrBatch)
    require(input.isContiguous())

    if (weightMM == null || weightMM.storage().isEmpty) {
      weightMM = weight.view(nInputPlane * kernelH * kernelW, nOutputPlane)
    }
    val dimWidth = if (input.dim() == 3) 2 else 3
    val dimHeight = if (input.dim() == 3) 1 else 2

    val inputWidth = input.size(dimWidth)
    val inputHeight = input.size(dimHeight)

    val outputWidth = (inputWidth + 2 * padW - kernelW) / strideW + 1
    val outputHeight = (inputHeight + 2 * padH - kernelH) / strideH + 1

    require(outputWidth >= 1 && outputHeight >= 1,
      s"output size is too small. outputWidth: $outputWidth, outputHeight: $outputHeight")

    if (onesBias.dim() != 1 || onesBias.size(1) != outputHeight * outputWidth) {
      onesBias.resize(Array(outputHeight * outputWidth)).fill(ev.fromType(1.0))
    }

    if (input.dim() == 3) {
      require(input.size(3) == nInputPlane)
      require(input.isContiguous())
      output.resize(Array(outputHeight, outputWidth, nOutputPlane))
      if (_1x1) {
        fInput.set(input)
        fInput.resize(Array(outputHeight * outputWidth, kernelW * kernelH * nInputPlane))
      } else {
        fInput.resize(Array(outputHeight * outputWidth, kernelW * kernelH * nInputPlane))
      }
      updateOutputFrame(
        input,
        output,
        weightMM,
        bias,
        fInput,
        kernelW, kernelH, strideW, strideH,
        padW, padH,
        nInputPlane, inputWidth, inputHeight,
        nOutputPlane, outputWidth, outputHeight)
    } else {
      require(input.size(4) == nInputPlane)
      val batchSize = input.size(1)
      output.resize(Array(batchSize, outputHeight, outputWidth, nOutputPlane))
      if (_1x1) {
        fInput.set(input)
        fInput.resize(Array(batchSize, outputHeight * outputWidth, kernelW * kernelH * nInputPlane))
      } else {
        fInput.resize(Array(batchSize, outputHeight * outputWidth, kernelW * kernelH * nInputPlane))
      }

      if (results == null || results.length != batchSize) {
        results = new Array[Future[Unit]](batchSize)
      }

      var i = 0
      while (i < batchSize) {
        val _i = i + 1
        results(i) = Engine.model.invoke(() => {
          val inputT = input.select(1, _i)
          require(inputT.isContiguous())
          val outputT = output.select(1, _i)
          val fInputT = fInput.select(1, _i)
          updateOutputFrame(
            inputT,
            outputT,
            weightMM,
            bias,
            fInputT,
            kernelW, kernelH, strideW, strideH,
            padW, padH,
            nInputPlane, inputWidth, inputHeight,
            nOutputPlane, outputWidth, outputHeight)
        })
        i += 1
      }
      Engine.model.sync(results)
    }
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    if (!propagateBack) {
      return gradInput
    }

    require(input.nDimension() == 3 || input.nDimension() == 4, "Only support 3D or 4D input")
    gradInput.resizeAs(input)
    if (_1x1) {
      fGradInput.set(gradInput)
      fGradInput.resizeAs(fInput)
    } else {
      fGradInput.resizeAs(fInput)
    }

    if (input.nDimension() == 3) {
      require(gradOutput.isContiguous())
      updateGradInputFrame(
          gradInput,
          gradOutput,
          weightMM,
          fGradInput,
          kernelW, kernelH, strideW, strideH, padW, padH)
    } else {
      val batchSize = input.size(1)
      var i = 0
      while (i < batchSize) {
        val _i = i + 1
        results(i) = Engine.model.invoke(() => {
          val gradInputT = gradInput.select(1, _i)
          val gradOutputT = gradOutput.select(1, _i)
          require(gradOutputT.isContiguous())
          val fgradInputT = fGradInput.select(1, _i)
          updateGradInputFrame(
              gradInputT,
              gradOutputT,
              weightMM,
              fgradInputT,
              kernelW, kernelH, strideW, strideH, padW, padH)
        })
        i += 1
      }
      Engine.model.sync(results)
    }

    return gradInput
  }
/*
  override def accGradParameters(input: Tensor[T], gradOutput: Tensor[T],
    scale: Double = 1.0): Unit = {
    require(input.nDimension() == 3 || input.nDimension() == 4, "Only support 3D or 4D input")
    require(gradOutput.isContiguous())

    if (input.nDimension() == 3) {
      if (gradWeightMM == null) {
        gradWeightMM = gradWeight.view(nInputPlane * kernelH * kernelW, nOutputPlane)
      }
      accGradParametersFrame(
          gradOutput,
          gradWeightMM,
          gradBias,
          fInput,
          ev.fromType[Double](scale))
    } else {
      val batchSize = input.size(1)
      if (gradWeightMMInBatch == null) {
        gradWeightMMInBatch = Tensor[T]().resize(Array(batchSize,
          nInputPlane * kernelH * kernelW, nOutputPlane))
      }
      if(gradientBiasMT.nElement() == 0) {
        gradientBiasMT.resize(Array(batchSize, nOutputPlane))
      }
      if (ones.dim() != 1 || ones.size(1) != gradOutput.size(2) * gradOutput.size(3)) {
        ones.resize(Array(gradOutput.size(2) * gradOutput.size(3))).fill(ev.fromType(1.0))
      }

      if (onesBatch.dim() != 1 || onesBatch.size(1) != batchSize) {
        onesBatch.resize(Array(batchSize)).fill(ev.fromType(1.0))
      }
      var i = 0
      while (i < batchSize) {
        val _i = i + 1
        results(i) = Engine.model.invoke(() => {
          val gradOutputT = gradOutput.select(1, _i)
          val fInputT = fInput.select(1, _i)
          calcGradParametersFrame(
              gradOutputT,
              gradWeightMMInBatch.select(1, _i),
              gradientBiasMT.select(1, _i),
              fInputT,
              ev.fromType[Double](scale))
        })
        i += 1
      }

      Engine.model.sync(results)

      val gradView = gradWeightMMInBatch.view(batchSize,
        nOutputPlane * nInputPlane * kernelH * kernelW).t
      val grad = gradWeight.view(nOutputPlane * nInputPlane * kernelH * kernelW)
      grad.addmv(ev.fromType(1.0), ev.fromType(1.0), gradView, onesBatch)
      gradBias.addmv(ev.fromType(1.0), ev.fromType(1.0), gradientBiasMT.t, onesBatch)
    }

    if (null != wRegularizer) {
      //wRegularizer.accRegularization(weight, gradWeight)
    }
    if (null != bRegularizer) {
      //bRegularizer.accRegularization(bias, gradBias)
    }
  }
*/
  override def updateParameters(learningRate: T): Unit = {
    weight.map(gradWeight, (a, b) => ev.minus(a, ev.times(learningRate, b)))
    bias.map(gradBias, (a, b) => ev.minus(a, ev.times(learningRate, b)))
  }

  override def zeroGradParameters(): Unit = {
    gradWeight.zero()
    gradBias.zero()
  }

  override def parameters(): (Array[Tensor[T]], Array[Tensor[T]]) = {
    (Array(this.weight, this.bias), Array(this.gradWeight, this.gradBias))
  }

  override def getParametersTable(): Table = {
    T(getName() -> T("weight" -> weight, "bias" -> bias,
      "gradWeight" -> gradWeight, "gradBias" -> gradBias))
  }

  override def equals(obj: Any): Boolean = {

    if (!super.equals(obj)) {
      return false
    }

    if (!obj.isInstanceOf[SpatialConvolution[T]]) {
      return false
    }
    val other = obj.asInstanceOf[SpatialConvolution[T]]
    if (this.eq(other)) {
      return true
    }

    nInputPlane == other.nInputPlane &&
      nOutputPlane == other.nOutputPlane &&
      kernelW == other.kernelW &&
      kernelH == other.kernelH &&
      strideW == other.strideW &&
      strideH == other.strideH &&
      padW == other.padW &&
      padH == other.padH &&
      propagateBack == other.propagateBack &&
      weight == other.weight &&
      bias == other.bias &&
      gradWeight == other.gradWeight &&
      gradBias == other.gradBias
  }

  override def hashCode(): Int = {
    val seed = 37
    var hash = super.hashCode()
    hash = hash * seed + nInputPlane.hashCode()
    hash = hash * seed + nOutputPlane.hashCode()
    hash = hash * seed + kernelW.hashCode()
    hash = hash * seed + kernelH.hashCode()
    hash = hash * seed + strideW.hashCode()
    hash = hash * seed + strideH.hashCode()
    hash = hash * seed + padW.hashCode()
    hash = hash * seed + padH.hashCode()
    hash = hash * seed + weight.hashCode()
    hash = hash * seed + bias.hashCode()
    hash = hash * seed + gradWeight.hashCode()
    hash = hash * seed + gradBias.hashCode()

    hash
  }

  override def clearState() : this.type = {
    super.clearState()
    fInput.set()
    fGradInput.set()
    ones.set()
    onesBatch.set()
    onesBias.set()
    gradientBiasMT.set()
    this
  }

  override def toString(): String = {
    s"${getPrintName}($nInputPlane -> $nOutputPlane, $kernelW x" +
      s" $kernelH, $strideW, $strideH, $padW, $padH)"
  }

  protected def updateOutputFrame(input: Tensor[T], output: Tensor[T], weight: Tensor[T],
    bias: Tensor[T], fInput: Tensor[T],
    kW: Int, kH: Int, dW: Int, dH: Int, padW: Int, padH: Int,
    nInputPlane: Int, inputWidth: Int, inputHeight: Int,
    nOutputPlane: Int, outputWidth: Int, outputHeight: Int)(
    implicit ev: TensorNumeric[T]): Unit = {

    val output2d = output.view(outputHeight * outputWidth, nOutputPlane)
    if (!_1x1) {
      ev.getType() match {
        case DoubleType =>
          val before = System.nanoTime()
          NNPrimitive.im2colDoubleNHWC(fInput.asInstanceOf[Tensor[Double]],
            input.asInstanceOf[Tensor[Double]], kW, kH, dW, dH, padW, padH, nInputPlane,
            inputWidth, inputHeight, outputWidth, outputHeight)
          im2colTime += System.nanoTime() - before
        case FloatType =>
          val before = System.nanoTime()
          NNPrimitive.im2colFloatNHWC(fInput.asInstanceOf[Tensor[Float]],
            input.asInstanceOf[Tensor[Float]], kW, kH, dW, dH, padW, padH, nInputPlane,
            inputWidth, inputHeight, outputWidth, outputHeight)
          im2colTime += System.nanoTime() - before
        case _ => throw new UnsupportedOperationException(s"Only Float/Double supported")
      }
    }
    output2d.addmm(ev.fromType[Int](0), output2d, ev.fromType[Int](1), fInput, weight)
    output2d.addr(ev.fromType(1), onesBias, bias)
  }

  protected def updateGradInputFrame(gradInput: Tensor[T], gradOutput: Tensor[T],
    weight: Tensor[T], fgradInput: Tensor[T], kW: Int, kH: Int, dW: Int, dH: Int,
    padW: Int, padH: Int)(implicit ev: TensorNumeric[T]): Unit = {
    ev.getType() match {
      case DoubleType =>
        val gradOutput2d = Tensor(gradOutput.storage().asInstanceOf[Storage[Double]],
          gradOutput.storageOffset(), Array(gradOutput.size(1) * gradOutput.size(2),
            gradOutput.size(3)))
        fgradInput.asInstanceOf[Tensor[Double]].addmm(0.0, fgradInput.asInstanceOf[Tensor[Double]],
          1.0, gradOutput2d, weight.asInstanceOf[Tensor[Double]].transpose(1, 2))
        if (!_1x1) {
          gradInput.asInstanceOf[Tensor[Double]].zero()
          val before = System.nanoTime()
          NNPrimitive.col2imDoubleNHWC(fgradInput.asInstanceOf[Tensor[Double]],
            gradInput.asInstanceOf[Tensor[Double]], kW, kH, dW, dH, padW, padH, gradInput.size(3),
            gradInput.size(2),
            gradInput.size(1), gradOutput.size(2), gradOutput.size(1))
          col2imTime += System.nanoTime() - before
        }
      case FloatType =>
        val gradOutput2d = Tensor(gradOutput.storage().asInstanceOf[Storage[Float]],
          gradOutput.storageOffset(),
          Array(gradOutput.size(1) * gradOutput.size(2), gradOutput.size(3)))
        fgradInput.asInstanceOf[Tensor[Float]].addmm(0.0f, fgradInput.asInstanceOf[Tensor[Float]],
          1.0f, gradOutput2d, weight.asInstanceOf[Tensor[Float]].transpose(1, 2))
        if (!_1x1) {
          gradInput.asInstanceOf[Tensor[Float]].zero()
          val before = System.nanoTime()
          NNPrimitive.col2imFloatNHWC(fgradInput.asInstanceOf[Tensor[Float]],
            gradInput.asInstanceOf[Tensor[Float]], kW, kH, dW, dH, padW, padH, gradInput.size(3),
            gradInput.size(2),
            gradInput.size(1), gradOutput.size(2), gradOutput.size(1))
          col2imTime += System.nanoTime() - before
        }
      case _ => throw new UnsupportedOperationException(s"Only Float/Double supported")
    }
  }

  protected def accGradParametersFrame(gradOutput: Tensor[T], gradWeight: Tensor[T],
    gradBias: Tensor[T], fInput: Tensor[T], scale: T)(implicit ev: TensorNumeric[T]): Unit = {

    val gradOutput2d = gradOutput
      .view(Array(gradOutput.size(1) * gradOutput.size(2), gradOutput.size(3)))

    gradWeight.addmm(ev.fromType(1.0), gradWeight, scale, fInput.t, gradOutput2d)
    var i = 0
    while (i < gradBias.size(1)) {
      var sum: T = ev.fromType(0.0)
      val data = gradOutput2d.storage().array()
      val offset = gradOutput2d.storageOffset() - 1 + i * gradOutput2d.stride(1)
      var k = 0
      while (k < gradOutput2d.size(2)) {
        sum = ev.plus(sum, data(k + offset))
        k += 1
      }
      gradBias.setValue(i + 1,
        ev.plus(gradBias.valueAt(i + 1), ev.times(scale, sum)))
      i += 1
    }
  }

  protected def calcGradParametersFrame(gradOutput: Tensor[T], gradWeight: Tensor[T],
    gradBias: Tensor[T],
    fInput: Tensor[T], scale: T)(implicit ev: TensorNumeric[T]): Unit = {

    val gradOutput2d = gradOutput
      .view(Array(gradOutput.size(1) * gradOutput.size(2), gradOutput.size(3)))

    gradWeight.addmm(ev.fromType(0.0), gradWeight, scale, fInput.t, gradOutput2d)

    gradBias.addmv(ev.fromType(0.0), ev.fromType(1.0), gradOutput2d.t, ones)
  }
}

object SpatialConvolutionNHWC {
  def apply[@specialized(Float, Double) T: ClassTag](
      nInputPlane: Int,
      nOutputPlane: Int,
      kernelW: Int,
      kernelH: Int,
      strideW: Int = 1,
      strideH: Int = 1,
      padW: Int = 0,
      padH: Int = 0,
      propagateBack: Boolean = true,
      wRegularizer: Regularizer[T] = null,
      bRegularizer: Regularizer[T] = null,
      initWeight: Tensor[T] = null,
      initBias: Tensor[T] = null,
      initGradWeight: Tensor[T] = null,
      initGradBias: Tensor[T] = null
  )(implicit ev: TensorNumeric[T]): SpatialConvolutionNHWC[T] = {
    new SpatialConvolutionNHWC[T](nInputPlane, nOutputPlane, kernelW, kernelH,
      strideW, strideH, padW, padH, propagateBack,
      wRegularizer, bRegularizer, initWeight, initBias, initGradWeight, initGradBias)
  }
}
