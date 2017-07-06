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

import breeze.optimize.BatchSize
import com.intel.analytics.bigdl.nn.abstractnn.{Initializable, TensorModule}
import com.intel.analytics.bigdl.optim.Regularizer
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor._
import com.intel.analytics.bigdl.utils._
import com.intel.analytics.bigdl.utils.RandomGenerator._

import scala.concurrent.duration.Duration
import scala.concurrent.{Await, Future}
import scala.reflect.ClassTag

sealed trait InputFormat
object InputFormat {
  case object NHWC extends InputFormat
  case object NCHW extends InputFormat
}

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

@SerialVersionUID(- 8446523046224797382L)
class SpatialConvolution[T: ClassTag](
  val nInputPlane: Int, // The number of expected input planes in the image given into forward()
  val nOutputPlane: Int, // The number of output planes the convolution layer will produce.
  val kernelW: Int, // The kernel width of the convolution
  val kernelH: Int, // The kernel height of the convolution
  val strideW: Int = 1, // The step of the convolution in the width dimension.
  val strideH: Int = 1, // The step of the convolution in the height dimension
  val padW: Int = 0, // The additional zeros added per width to the input planes.
  val padH: Int = 0, // The additional zeros added per height to the input planes.
  val nGroup: Int = 1, // Kernel group number
  val propagateBack: Boolean = true, // propagate gradient back
  var wRegularizer: Regularizer[T] = null,
  var bRegularizer: Regularizer[T] = null,
  val initWeight: Tensor[T] = null,
  val initBias: Tensor[T] = null,
  val initGradWeight: Tensor[T] = null,
  val initGradBias: Tensor[T] = null,
  format: InputFormat = InputFormat.NCHW
)(implicit ev: TensorNumeric[T]) extends TensorModule[T] with Initializable {

  require(nInputPlane % nGroup == 0, "Number of input channels should be multiples of group.")
  require(nOutputPlane % nGroup == 0, "Number of output channels should be multiples of group.")
  if (nGroup != 1) {
    require(format == InputFormat.NCHW, "group convolution is not supported in NHWC format " )
  }

  private val weightShape = format match {
    case InputFormat.NCHW =>
      Array(nGroup, nOutputPlane / nGroup, nInputPlane / nGroup, kernelH, kernelW)
    case InputFormat.NHWC =>
      Array(kernelH, kernelW, nInputPlane, nOutputPlane)
  }

  private val weightFormat = format match {
    case InputFormat.NCHW =>
      VariableFormat.GP_OUT_IN_KW_KH
    case InputFormat.NHWC =>
      VariableFormat.KH_KW_IN_OUT
  }

  private val weightMMShape = format match {
    case InputFormat.NCHW =>
      Array(nGroup, nOutputPlane / nGroup, nInputPlane * kernelH * kernelW / nGroup)
    case InputFormat.NHWC =>
      Array(1, nInputPlane * kernelH * kernelW, nOutputPlane)
  }

  val weight: Tensor[T] = if (initWeight != null) {
    initWeight
  } else {
    Tensor[T](weightShape)
  }

  val bias: Tensor[T] = if (initBias != null) initBias else Tensor[T](nOutputPlane)

  val gradWeight: Tensor[T] = if (initGradWeight != null) {
    initGradWeight
  } else {
    Tensor[T](weightShape)
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
      weightInitMethod.init(weight, weightFormat)
    }
    if (initBias == null) {
      biasInitMethod.init(bias, VariableFormat.ONE_D)
    }
    zeroGradParameters()
  }

  @inline
  private def getHWCDims(inputDim: Int): (Int, Int, Int) = {
    format match {
      case InputFormat.NCHW =>
        if (inputDim == 3) (2, 3, 1) else (3, 4, 2)
      case InputFormat.NHWC =>
        if (inputDim == 3) (1, 2, 3) else (2, 3, 4)
    }
  }

  private def getOutputShape(oh: Int, ow: Int, batchSize: Int = -1): Array[Int] = {
    format match {
      case InputFormat.NCHW =>
        if (batchSize == -1) {
          Array(nOutputPlane, oh, ow)
        } else {
          Array(batchSize, nOutputPlane, oh, ow)
        }
      case InputFormat.NHWC =>
        if (batchSize == -1) {
          Array(oh, ow, nOutputPlane)
        } else {
          Array(batchSize, oh, ow, nOutputPlane)
        }

    }
  }

  private def getFInputShape(oh: Int, ow: Int, batchSize: Int = -1): Array[Int] = {
    format match {
      case InputFormat.NCHW =>
        if (batchSize == -1) {
          Array(nGroup, kernelW * kernelH * nInputPlane / nGroup, oh * ow)
        } else {
          Array(batchSize, nGroup, kernelW * kernelH * nInputPlane / nGroup, oh * ow)
        }
      case InputFormat.NHWC =>
        if (batchSize == -1) {
          Array(1, oh * ow, kernelW * kernelH * nInputPlane)
        } else {
          Array(batchSize, 1, oh * ow, kernelW * kernelH * nInputPlane)
        }

    }
  }

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    require(input.dim() == 3 || input.dim() == 4,
      "SpatialConvolution: " + ErrorInfo.constrainInputAs3DOrBatch)
    require(input.isContiguous())

    if (weightMM == null || weightMM.storage().isEmpty) {
      weightMM = weight.view(weightMMShape)
    }

    val (dimHeight, dimWidth, channelDim) = getHWCDims(input.dim())
    require(input.size(channelDim) == nInputPlane, s"input channel size " +
      s"${input.size(channelDim)} is not the same as nInputPlane $nInputPlane")

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
      require(input.isContiguous())
      output.resize(getOutputShape(outputHeight, outputWidth))
      if (_1x1) {
        fInput.set(input)
        fInput.resize(getFInputShape(outputHeight, outputWidth))
      } else {
        fInput.resize(getFInputShape(outputHeight, outputWidth))
      }
      var g = 0
      while (g < nGroup) {
        updateOutputFrame(
          input.narrow(channelDim, g * nInputPlane / nGroup + 1, nInputPlane / nGroup),
          output.narrow(channelDim, g * nOutputPlane / nGroup + 1, nOutputPlane / nGroup),
          weightMM.select(1, g + 1),
          bias.narrow(1, g * nOutputPlane / nGroup + 1, nOutputPlane / nGroup),
          fInput.select(1, g + 1),
          kernelW, kernelH, strideW, strideH,
          padW, padH,
          nInputPlane / nGroup, inputWidth, inputHeight,
          nOutputPlane / nGroup, outputWidth, outputHeight)
        g += 1
      }
    } else {
      val batchSize = input.size(1)
      output.resize(getOutputShape(outputHeight, outputWidth, batchSize))
      if (_1x1) {
        fInput.set(input)
        fInput.resize(getFInputShape(outputHeight, outputWidth, batchSize))
      } else {
        fInput.resize(getFInputShape(outputHeight, outputWidth, batchSize))
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
          var g = 0
          while (g < nGroup) {
            updateOutputFrame(
              inputT.narrow(channelDim - 1, g * nInputPlane / nGroup + 1, nInputPlane / nGroup),
              outputT.narrow(channelDim - 1, g * nOutputPlane / nGroup + 1, nOutputPlane / nGroup),
              weightMM.select(1, g + 1),
              bias.narrow(1, g * nOutputPlane / nGroup + 1, nOutputPlane / nGroup),
              fInputT.select(1, g + 1),
              kernelW, kernelH, strideW, strideH,
              padW, padH,
              nInputPlane / nGroup, inputWidth, inputHeight,
              nOutputPlane / nGroup, outputWidth, outputHeight)
            g += 1
          }
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

    val (ohDim, owDim, cDim) = getHWCDims(input.dim())
    val oh = gradOutput.size(ohDim)
    val ow = gradOutput.size(owDim)

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
      var g = 0
      while (g < nGroup) {
        updateGradInputFrame(
          gradInput.narrow(cDim, g * nInputPlane / nGroup + 1, nInputPlane / nGroup),
          gradOutput.narrow(cDim, g * nOutputPlane / nGroup + 1, nOutputPlane / nGroup),
          weightMM.select(1, g + 1).transpose(1, 2),
          fGradInput.select(1, g + 1),
          kernelW, kernelH, strideW, strideH, padW, padH, oh, ow)
        g += 1
      }
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
          var g = 0
          while (g < nGroup) {
            updateGradInputFrame(
              gradInputT.narrow(cDim - 1, g * nInputPlane / nGroup + 1, nInputPlane / nGroup),
              gradOutputT.narrow(cDim - 1, g * nOutputPlane / nGroup + 1, nOutputPlane / nGroup),
              weightMM.select(1, g + 1).transpose(1, 2),
              fgradInputT.select(1, g + 1),
              kernelW, kernelH, strideW, strideH, padW, padH, oh, ow)
            g += 1
          }
        })
        i += 1
      }
      Engine.model.sync(results)
    }

    gradInput
  }

  private def getGradWeightMMInBatchShape(batchSize: Int) = format match {
    case InputFormat.NCHW =>
      Array(batchSize, nGroup, nOutputPlane / nGroup, nInputPlane * kernelH * kernelW / nGroup)
    case InputFormat.NHWC =>
      Array(batchSize, 1, nInputPlane * kernelH * kernelW, nOutputPlane)
  }

  override def accGradParameters(input: Tensor[T], gradOutput: Tensor[T]): Unit = {
    require(input.nDimension() == 3 || input.nDimension() == 4, "Only support 3D or 4D input")
    require(gradOutput.isContiguous())

    val (ohDim, owDim, cDim) = getHWCDims(input.dim())
    val oh = gradOutput.size(ohDim)
    val ow = gradOutput.size(owDim)

    if (input.nDimension() == 3) {
      if (gradWeightMM == null) {
        gradWeightMM = gradWeight.view(weightMMShape)
      }
      var g = 0
      while (g < nGroup) {
        accGradParametersFrame(
          gradOutput.narrow(cDim, g * nOutputPlane / nGroup + 1, nOutputPlane / nGroup),
          gradWeightMM.select(1, g + 1),
          gradBias.narrow(1, g * nOutputPlane / nGroup + 1, nOutputPlane / nGroup),
          fInput.select(1, g + 1),
          ev.fromType[Double](scaleW),
          ev.fromType[Double](scaleB))
        g += 1
      }
    } else {
      val batchSize = input.size(1)
      if (gradWeightMMInBatch == null) {
        gradWeightMMInBatch = Tensor[T]().resize(getGradWeightMMInBatchShape(batchSize))
      }
      if(gradientBiasMT.nElement() == 0) {
        gradientBiasMT.resize(Array(batchSize, nOutputPlane))
      }
      if (ones.dim() != 1 || ones.size(1) != oh * ow) {
        ones.resize(Array(oh * ow)).fill(ev.fromType(1.0))
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
          var g = 0
          while (g < nGroup) {
            calcGradParametersFrame(
              gradOutputT.narrow(cDim - 1, g * nOutputPlane / nGroup + 1, nOutputPlane / nGroup),
              gradWeightMMInBatch.select(1, _i).select(1, g + 1),
              gradientBiasMT.select(1, _i).narrow(1, g * nOutputPlane / nGroup + 1,
                nOutputPlane / nGroup),
              fInputT.select(1, g + 1),
              ev.fromType[Double](scaleW),
              ev.fromType[Double](scaleB))
            g += 1
          }
        })
        i += 1
      }

      Engine.model.sync(results)

      val gradView = gradWeightMMInBatch.view(batchSize,
        nOutputPlane * nInputPlane * kernelH * kernelW / nGroup).t
      val grad = gradWeight.view(nOutputPlane * nInputPlane * kernelH * kernelW / nGroup)
      grad.addmv(ev.fromType(1.0), ev.fromType(1.0), gradView, onesBatch)
      gradBias.addmv(ev.fromType(1.0), ev.fromType(1.0), gradientBiasMT.t, onesBatch)
    }

    if (null != wRegularizer) {
      wRegularizer.accRegularization(weight, gradWeight, scaleW)
    }
    if (null != bRegularizer) {
      bRegularizer.accRegularization(bias, gradBias, scaleB)
    }
  }

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
      nGroup == other.nGroup &&
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

    format match {
      case InputFormat.NCHW =>
        val output2d = output.view(nOutputPlane, outputHeight * outputWidth)
        if (!_1x1) {
          ev.getType() match {
            case DoubleType =>
              val before = System.nanoTime()
              NNPrimitive.im2colDouble(fInput.asInstanceOf[Tensor[Double]],
                input.asInstanceOf[Tensor[Double]], kW, kH, dW, dH, padW, padH, nInputPlane,
                inputWidth, inputHeight, outputWidth, outputHeight)
              im2colTime += System.nanoTime() - before
            case FloatType =>
              val before = System.nanoTime()
              NNPrimitive.im2colFloat(fInput.asInstanceOf[Tensor[Float]],
                input.asInstanceOf[Tensor[Float]], kW, kH, dW, dH, padW, padH, nInputPlane,
                inputWidth, inputHeight, outputWidth, outputHeight)
              im2colTime += System.nanoTime() - before
            case _ => throw new UnsupportedOperationException(s"Only Float/Double supported")
          }
        }
        output2d.addmm(ev.fromType[Int](0), output2d, ev.fromType[Int](1), weight, fInput)
        output2d.addr(ev.fromType(1), bias, onesBias)
      case InputFormat.NHWC =>
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

  }

  protected def updateGradInputFrame(gradInput: Tensor[T], gradOutput: Tensor[T],
     weight: Tensor[T], fgradInput: Tensor[T], kW: Int, kH: Int, dW: Int, dH: Int,
     padW: Int, padH: Int, oh: Int, ow: Int)(implicit ev: TensorNumeric[T]): Unit = {
    ev.getType() match {
      case DoubleType =>
        val gradOutDouble = gradOutput.asInstanceOf[Tensor[Double]]
        val fGradInDouble = fgradInput.asInstanceOf[Tensor[Double]]
        val weightDouble = weight.asInstanceOf[Tensor[Double]]
        val gradInputDouble = gradInput.asInstanceOf[Tensor[Double]]
        format match {
          case InputFormat.NCHW =>
            val gradOutput2d = gradOutDouble.view(Array(nOutputPlane, oh * ow))
            fGradInDouble.addmm(0.0, fGradInDouble, 1.0, weightDouble, gradOutput2d)
            if (!_1x1) {
              gradInputDouble.zero()
              val before = System.nanoTime()
              NNPrimitive.col2imDouble(fGradInDouble,
                gradInputDouble, kW, kH, dW, dH, padW, padH, gradInput.size(1),
                gradInput.size(3), gradInput.size(2),
                gradOutput.size(3), gradOutput.size(2))
              col2imTime += System.nanoTime() - before
            }
          case InputFormat.NHWC =>
            val gradOutput2d = gradOutDouble.view(Array(oh * ow, nOutputPlane))
            fGradInDouble.addmm(0.0, fGradInDouble, 1.0, gradOutput2d, weightDouble)
            if (!_1x1) {
              gradInputDouble.zero()
              val before = System.nanoTime()
              NNPrimitive.col2imDoubleNHWC(fGradInDouble,
                gradInputDouble, kW, kH, dW, dH, padW, padH, gradInput.size(3),
                gradInput.size(2), gradInput.size(1),
                gradOutput.size(2), gradOutput.size(1))
              col2imTime += System.nanoTime() - before
            }
        }
      case FloatType =>
        val gradOutFloat = gradOutput.asInstanceOf[Tensor[Float]]
        val fGradInFloat = fgradInput.asInstanceOf[Tensor[Float]]
        val weightFloat = weight.asInstanceOf[Tensor[Float]]
        val gradInputFloat = gradInput.asInstanceOf[Tensor[Float]]
        format match {
          case InputFormat.NCHW =>
            val gradOutput2d = gradOutFloat.view(Array(nOutputPlane, oh * ow))
            fGradInFloat.addmm(0.0f, fGradInFloat, 1.0f, weightFloat, gradOutput2d)
            if (!_1x1) {
              gradInputFloat.zero()
              val before = System.nanoTime()
              NNPrimitive.col2imFloat(fGradInFloat,
                gradInputFloat, kW, kH, dW, dH, padW, padH, gradInput.size(1),
                gradInput.size(3), gradInput.size(2),
                gradOutput.size(3), gradOutput.size(2))
              col2imTime += System.nanoTime() - before
            }
          case InputFormat.NHWC =>
            val gradOutput2d = gradOutFloat.view(Array(oh * ow, nOutputPlane))
            fGradInFloat.addmm(0.0f, fGradInFloat, 1.0f, gradOutput2d, weightFloat)
            if (!_1x1) {
              gradInputFloat.zero()
              val before = System.nanoTime()
              NNPrimitive.col2imFloatNHWC(fGradInFloat,
                gradInputFloat, kW, kH, dW, dH, padW, padH, gradInput.size(3),
                gradInput.size(2), gradInput.size(1),
                gradOutput.size(2), gradOutput.size(1))
              col2imTime += System.nanoTime() - before
            }
        }
      case _ => throw new UnsupportedOperationException(s"Only Float/Double supported")
    }
  }

  protected def accGradParametersFrame(gradOutput: Tensor[T], gradWeight: Tensor[T],
    gradBias: Tensor[T], fInput: Tensor[T],
    scaleW: T, scaleB: T)(implicit ev: TensorNumeric[T]): Unit = {

    ev.getType() match {
      case DoubleType =>
        val gradODouble = gradOutput.asInstanceOf[Tensor[Double]]
        val gradWDouble = gradWeight.asInstanceOf[Tensor[Double]]
        val fIDouble = fInput.asInstanceOf[Tensor[Double]]
        val sWDouble = ev.toType[Double](scaleW)
        val sBDouble = ev.toType[Double](scaleB)
        val gradBDouble = gradBias.asInstanceOf[Tensor[Double]]
        format match {
          case InputFormat.NCHW =>
            val outChannel = gradOutput.size(1)
            val outSize = gradOutput.size(2) * gradOutput.size(3)
            val gradOutput2d = gradODouble.view(Array(outChannel, outSize))
            if (sWDouble != 0) {
              gradWDouble.addmm(1.0, gradWDouble, sWDouble, gradOutput2d, fIDouble.t)
            }

            if (sBDouble != 0) {
              var i = 0
              while (i < gradBias.size(1)) {
                var sum = 0.0
                val data = gradOutput2d.storage().array()
                val offset = gradOutput2d.storageOffset() - 1 + i * gradOutput2d.stride(1)
                var k = 0
                while (k < gradOutput2d.size(2)) {
                  sum += data(k + offset)
                  k += 1
                }
                gradBDouble.setValue(i + 1, gradBDouble.valueAt(i + 1) + (sBDouble * sum))
                i += 1
              }
            }
          case InputFormat.NHWC =>
            val outChannel = gradOutput.size(3)
            val outSize = gradOutput.size(1) * gradOutput.size(2)
            val gradOutput2d = gradODouble.view(Array(outSize, outChannel))

            if (sWDouble != 0) {
              gradWDouble.addmm(1.0, gradWDouble, sWDouble, fIDouble.t, gradOutput2d)
            }

            if (sBDouble != 0) {
              var i = 0
              val gradData = gradOutput2d.storage().array()
              val biasData = gradBDouble.storage().array()
              val biasOffset = gradBDouble.storageOffset() - 1

              while (i < gradODouble.size(1)) {
                val gradOffset = gradOutput2d.storageOffset() - 1 + i * gradOutput2d.stride(1)
                var j = 0
                while (j < gradOutput2d.size(2)) {
                  biasData(biasOffset + j) += gradData(gradOffset + j)
                  j = j + 1
                }
                i = i + 1
              }
            }
        }

      case FloatType =>
        val gradOFloat = gradOutput.asInstanceOf[Tensor[Float]]
        val gradWFloat = gradWeight.asInstanceOf[Tensor[Float]]
        val fIFloat = fInput.asInstanceOf[Tensor[Float]]
        val sWFloat = ev.toType[Float](scaleW)
        val sBFloat = ev.toType[Float](scaleB)
        val gradBFloat = gradBias.asInstanceOf[Tensor[Float]]
        format match {
          case InputFormat.NCHW =>
            val outChannel = gradOutput.size(1)
            val outSize = gradOutput.size(2) * gradOutput.size(3)
            val gradOutput2d = gradOFloat.view(Array(outChannel, outSize))
            if (sWFloat != 0) {
              gradWFloat.addmm(1.0f, gradWFloat, sWFloat, gradOutput2d, fIFloat.t)
            }

            if (sBFloat != 0) {
              var i = 0
              while (i < gradBias.size(1)) {
                var sum = 0.0f
                val data = gradOutput2d.storage().array()
                val offset = gradOutput2d.storageOffset() - 1 + i * gradOutput2d.stride(1)
                var k = 0
                while (k < gradOutput2d.size(2)) {
                  sum += data(k + offset)
                  k += 1
                }
                gradBFloat.setValue(i + 1, gradBFloat.valueAt(i + 1) + (sBFloat * sum))
                i += 1
              }
            }
          case InputFormat.NHWC =>
            val outChannel = gradOutput.size(3)
            val outSize = gradOutput.size(1) * gradOutput.size(2)
            val gradOutput2d = gradOFloat.view(Array(outSize, outChannel))

            if (sWFloat != 0) {
              gradWFloat.addmm(1.0f, gradWFloat, sWFloat, fIFloat.t, gradOutput2d)
            }

            if (sBFloat != 0) {
              var i = 0
              val gradData = gradOutput2d.storage().array()
              val biasData = gradBFloat.storage().array()
              val biasOffset = gradBFloat.storageOffset() - 1

              while (i < gradOFloat.size(1)) {
                val gradOffset = gradOutput2d.storageOffset() - 1 + i * gradOutput2d.stride(1)
                var j = 0
                while (j < gradOutput2d.size(2)) {
                  biasData(biasOffset + j) += gradData(gradOffset + j)
                  j = j + 1
                }
                i = i + 1
              }
            }
        }

      case _ => throw new UnsupportedOperationException(s"Only Float/Double supported")
    }
  }

  protected def calcGradParametersFrame(gradOutput: Tensor[T], gradWeight: Tensor[T],
    gradBias: Tensor[T],
    fInput: Tensor[T], scaleW: T, scaleB: T)(implicit ev: TensorNumeric[T]): Unit = {

    ev.getType() match {
      case DoubleType =>
        val gradODouble = gradOutput.asInstanceOf[Tensor[Double]]
        val gradWDouble = gradWeight.asInstanceOf[Tensor[Double]]
        val sWDouble = ev.toType[Double](scaleW)
        val sBDouble = ev.toType[Double](scaleB)
        val fIDouble = fInput.asInstanceOf[Tensor[Double]]
        val gradBDouble = gradBias.asInstanceOf[Tensor[Double]]
        val onesDouble = ones.asInstanceOf[Tensor[Double]]

        format match {
          case InputFormat.NCHW =>
            val channel = gradODouble.size(1)
            val oh = gradODouble.size(2)
            val ow = gradODouble.size(3)
            val gradOutput2d = gradODouble.view(Array(channel, oh * ow))

            if (scaleW != 0) {
              gradWDouble.addmm(0.0, gradWDouble, sWDouble, gradOutput2d, fIDouble.t)
            }

            if (scaleB != 0) {
              gradBDouble.addmv(0.0, sBDouble, gradOutput2d, onesDouble)
            }

          case InputFormat.NHWC =>
            val channel = gradODouble.size(3)
            val oh = gradODouble.size(1)
            val ow = gradODouble.size(2)
            val gradOutput2d = gradODouble.view(Array(oh * ow, channel))

            if (scaleW != 0) {
              gradWDouble.addmm(0.0, gradWDouble, sWDouble, fIDouble.t, gradOutput2d)
            }

            if (scaleB != 0) {
              gradBDouble.addmv(0.0, sBDouble, gradOutput2d.t, onesDouble)
            }
        }

      case FloatType =>
        val gradOFloat = gradOutput.asInstanceOf[Tensor[Float]]
        val gradWFloat = gradWeight.asInstanceOf[Tensor[Float]]
        val sWFloat = ev.toType[Float](scaleW)
        val sBFloat = ev.toType[Float](scaleB)
        val fIFloat = fInput.asInstanceOf[Tensor[Float]]
        val gradBFloat = gradBias.asInstanceOf[Tensor[Float]]
        val onesFloat = ones.asInstanceOf[Tensor[Float]]

        format match {
          case InputFormat.NCHW =>
            val channel = gradOFloat.size(1)
            val oh = gradOFloat.size(2)
            val ow = gradOFloat.size(3)
            val gradOutput2d = gradOFloat.view(Array(channel, oh * ow))

            if (scaleW != 0) {
              gradWFloat.addmm(0.0f, gradWFloat, sWFloat, gradOutput2d, fIFloat.t)
            }

            if (scaleB != 0) {
              gradBFloat.addmv(0.0f, sBFloat, gradOutput2d, onesFloat)
            }

          case InputFormat.NHWC =>
            val channel = gradOFloat.size(3)
            val oh = gradOFloat.size(1)
            val ow = gradOFloat.size(2)
            val gradOutput2d = gradOFloat.view(Array(oh * ow, channel))

            if (scaleW != 0) {
              gradWFloat.addmm(0.0f, gradWFloat, sWFloat, fIFloat.t, gradOutput2d)
            }

            if (scaleB != 0) {
              gradBFloat.addmv(0.0f, sBFloat, gradOutput2d.t, onesFloat)
            }
        }

      case _ => throw new UnsupportedOperationException(s"Only Float/Double supported")
    }
  }
}

object SpatialConvolution {
  def apply[@specialized(Float, Double) T: ClassTag](
      nInputPlane: Int,
      nOutputPlane: Int,
      kernelW: Int,
      kernelH: Int,
      strideW: Int = 1,
      strideH: Int = 1,
      padW: Int = 0,
      padH: Int = 0,
      nGroup: Int = 1,
      propagateBack: Boolean = true,
      wRegularizer: Regularizer[T] = null,
      bRegularizer: Regularizer[T] = null,
      initWeight: Tensor[T] = null,
      initBias: Tensor[T] = null,
      initGradWeight: Tensor[T] = null,
      initGradBias: Tensor[T] = null
  )(implicit ev: TensorNumeric[T]): SpatialConvolution[T] = {
    new SpatialConvolution[T](nInputPlane, nOutputPlane, kernelW, kernelH,
      strideW, strideH, padW, padH, nGroup, propagateBack,
      wRegularizer, bRegularizer, initWeight, initBias, initGradWeight, initGradBias)
  }
}
