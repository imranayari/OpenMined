using UnityEngine;
using System;
using System.Threading.Tasks;
using System.Collections.Generic;
using System.Linq;
using System.Xml.Schema;
using UnityEditor.Experimental.Build.AssetBundle;

namespace OpenMined.Syft.Tensor
{
    public partial class FloatTensor
    {
        internal FloatTensor emptyTensorCopy()
        {
//			FloatTensor result = new FloatTensor(ctrl, _shape:shape, _data:data, _dataBuffer:dataBuffer, _shader:this.shader);
//			return new FloatTensor(ctrl, _shape:shape, _dataOnGpu:dataOnGpu, _shader:shader);
            return Copy();
        }

        public FloatTensor Abs(bool inline = false)
// Returns a new Tensor with the smallest integer greater than or equal to each element
        {
            var result = inline ? this : this.emptyTensorCopy();

            if (dataOnGpu)
            {
                if (!inline) return AbsGPU(result);
                AbsGPU_();
                return this;
            }

            result.Data = data.AsParallel().Select(x => (float) Math.Abs((double) x)).ToArray();
            return result;
        }

        public FloatTensor Add(FloatTensor x, bool inline = false)
        {
            // Check if both tensors are compatible for sum
            SameSizeDimensionsShapeAndLocation(ref x);

            var result = inline ? this : this.emptyTensorCopy();
            if (dataOnGpu & x.dataOnGpu)
            {
                if (inline)
                {
                    if (autograd)
                        throw new InvalidOperationException(
                            "Cannot call inline functions if you intend to run backprop.");

                    AddElemGPU_(x);
                    return this;
                }
                result = AddElemGPU(x, result);
            }
            else
            {
                result.Data = data.AsParallel().Zip(x.Data.AsParallel(), (a, b) => a + b).ToArray();
            }


            if (autograd)
            {
                HookAutograd(ref result, ref x, "add_elem");
            }


            return result;
        }


        public FloatTensor Acos(bool inline = false)
        {
            if (dataOnGpu)
            {
                if (!inline) return AcosGPU();
                AcosGPU_();
                return this;
            }
            var result = inline ? this : this.emptyTensorCopy();
            result.Data = data.AsParallel().Select(x => (float) Math.Acos((double) x)).ToArray();
            return result;
        }

        public FloatTensor Asin(bool inline = false)
        {
            if (dataOnGpu)
            {
                if (!inline) return AsinGPU();
                AsinGPU_();
                return this;
            }
            var result = inline ? this : this.emptyTensorCopy();
            result.Data = data.AsParallel().Select(x => (float) Math.Asin((double) x)).ToArray();
            return result;
        }

        public FloatTensor Atan(bool inline = false)
        {
            if (dataOnGpu)
            {
                if (!inline) return AtanGPU();
                AtanGPU_();
                return this;
            }
            var result = inline ? this : this.emptyTensorCopy();
            result.Data = data.AsParallel().Select(x => (float) Math.Atan((double) x)).ToArray();
            return result;
        }

        public FloatTensor Add(float value, bool inline = false)
        {
            var result = inline ? this : this.emptyTensorCopy();

            if (dataOnGpu)
            {
                result.Gpu(shader);
                if (!inline) return AddScalarGPU(value, result);
                AddScalarGPU_(value);
                return this;
            }
            result.Data = data.AsParallel().Select(x => x + value).ToArray();
            return result;
        }

        public FloatTensor AddMatrixMultiply(FloatTensor tensor1, FloatTensor tensor2)
        {
            var gpu = dataOnGpu & tensor1.DataOnGpu & tensor2.DataOnGpu;
            var cpu = !(dataOnGpu | tensor1.DataOnGpu | tensor2.DataOnGpu);

            var res_shape = this.Shape;
            var shape1 = tensor1.Shape;
            var shape2 = tensor2.Shape;

            if (shape1[1] != shape2[0])
                throw new InvalidOperationException(String.Format("Matrix multiply not possible: {0} & {1}.", shape1[1],
                    shape2[0]));
            if (res_shape[0] != shape1[0])
                throw new InvalidOperationException(String.Format("First dimension doesn't match: {0} vs {1}.",
                    res_shape[0], shape1[0]));
            if (res_shape[1] != shape2[1])
                throw new InvalidOperationException(String.Format("Last dimension doesn't match: {0} vs {1}.",
                    res_shape[res_shape.Length - 1], shape2[shape2.Length - 1]));

            if (gpu)
            {
                AddMatrixMultiplyGPU(tensor1, tensor2);
            }
            else if (cpu)
            {
                var nCpu = SystemInfo.processorCount;
                Parallel.For(0, nCpu, workerId =>
                {
                    var max = size * (workerId + 1) / nCpu;
                    for (var idx = size * workerId / nCpu; idx < max; idx++)
                    {
                        var col = idx % res_shape[1];
                        var row = (idx - col) / res_shape[1];
                        var row_offset = row * shape1[1];
                        for (var j = 0; j < shape1[1]; j++)
                        {
                            this[idx] += tensor1[j + row_offset] * tensor2[j * shape2[1] + col];
                        }
                    }
                });
                return this;
            }
            else
            {
                Debug.Log("Data for all Tensors needs to be colocated on the same device. - CPU != GPU");
            }
            return this;
        }

        public FloatTensor Ceil(bool inline = false)

// Returns a new Tensor with the smallest integer greater than or equal to each element
        {
            var result = inline ? this : this.emptyTensorCopy();

            if (dataOnGpu)
            {
                //TODO: Fix GPU operations. https://github.com/OpenMined/OpenMined/issues/126
                result.Gpu(shader);
                if (!inline) return CeilGPU(result);
                CeilGPU_();
                return this;
            }

            result.Data = data.AsParallel().Select(x => (float) Math.Ceiling(x)).ToArray();
            return result;
        }

        public FloatTensor Cos(bool inline = false)
        {
            if (dataOnGpu)
            {
                if (!inline) return CosGPU();
                CosGPU_();
                return this;
            }
            var result = inline ? this : this.emptyTensorCopy();
            result.Data = data.AsParallel().Select(x => (float) Math.Cos((double) x)).ToArray();
            return result;
        }

        public FloatTensor Cosh(bool inline = false)
        {
            if (dataOnGpu)
            {
                if (!inline) return CoshGPU();
                CoshGPU_();
                return this;
            }
            var result = inline ? this : this.emptyTensorCopy();
            result.Data = data.AsParallel().Select(x => (float) Math.Cosh((double) x)).ToArray();
            return result;
        }

        public FloatTensor Div(FloatTensor x, bool inline = false)
        {
            // Check if both tensors are compatible for sum
            SameSizeDimensionsShapeAndLocation(ref x);

            var result = inline ? this : this.emptyTensorCopy();

            if (dataOnGpu & x.dataOnGpu)
            {
                result.Gpu(shader);
                if (inline)
                {
                    if (autograd)
                        throw new InvalidOperationException(
                            "Cannot call inline functions if you intend to run backprop.");
                    DivElemGPU_(x);
                    return this;
                }
                result = DivElemGPU(x, result);
            }
            else
            {
                result.Data = data.AsParallel().Zip(x.Data.AsParallel(), (a, b) => a / b).ToArray();
            }

            if (autograd)
                HookAutograd(ref result, ref x, "div_elem");

            return result;
        }

        public FloatTensor AddMatrixVectorProduct(FloatTensor matrix, FloatTensor vector)
        {
            var gpu = dataOnGpu & matrix.DataOnGpu & vector.DataOnGpu;
            var cpu = !(dataOnGpu | matrix.DataOnGpu | vector.DataOnGpu);

            var ref_shape = this.Shape;
            var matrix_shape = matrix.Shape;
            var vector_shape = vector.Shape;

            if (ref_shape.Length != 1)
                throw new InvalidOperationException(
                    "Cannot perform this operation on a tensor with more than one dimension");
            if (ref_shape[0] != vector_shape[0])
                throw new InvalidOperationException(String.Format(
                    "Cannot add matrix-vector product to tensor: {0} & {1}.", ref_shape[0], vector_shape[0]));
            if (matrix_shape[1] != vector_shape[0])
                throw new InvalidOperationException(String.Format("Last dimension of matrix doesn't match: {0} vs {1}.",
                    matrix_shape[1], vector_shape[0]));

            if (gpu)
            {
                AddMatrixVectorProductGPU(matrix, vector);
            }
            else if (cpu)
            {
                var nCpu = SystemInfo.processorCount;
                Parallel.For(0, nCpu, workerId =>
                {
                    var max = size * (workerId + 1) / nCpu;
                    for (var idx = size * workerId / nCpu; idx < max; idx++)
                    {
                        for (var j = 0; j < ref_shape[0]; j++)
                        {
                            this[idx] += vector[j] * matrix[j + (idx * ref_shape[0])];
                        }
                    }
                });
            }
            else
            {
                Debug.Log("Data for all Tensors needs to be colocated on the same device. - CPU != GPU");
            }

            return this;
        }

        public FloatTensor Div(float value, bool inline = false)
        {
            var result = inline ? this : this.emptyTensorCopy();
            if (dataOnGpu)
            {
                result.Gpu(shader);
                if (!inline) return DivScalarGPU(value, result);
                DivScalarGPU_(value);
                return this;
            }
            result.Data = data.AsParallel().Select(x => x / value).ToArray();
            return result;
        }

        public FloatTensor Exp(bool inline = false)
        {
            //var result = new FloatTensor(_ctrl:ctrl, _shape:shape, _shader:this.shader);
            var result = inline ? this : this.emptyTensorCopy();

            if (dataOnGpu)
            {
                if (!inline) return ExpGPU();
                ExpGPU_();
                return this;
            }
            result.Data = data.AsParallel().Select(x => (float) Math.Exp((double) x)).ToArray();
            return result;
        }

        public FloatTensor Floor(bool inline = false)
        {
            var result = inline ? this : this.emptyTensorCopy();
            if (dataOnGpu)
            {
                result.Gpu(shader);
                if (!inline) return FloorGPU(result);
                FloorGPU_();
                return this;
            }
            result.Data = data.AsParallel().Select(x => (float) Math.Floor(x)).ToArray();
            return result;
        }

        public FloatTensor Round()
        {
            var result = new FloatTensor(_controller: controller, _shape: shape, _shader: this.shader);

            if (dataOnGpu)
            {
                return RoundGPU();
            }
            result.Data = data.AsParallel().Select(x => (float) Math.Round(x)).ToArray();
            return result;
        }

        public bool IsContiguous()
        {
            return strides[strides.Length - 1] == 1L;
        }

        public FloatTensor Log1p()
        {
            var result = new FloatTensor(_controller: controller, _shape: shape, _shader: this.shader);

            if (dataOnGpu)
            {
            	return Log1pGPU();
            }
            result.Data = data.AsParallel().Select(x => (float) (Math.Log(1 + x))).ToArray();
            return result;
        }

        public FloatTensor MM(FloatTensor x)
        {
            if (this.shape.Length != 2 || x.shape.Length != 2)
            {
                throw new InvalidOperationException(
                    "Cannot do MM on tensors that aren't 2 dimentional. Try calling view() to reshape");
            }

            var resultShape = new int[2];
            resultShape[0] = shape[0];
            resultShape[1] = x.shape[1];

            var result = new FloatTensor(_controller: controller, _shape: resultShape);

            if (this.dataOnGpu)
            {
                result.Gpu(shader);
            }

            result.AddMatrixMultiply(this, x);

            if (autograd)
            {
                HookAutograd(ref result, ref x, "mm");
            }

            return result;
        }

        public FloatTensor Mul(FloatTensor x, bool inline = false)
        {
            // Check if both tensors are compatible for sum
            SameSizeDimensionsShapeAndLocation(ref x);

            var result = inline ? this : this.emptyTensorCopy();

            if (dataOnGpu && x.dataOnGpu)
            {
                if (inline)
                {
                    if (autograd)
                    {
                        throw new InvalidOperationException(
                            "Cannot call inline functions if you intend to run backprop.");
                    }
                    MulElemGPU_(x);
                    return this;
                }
                result = MulElemGPU(x, result);
            }
            else
            {
                result.Data = data.AsParallel().Zip(x.Data.AsParallel(), (a, b) => a * b).ToArray();
            }

            if (autograd)
            {
                HookAutograd(ref result, ref x, "mul_elem");
            }

            return result;
        }

        public FloatTensor Mul(float value, bool inline = false)
        {
            var result = inline ? this : this.emptyTensorCopy();

            if (dataOnGpu)
            {
                if (!inline) return MulScalarGPU(value, result);
                MulScalarGPU_(value);
                return this;
            }

            result.Data = data.AsParallel().Select(x => x * value).ToArray();
            return result;
        }


        public FloatTensor Sub(FloatTensor x, bool inline = false)
        {
            // Check if both tensors are compatible for sum
            SameSizeDimensionsShapeAndLocation(ref x);

            var result = inline ? this : this.emptyTensorCopy();

            if (dataOnGpu & x.dataOnGpu)
            {
                if (inline)
                {
                    if (autograd)
                        throw new InvalidOperationException(
                            "Cannot call inline functions if you intend to run backprop.");
                    SubElemGPU_(x);
                    return this;
                }
                result = SubElemGPU(x, result);
            }
            else
            {
                result.Data = data.AsParallel().Zip(x.Data.AsParallel(), (a, b) => a - b).ToArray();

                if (autograd && !inline)
                {
                    HookAutograd(ref result, ref x, "sub_elem");
                }
            }

            return result;
        }

        public FloatTensor Pow(FloatTensor x, bool inline = false)
        {
            // Check if both tensors are compatible for sum
            SameSizeDimensionsShapeAndLocation(ref x);

            if (inline & autograd)
                throw new InvalidOperationException("Cannot call inline functions if you intend to run backprop.");

            var result = inline ? this : this.emptyTensorCopy();

            if (dataOnGpu)
            {
                result.Gpu(shader);
                if (!inline) return PowElemGPU(x, result);
                result.PowElemGPU_(x);
                return this;
            }

            result.Data = data.AsParallel().Zip(x.Data.AsParallel(), (a, b) => (float) Math.Pow((double) a, b))
                .ToArray();
            HookAutograd(ref result, ref x, "pow_elem");

            return result;
        }

        public FloatTensor Pow(float value, bool inline = false)
        {
            if (inline & autograd)
                throw new InvalidOperationException("Cannot call inline functions if you intend to run backprop.");

            var result = inline ? this : this.emptyTensorCopy();

            if (dataOnGpu)
            {
                result.Gpu(shader);
                if (!inline) return PowScalarGPU(value, result);
                PowScalarGPU_(value);
                return this;
            }

            result.Data = data.AsParallel().Select(x => x * (float) Math.Pow((double) x, value)).ToArray();
            HookAutograd(ref result, value, "pow_scalar");

            return result;
        }


        public FloatTensor Neg(bool inline = false)
        {
            var result = inline ? this : this.emptyTensorCopy();

            if (dataOnGpu)
            {
                result.Gpu(shader);
                if (!inline) return NegateGPU();
                NegateGPU_();
                return this;
            }
            result.Data = data.AsParallel().Select(x => -x).ToArray();
            return result;
        }

        public FloatTensor Rsqrt()
        {
            if (dataOnGpu)
            {
                return RsqrtGPU();
            }

            var result = new FloatTensor(_controller: controller, _shape: shape, _shader: this.shader);
            result.Data = data.AsParallel().Select(x => 1 / (float) Math.Sqrt(x)).ToArray();
            return result;
        }

        public FloatTensor Sign(bool inline = false)
        {
            var result = inline ? this : this.emptyTensorCopy();

            if (dataOnGpu)
            {
                result.Gpu(shader);
                if (!inline) return SignGPU(result);
                SignGPU_();
                return this;
            }

            result.Data = data.AsParallel().Select(x => (float) Math.Sign(x)).ToArray();
            return result;
        }

        public FloatTensor Sin(bool inline = false)
        {
            if (dataOnGpu)
            {
                if (!inline) return SinGPU();
                SinGPU_();
                return this;
            }
            var result = inline ? this : this.emptyTensorCopy();
            result.Data = data.AsParallel().Select(x => (float) Math.Sin((double) x)).ToArray();
            return result;
        }

        public FloatTensor SizeTensor()
        {
            var data = new float[shape.Length];
            var ndims = new int[shape.Length];
            for (var dim = 0; dim < shape.Length; dim++)
            {
                data[dim] = shape[dim];
            }

            var result = new FloatTensor(_controller: controller, _data: data, _shape: ndims);
            return result;
        }

        public FloatTensor Sqrt()
        {
            if (dataOnGpu)
            {
                return SqrtGPU();
            }

            var result = new FloatTensor(_controller: controller, _shape: shape, _shader: shader);
            result.Data = data.AsParallel().Select(x => (float) Math.Sqrt((double) x)).ToArray();
            return result;
        }

        public FloatTensor Sub(float value, bool inline = false)
        {
            var result = inline ? this : this.emptyTensorCopy();

            if (dataOnGpu)
            {
                result.Gpu(shader);
                if (!inline) return SubScalarGPU(value, result);
                SubScalarGPU_(value);
                return this;
            }

            result.Data = data.AsParallel().Select(x => x - value).ToArray();
            return result;
        }

        public FloatTensor Tan(bool inline = false)
        {
            if (dataOnGpu)
            {
                if (!inline) return TanGPU();
                TanGPU_();
                return this;
            }
            var result = inline ? this : this.emptyTensorCopy();
            result.Data = data.AsParallel().Select(x => (float) Math.Tan((double) x)).ToArray();
            return result;
        }

        public FloatTensor Tanh(bool inline = false)
        {
            if (dataOnGpu)
            {
                return TanhGPU();
            }
            var result = new FloatTensor(_controller: controller, _shape: shape, _shader: this.shader);
            result.Data = data.AsParallel().Select(x => (float) Math.Tanh((double) x)).ToArray();
            return result;
        }

        public FloatTensor Transpose()
        {
            if (shape.Length != 2)
                throw new InvalidOperationException("Need to specify parameters for tensors with more than 2 dims.");

            return Transpose(0, 1);
        }

        public FloatTensor Trunc(bool inline = false)
        {
            if (dataOnGpu)
            {
                return TruncGPU();
            }
            var result = new FloatTensor(_controller: controller, _shape: shape, _shader: this.shader);
            result.Data = data.AsParallel().Select(x => (float) Math.Truncate((double) x)).ToArray();
            return result;
        }

        public FloatTensor Transpose(int dimension1, int dimension2)
        {
            //TODO: Should we create a new Tensor object here?
            if (dimension1 < 0 || dimension1 >= shape.Length)
                throw new ArgumentOutOfRangeException("dimension1");
            if (dimension2 < 0 || dimension2 >= shape.Length)
                throw new ArgumentOutOfRangeException("dimension2");

            if (dimension1 == dimension2)
            {
                return this;
            }

            var newShape = (int[]) Shape.Clone();
            var tmpDim = newShape[dimension1];
            newShape[dimension1] = newShape[dimension2];
            newShape[dimension2] = tmpDim;

            var result = new FloatTensor(_controller: controller, _shape: newShape, _shader: this.shader);
            var nCpu = SystemInfo.processorCount;
            Parallel.For(0, nCpu, workerId =>
            {
                var max = size * (workerId + 1) / nCpu;
                for (var i = size * workerId / nCpu; i < max; i++)
                {
                    var idxs = GetIndices(i);
                    var tmp = idxs[dimension1];
                    idxs[dimension1] = idxs[dimension2];
                    idxs[dimension2] = tmp;
                    result[idxs] = this[i];
                }
            });

            return result;
        }

        public void Triu_(int k)
        {
            if (shape.Length != 2)
            {
                throw new InvalidOperationException(
                    String.Format("Matrix multiply not possible: Num. Dimensions {0} != 2.", shape.Length));
            }
            if (dataOnGpu)
            {
                //UnityEngine.Debug.Log ("Entra");
                TriuGPU_(k);
                return;
            }
            var nCpu = SystemInfo.processorCount;
            Parallel.For(0, nCpu, workerId =>
            {
                var max = size * (workerId + 1) / nCpu;
                for (var i = size * workerId / nCpu; i < max; i++)
                {
                    var col = i % this.shape[1];
                    var row = (i - col) / this.shape[1];
                    if (col < row + k)
                    {
                        this[i] = 0.0f;
                    }
                }
            });
        }

        public FloatTensor Sinh(bool inline = false)
        {
            if (dataOnGpu)
            {
                if (!inline) return SinhGPU();
                SinhGPU_();
                return this;
            }
            var result = inline ? this : this.emptyTensorCopy();
            result.Data = data.AsParallel().Select(x => (float) Math.Sinh((double) x)).ToArray();
            return result;
        }

        public float Trace()
        {
            if ((shape.Length != 2) || (shape[0] != shape[1]))
                throw new InvalidOperationException("Trace is defined on square 2d matrices only.");

            var stride = strides[0] + strides[1];
            return dataOnGpu
                ? TraceGPU()
                : Enumerable.Range(0, shape.Min()).AsParallel().Select(i => this[i * stride]).Sum();
        }

        public FloatTensor Sigmoid(bool inline = false)
        {
            if (dataOnGpu)
            {
                if (!inline) return SigmoidGPU(this.emptyTensorCopy());
                if (autograd)
                    throw new InvalidOperationException(
                        "Cannot call inline functions if you intend to run backprop.");

                SigmoidGPU_();
                return this;
            }

            var result = inline ? this : this.emptyTensorCopy();
            var nCpu = SystemInfo.processorCount;
            Parallel.For(0, nCpu, workerId =>
            {
                var max = size * (workerId + 1) / nCpu;
                for (var i = size * workerId / nCpu; i < max; i++)
                {
                    if (this[i] >= 0)
                    {
                        var s = Math.Exp(-(double) this[i]);
                        result[i] = (float) (1 / (1.0f + s));
                    }
                    else
                    {
                        var s = Math.Exp((double) this[i]);
                        result[i] = (float) (s / (1.0f + s));
                    }
                }
            });

            if (autograd)
            {
                HookAutograd(ref result, "sigmoid");
            }

            return result;
        }

        public FloatTensor View(int[] new_shape, bool inline = false)
        {
            var newSize = 1;
            for (var i = 0; i < new_shape.Length; i++)
            {
                newSize *= new_shape[i];
            }

            var result = this;
            if (newSize != size) return result;
            if (dataOnGpu)
            {
                if (inline)
                {
                    shape = new_shape;

                    shapeBuffer.Release();
                    shapeBuffer = new ComputeBuffer(shape.Length, sizeof(int));
                    shapeBuffer.SetData(shape);
                }
                else
                {
                    result = new FloatTensor(_controller: controller, _shape: new_shape, _shader: this.shader);
                    result.Gpu(shader);
                    CopyBuffer(dataBuffer, result.DataBuffer);
                }
            }
            else if (inline)
            {
                shape = new_shape;
            }
            else
            {
                result = new FloatTensor(_controller: controller, _data: data, _shape: new_shape, _shader: shader);
            }
            return result;
        }

        public FloatTensor Remainder(float divisor, bool inline = false)
        {
            if (inline & autograd)
                throw new InvalidOperationException("Cannot call inline functions if you intend to run backprop.");
            if (autograd)
                throw new InvalidOperationException("Autograd not available for Remainder.");

            var result = inline ? this : this.emptyTensorCopy();

            if (dataOnGpu)
            {
                result.Gpu(shader);
                if (inline) { RemainderScalarGPU_(divisor); return this; }
                else { result = RemainderScalarGPU(result, divisor); }
            }
            else
            {
                var nCpu = SystemInfo.processorCount;
                Parallel.For(0, nCpu, workerId => {
                    var max = size * (workerId + 1) / nCpu;
                    for (var i = size * workerId / nCpu; i < max; i++)
                    {
                        result[i] = this[i] % divisor;
                    };
                });
            }

            return result;
        }

        public FloatTensor Remainder(FloatTensor divisor, bool inline = false)
        {
            SameSizeDimensionsShapeAndLocation(ref divisor);
            if (inline & autograd)
                throw new InvalidOperationException("Cannot call inline functions if you intend to run backprop.");
            if (autograd)
                throw new InvalidOperationException("Autograd not available for Remainder.");

            var result = inline ? this : this.emptyTensorCopy();

            if (dataOnGpu)
            {
                result.Gpu(shader);
                if (inline)
                {
                    RemainderElemGPU_(divisor);
                    return this;
                }
                else
                {
                    result = RemainderElemGPU(divisor,result);
                }
            }
            else
            {
                var nCpu = SystemInfo.processorCount;
                Parallel.For(0, nCpu, workerId => {
                    var max = size * (workerId + 1) / nCpu;
                    for (var i = size * workerId / nCpu; i < max; i++)
                    {
                        result[i] = this[i] % divisor[i];
                    };
                });
            }

            return result;
        }

        public FloatTensor ViewAs(FloatTensor x, bool inline = false)
        {
            return this.View(x.shape, inline);
        }

        public void Zero_()
        {
            if (dataOnGpu)
            {
                ZeroGPU_();
                return;
            }

            Array.Clear(data, 0, size);
        }


        public FloatTensor Squeeze(int dim = -1, bool inline = false)
        {
            var list = new List<int>();

            if (dim >= 0)
            {
                for (int i = 0; i < shape.Length; i++)
                {
                    if (i != dim)
                    {
                        list.Add(shape[i]);
                    }
                    else
                    {
                        if (shape[i] != 1)
                        {
                            list.Add(shape[i]);
                        }
                    }
                }
            }
            else
            {
                for (int i = 0; i < shape.Length; i++)
                {
                    if (shape[i] > 1)
                    {
                        list.Add(shape[i]);
                    }
                }
            }

            FloatTensor result = this;
            if (list.Count == 0)
            {
                if (!inline)
                {
                    result = new FloatTensor(_controller: controller, _data: data, _shape: shape, _shader: shader);
                }
            }
            else
            {
                if (inline)
                {
                    View(list.ToArray(), inline: true);
                }
                else
                {
                    result = View(list.ToArray());
                }
            }

            return result;
        }

/*** Reduce Functions ***/

        public FloatTensor Reduce(
            Func<float, float, int, float[], float> reducer,
            Func<float, int, float> mapper
        )
        {
            int[] outDims = {1};
            var output = new float[1];
            output[0] = mapper(MultiThread.Reduce(data, reducer), Size);

            return new FloatTensor(controller, outDims, output);
        }

        internal void ForEach(
            int dim,
            Action<float[], int, int> iterator
        )
        {
            int interations = size / shape[dim];
            int values = shape[dim];
            var stride = strides[dim];
            MultiThread.For(interations, (i, len) =>
            {
                var temp = new float[values];
                var offset = GetDimReduceOffset(i, values, stride);

                for (int v = 0; v < values; v++)
                {
                    temp[v] = this[offset + v * stride];
                }

                iterator(temp, offset, stride);
            });
        }

        public FloatTensor Reduce(
            int dim,
            bool keepdim,
            Func<float, float, int, float[], float> reducer,
            Func<float, int, float> mapper
        )
        {
            int len = shape.Length;

            if (dim < 0)
            {
                return Reduce(reducer, mapper);
            }

            AssertDim(dim, len);

            if (len == 1)
            {
                keepdim = true;
            }

            var stride = strides[dim];
            int values = shape[dim];

            int outSize = 1;
            var outDims = keepdim ? new int[len] : new int[len - 1];

            for (int i = 0; i < len; i++)
            {
                if (i < dim)
                {
                    outDims[i] = shape[i];
                }
                else if (i > dim)
                {
                    outDims[keepdim ? i : i - 1] = shape[i];
                }
                else if (i == dim)
                {
                    if (keepdim)
                    {
                        outDims[i] = 1;
                    }

                    continue;
                }

                outSize *= shape[i];
            }

            var output = new float[outSize];

            _dimForEach(outSize, values, stride, (vals, index, length) =>
            {
                var acc = vals[0];

                for (int i = 1; i < length; i++)
                {
                    acc = reducer(acc, vals[i], i, vals);
                }

                output[index] = mapper(acc, length);
            });

            return new FloatTensor(controller, outDims, output);
        }

        public FloatTensor Min(int dim = -1, bool keepdim = false)
        {
            // TODO: Implement GPU op. with GPU tests.
            return Reduce(dim, keepdim, (acc, val, index, arr) => acc < val ? acc : val, (val, len) => val);
        }

        public FloatTensor Max(int dim = -1, bool keepdim = false)
        {
            // TODO: Implement GPU op. with GPU tests.
            return Reduce(dim, keepdim, (acc, val, index, arr) => acc > val ? acc : val, (val, len) => val);
        }

        public FloatTensor Sum(int dim = -1, bool keepdim = false)
        {
            // TODO: Implement GPU op. with GPU tests.
            return Reduce(dim, keepdim, (acc, val, index, arr) => acc + val, (val, len) => val);
        }

        public FloatTensor Prod(int dim = -1, bool keepdim = false)
        {
            // TODO: Implement GPU op. with GPU tests.
            return Reduce(dim, keepdim, (acc, val, index, arr) => acc * val, (val, len) => val);
        }

        public FloatTensor Mean(int dim = -1, bool keepdim = false)
        {
            // TODO: Implement GPU op. with GPU tests.
            return Reduce(dim, keepdim, (acc, val, index, arr) => acc + val, (val, len) => val / (float) len);
        }


/*** Reduce Functions End ***/

// closes class and namespace
    }
}
