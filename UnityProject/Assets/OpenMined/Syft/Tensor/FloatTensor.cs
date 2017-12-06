using System;
using System.Runtime.Remoting.Messaging;
using UnityEngine;
using OpenMined.Network.Utils;
using OpenMined.Network.Controllers;

namespace OpenMined.Syft.Tensor
{
public partial class FloatTensor
{
// Should we put a check incase this variable overflows?
private static volatile int nCreated = 0;

private float[] data;
private long[] strides;
private int[] shape;
private int size;

private int id;

//Making this public for now. Usage of below functions might not be efficient but can help development
//private
public long GetIndex (params long[] indices)
{
	long offset = 0;
	for (int i = 0; i < indices.Length; ++i) {
		if (indices [i] >= shape [i] || indices [i] < 0)
			throw new IndexOutOfRangeException ();
		offset += indices [i] * strides [i];
	}
	return offset;
}

//private
public long GetIndex (params int[] indices)
{
	var long_indices = Array.ConvertAll (indices, item => (long)item);
	return GetIndex (long_indices);
}

//private
public long[] GetIndices (long index)
{
	var idx = index;
	long[] indices = new long[Shape.Length];
	for (int i = 0; i < Shape.Length; ++i) {
		indices [i] = (idx - (idx % (strides [i]))) / strides [i];
		idx -= indices [i] * strides [i];
	}
	return indices;
}

public float[] Data {
	get { return data; }
}

public int[] Shape {
	get { return shape; }
}

public int Size {
	get { return size; }
}

public int Id {
	get { return id; }

	set { id = value; }
}

public static int CreatedObjectCount {
	get { return nCreated; }
}

public long[] Strides {
	get { return strides; }
}

		public FloatTensor (int[] _shape, float[] _data = null, ComputeBuffer _dataBuffer = null, ComputeBuffer _shapeBuffer = null, ComputeShader _shader = null, bool _copyData = true, bool _dataOnGpu=false)
{
			// First: check that shape is valid.
			if (_shape == null || _shape.Length == 0) {
				throw new InvalidOperationException ("Tensor shape can't be an empty array.");
			}	

			// Second: since shape is valid, let's save it
			shape = (int[])_shape.Clone ();

			// Third: let's see what kind of data we've got. We should either have
			// a GPU ComputeBuffer or a data[] object. 
			if (_data != null && _shapeBuffer == null && _dataBuffer == null) {
				
				InitCpu (_data:_data,_copyData:_copyData);

			} else if (_dataBuffer != null && _shapeBuffer != null && _data == null) {
				
				// looks like we have GPU data being passed in... initialize a GPU tensor.

				InitGpu (_shader, _dataBuffer, _shapeBuffer, _copyData);


			} else {

				// no data seems to be passed in... or its got missing stuff

				// if CPU works... go with that
				if (_data != null) {
					InitCpu (_data, _copyData);
				} else if (_dataBuffer != null && _shader != null) {

					if (SystemInfo.supportsComputeShaders) {

						// seems i'm just missing a shape buffer - no biggie
						shapeBuffer = new ComputeBuffer (shape.Length, sizeof(int));
						shapeBuffer.SetData (shape);

						InitGpu (_shader, _dataBuffer, _shapeBuffer, _copyData);

					} else {
						throw new InvalidOperationException ("You seem to be trying to create a GPU tensor without having access to a GPU...");
					}

				} else {

					// nothing else seems to work - i suppose i'm just supposed to initialize an empty tensor.
					long acc = 1;
					for (var i = shape.Length - 1; i >= 0; --i) {
						acc *= shape [i];
					}

					if(_dataOnGpu) {

						_shapeBuffer = new ComputeBuffer (shape.Length, sizeof(int));
						_shapeBuffer.SetData (shape);

						_dataBuffer = new ComputeBuffer(size, sizeof(float));

						InitGpu(_shader:_shader, _dataBuffer: _dataBuffer, _shapeBuffer:_shapeBuffer, _copyData:false);
						this.Zero_();

					} else {

						_data = new float[acc];

						InitCpu (_data, false);

					}

				}


			}

			// Lastly: let's set the ID of the tensor.
			// IDEs might show a warning, but ref and volatile seems to be working with Interlocked API.
			id = System.Threading.Interlocked.Increment (ref nCreated);

}

		public void InitCpu(float[] _data, bool _copyData)  
		{
			// looks like we have CPU data being passed in... initialize a CPU tensor.
			dataOnGpu = false;

			if (_copyData) {
				data = (float[])_data.Clone ();
			} else {
				data = _data;
			}
				
			size = _data.Length;

			setStridesAndCheckShape();

		}

		public void InitGpu(ComputeShader _shader, ComputeBuffer _dataBuffer, ComputeBuffer _shapeBuffer, bool _copyData) 
		{
			// First: we need to check that we have a shader
			if (SystemInfo.supportsComputeShaders && _shader != null) {
				shader = _shader;
				initShaderKernels ();
			} else {
				throw new FormatException ("You tried to initialize a GPU tensor without access to a shader or gpu.");
			}

			// Second: let's save our buffer.
			dataBuffer = _dataBuffer;
			shapeBuffer = _shapeBuffer;

			if (_copyData) {
				// TODO:
				throw new FormatException ("Cannot copy data buffers yet");
			}

			// Third: let's set the tensor's size to be equal to that of the buffer
			size = _dataBuffer.count;

			// Fourth:
			setStridesAndCheckShape();

		}

		public void setStridesAndCheckShape() {
			// Third: let's initialize our strides.
			strides = new long[shape.Length];

			// Fifth: we should check that the buffer's size matches our shape.
			long acc = 1;
			for (var i = shape.Length - 1; i >= 0; --i) {
				strides [i] = acc;
				acc *= shape [i];
			}

			// Sixth: let's check to see that our shape and data sizes match.
			if (acc != size)
				throw new FormatException ("Tensor shape and data do not match.");
		}


public FloatTensor Copy ()
{
	FloatTensor copy = new FloatTensor (_data:this.data, _shape:this.shape, _shader:this.shader);
	return copy;
}

public float this [params long[] indices] {
	get { return this [GetIndex (indices)]; }
	set { this [GetIndex (indices)] = value; }
}

public float this [long index] {
	get { return Data [index]; }
	set { Data [index] = value; }
}

public string ProcessMessage (Command msgObj, SyftController ctrl)
{
	switch (msgObj.functionCall) {

	case "abs":
	{
		// calls the function on our tensor object
		var result = this.Abs ();
		// returns the function call name with the OK status
		return ctrl.addTensor (result) + "";
	}
	case "abs_":
	{
		// calls the function on our tensor object
		this.Abs(inline: true);
		// returns the function call name with the OK status
		return id.ToString();
	}
	case "add_elem":
	{
		Debug.LogFormat("add_elem");
		var tensor_1 = ctrl.getTensor(int.Parse(msgObj.tensorIndexParams[0]));
		var result = this.Add(tensor_1);
		return ctrl.addTensor(result) + "";
	}
	case "acos":
	{
		var result = Acos ();
		ctrl.addTensor (result);
		return result.Id.ToString ();
	}
	case "acos_":
	{
		Acos (inline: true);
		return Id.ToString ();
	}
	case "asin":
	{
		var result = Asin ();
		ctrl.addTensor (result);
		return result.Id.ToString ();
	}
	case "asin_":
	{
		Asin (inline: true);
		return Id.ToString ();
	}
	case "atan":
	{
		var result = Atan ();
		ctrl.addTensor (result);
		return result.Id.ToString ();
	}
	case "atan_":
	{
		Atan (inline: true);
		return Id.ToString ();
	}
	case "add_elem_":
	{
		Debug.LogFormat("add_elem_");
		var tensor_1 = ctrl.getTensor(int.Parse(msgObj.tensorIndexParams[0]));
		this.Add(tensor_1, inline: true);
		return this.id + "";
	}
	case "add_scalar":
	{
		Debug.LogFormat("add_scalar");
		FloatTensor result = Add(float.Parse(msgObj.tensorIndexParams[0]));
		return ctrl.addTensor (result) + "";
	}
	case "add_scalar_":
	{
		Debug.LogFormat("add_scalar_");
		this.Add(float.Parse( msgObj.tensorIndexParams[0]), inline: true);
		return this.id + "";
	}
	case "addmm_":
	{
		var tensor_1 = ctrl.getTensor(int.Parse(msgObj.tensorIndexParams[0]));
		var tensor_2 = ctrl.getTensor(int.Parse(msgObj.tensorIndexParams[1]));
		AddMatrixMultiply(tensor_1, tensor_2);
		return msgObj.functionCall + ": OK";
	}
	case "addmv_":
	{
		var tensor_1 = ctrl.getTensor (int.Parse (msgObj.tensorIndexParams [0]));
		var tensor_2 = ctrl.getTensor (int.Parse (msgObj.tensorIndexParams [1]));
		AddMatrixVectorProduct (tensor_1, tensor_2);
		return msgObj.functionCall + ": OK";
	}
	case "ceil":
	{
		var result = this.Ceil();
		return ctrl.addTensor(result) + "";
	}
	case "ceil_":
	{
		this.Ceil(inline: true);
		return this.id + "";
	}
	case "copy":
	{
		var result = Copy ();
		ctrl.addTensor (result);
		return result.Id.ToString ();
	}
	case "cos":
	{
		var result = Cos ();
		ctrl.addTensor (result);
		return result.Id.ToString ();
	}
	case "cos_":
	{
		Cos (inline: true);
		return Id.ToString ();
	}
	case "cosh":
	{
		var result = Cosh ();
		ctrl.addTensor (result);
		return result.Id.ToString ();
	}
	case "Cosh_":
	{
		Cosh (inline: true);
		return Id.ToString ();
	}
	case "cpu":
	{
		Cpu ();
		return msgObj.functionCall + ": OK";
	}
	case "div_elem":
	{
		var tensor_1 = ctrl.getTensor(int.Parse(msgObj.tensorIndexParams[0]));
		var result = this.Div(tensor_1);
		return ctrl.addTensor(result) + "";
	}
	case "div_elem_":
	{
		var tensor_1 = ctrl.getTensor(int.Parse(msgObj.tensorIndexParams[0]));
		this.Div(tensor_1, inline: true);
		return this.id + "";
	}
	case "div_scalar":
	{
		FloatTensor result = Div (float.Parse (msgObj.tensorIndexParams [0]));

		return ctrl.addTensor (result) + "";
	}
	case "div_scalar_":
	{
		this.Div (float.Parse (msgObj.tensorIndexParams [0]), inline: true);
		return this.id + "";
	}
	case "mul_scalar":
	{
		FloatTensor result = Mul (float.Parse (msgObj.tensorIndexParams [0]));

		return ctrl.addTensor (result) + "";
	}
	case "mul_scalar_":
	{
		this.Mul(float.Parse( msgObj.tensorIndexParams[0]), inline: true);
		return this.id + "";
	}
	case "pow_elem":
	{
		var tensor_1 = ctrl.getTensor (int.Parse (msgObj.tensorIndexParams [0]));
		var result = this.Pow (tensor_1);
		return ctrl.addTensor (result) + "";
	}
	case "floor":
	{
		var result = this.Floor();
		return ctrl.addTensor(result) + "";
	}
	case "floor_":
	{
		this.Floor(inline: true);
		return this.id + "";
	}
	case "gpu":
	{
		if (Gpu())
		{
			return msgObj.functionCall + ": OK : Moved data to GPU.";
		}
		else
		{
			return msgObj.functionCall + ": FAILED : Did not move data.";
		}
	}
	case "mul_elem":
	{
		var tensor_1 = ctrl.getTensor(int.Parse(msgObj.tensorIndexParams[0]));
		var result = this.Mul(tensor_1);

		return ctrl.addTensor(result) + "";
	}
	case "mul_elem_":
	{
		var tensor_1 = ctrl.getTensor(int.Parse(msgObj.tensorIndexParams[0]));
		this.Mul (tensor_1, inline: true);
		return this.id + "";
	}
	case "pow_elem_":
	{
		var tensor_1 = ctrl.getTensor (int.Parse (msgObj.tensorIndexParams [0]));
		this.Pow (tensor_1, inline: true);
		return this.id + "";
	}
	case "pow_scalar":
	{
		FloatTensor result = Pow (float.Parse (msgObj.tensorIndexParams [0]));
		return ctrl.addTensor (result) + "";
	}
	case "pow_scalar_":
	{
		this.Pow (float.Parse (msgObj.tensorIndexParams [0]), inline: true);
		return this.id + "";
	}
	case "sigmoid_":
	{
		this.Sigmoid(inline: true);
		return this.id + "";
	}
	case "sigmoid":
	{
		var result = this.Sigmoid();
		return ctrl.addTensor(result) + "";
	}
	case "sub_elem":
	{
		Debug.LogFormat("sub_elem");
		var tensor_1 = ctrl.getTensor(int.Parse(msgObj.tensorIndexParams[0]));
		var result = this.Sub(tensor_1);

		return ctrl.addTensor(result) + "";
	}
	case "sub_elem_":
	{
		Debug.LogFormat("sub_elem_");
		var tensor_1 = ctrl.getTensor(int.Parse(msgObj.tensorIndexParams[0]));
		this.Sub(tensor_1, inline: true);
		return this.id + "";
	}
	case "neg":
	{
		var result = Neg ();
		ctrl.addTensor (result);
		return result.Id.ToString ();
	}
	case "rsqrt":
	{
		var result = Rsqrt();
		ctrl.addTensor(result);
		return result.Id.ToString();
	}
	case "print":
	{
		bool dataOriginallyOnGpu = dataOnGpu;
		if (dataOnGpu) {
			Cpu ();
		}

		string data = this.Print ();
		Debug.LogFormat ("<color=cyan>Print:</color> {0}", string.Join (",", this.Data));

		if (dataOriginallyOnGpu) {
			Gpu ();
		}
		return data;
	}
	case "sign":
	{
		Debug.LogFormat ("sign");
		var result = this.Sign ();
		return ctrl.addTensor (result) + "";
	}
	case "sign_":
	{
		Debug.LogFormat("sign_");
		Sign (inline: true);
		return Id.ToString();
	}
	case "sin":
	{
		var result = Sin ();
		ctrl.addTensor (result);
		return result.Id.ToString ();
	}
	case "sin_":
	{
		Sin (inline: true);
		return Id.ToString ();
	}
	case "sqrt":
	{
		var result = Sqrt ();
		ctrl.addTensor (result);
		return result.id.ToString ();
	}
	case "size":
	{
		var result = SizeTensor ();
		ctrl.addTensor (result);
		return result.id.ToString ();
	}

	case "sub_scalar":
	{
		Debug.LogFormat ("sub_scalar");
		FloatTensor result = Sub (float.Parse (msgObj.tensorIndexParams [0]));

		return ctrl.addTensor (result) + "";
	}
	case "sub_scalar_":
	{
		Debug.LogFormat("sub_scalar_");
		this.Sub(float.Parse( msgObj.tensorIndexParams[0]), inline: true);
		return this.id + "";
	}
	case "sum_dim":
	{
		Debug.LogFormat ("sum_dim");
		FloatTensor result = this.Sum (int.Parse (msgObj.tensorIndexParams [0]));
		return ctrl.addTensor (result) + "";
	}
	case "to_numpy":
	{
		return string.Join(" ", data);
	}
	case "tan":
	{
		var result = Tan ();
		ctrl.addTensor (result);
		return result.Id.ToString ();
	}
	case "tan_":
	{
		Tan (inline: true);
		return Id.ToString ();
	}
	case "tanh":
	{
		var result = Tanh ();
		ctrl.addTensor (result);
		return result.Id.ToString ();
	}
	case "sinh":
	{
		var result = Sinh ();
		ctrl.addTensor (result);
		return result.Id.ToString ();
	}
	case "sinh_":
	{
		Sinh (inline: true);
		return Id.ToString ();
	}
	case "transpose":
	{
		var result = Transpose ();
		ctrl.addTensor (result);
		return result.Id.ToString ();
	}

	case "triu":
	{
		var K = int.Parse (msgObj.tensorIndexParams [0]);
		var result = Copy ();
		result.Triu_ (K);
		ctrl.addTensor (result);
		return result.Id.ToString ();
	}
	case "triu_":
	{
		var K = int.Parse (msgObj.tensorIndexParams [0]);
		Triu_ (K);
		return Id.ToString ();
	}

	case "trunc":
	{
		var result = Trunc ();
		ctrl.addTensor (result);
		return result.Id.ToString ();
	}

	case "view":
	{

		int[] new_dims = new int[msgObj.tensorIndexParams.Length];
		for (int i = 0; i < msgObj.tensorIndexParams.Length; i++) {
			new_dims [i] = int.Parse (msgObj.tensorIndexParams [i]);
		}
		var result = View (new_dims);
		ctrl.addTensor (result);
		return result.Id.ToString ();
	}

	case "view_":
	{
		int[] new_dims = new int[msgObj.tensorIndexParams.Length];
		for (int i = 0; i < msgObj.tensorIndexParams.Length; i++) {
			new_dims [i] = int.Parse (msgObj.tensorIndexParams [i]);
		}
		View (new_dims, inline: true);
		return Id.ToString ();
	}

	case "zero_":
	{
		Zero_ ();
		return msgObj.functionCall + ": OK";
	}
	case "is_contiguous":
	{
		return Convert.ToString (IsContiguous ());
	}
	default:
		break;
	}
	return "SyftController.processMessage: Command not found.";
}

public string Print ()
{
	bool dataOriginallyOnGpu = dataOnGpu;
	if (dataOnGpu) {
		Cpu ();
	}

	string print = "";

	if (shape.Length > 3)
		print += "Only printing the last 3 dimesnions\n";
	int d3 = 1;
	if (shape.Length > 2)
		d3 = shape [shape.Length - 3];
	int d2 = 1;
	if (shape.Length > 1)
		d2 = shape [shape.Length - 2];
	int d1 = shape [shape.Length - 1];

	for (int k = 0; k < d3; k++) {
		for (int j = 0; j < d2; j++) {
			for (int i = 0; i < d1; i++) {
				float f = data [i + j * d1 + k * d1 * d2];
				print += f.ToString () + ",\t";
			}
			print += "\n";
		}
		print += "\n";
	}

	if (dataOriginallyOnGpu) {
		Gpu ();
	}
	return print;
}
}
}
