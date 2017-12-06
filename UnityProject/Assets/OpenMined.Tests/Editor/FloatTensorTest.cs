using System;
using UnityEngine;
using UnityEditor;
using NUnit.Framework;

using OpenMined.Syft.Tensor;

namespace OpenMined.Tests
{

public class FloatTensorTest
{

[TestFixtureSetUp]
public void Init()
{
	//Init runs once before running test cases.
}

[TestFixtureTearDown]
public void CleanUp()
{
	//CleanUp runs once after all test cases are finished.
}

[SetUp]
public void SetUp()
{
	//SetUp runs before all test cases
}

[TearDown]
public void TearDown()
{
	//SetUp runs after all test cases
}

[Test]
public void Copy()
{
	float[] array = { 1, 2, 3, 4, 5 };
	int[] shape = { 5 };

	var tensor = new FloatTensor(_data:array, _shape:shape);
	var copy = tensor.Copy();

	Assert.AreEqual(copy.Shape,tensor.Shape);
	Assert.AreEqual(copy.Data, tensor.Data);
	Assert.AreNotEqual(copy.Id, tensor.Id);
}

[Test]
public void Cos()
{
	float[] data1 = { 0.4f, 0.5f, 0.3f, -0.1f };
	int[] shape1 = { 4 };
	var tensor = new FloatTensor(_data:data1, _shape:shape1);

	float[] data2 = { 0.92106099f,  0.87758256f,  0.95533649f,  0.99500417f };
	int[] shape2 = { 4 };
	var expectedCosTensor = new FloatTensor(_data:data2, _shape:shape2);

	var actualCosTensor = tensor.Cos();

	for (int i = 2; i < actualCosTensor.Size; i++)
	{
		Assert.AreEqual (expectedCosTensor.Data[i], actualCosTensor.Data[i]);
	}
}

[Test]
public void Cos_()
{
	float[] data1 = { 0.4f, 0.5f, 0.3f, -0.1f };
	int[] shape1 = { 4 };
	var tensor = new FloatTensor(_data:data1, _shape:shape1);

	float[] data2 = {  0.92106099f,  0.87758256f,  0.95533649f,  0.99500417f };
	int[] shape2 = { 4 };
	var expectedCosTensor = new FloatTensor(_data:data2, _shape:shape2);

	tensor.Cos (inline: true);

	for (int i = 2; i < tensor.Size; i++)
	{
		Assert.AreEqual (expectedCosTensor.Data[i], tensor.Data[i]);
	}
}

[Test]
public void Acos()
{
	float[] data1 = { 0.4f, 0.5f, 0.3f, -0.1f };
	int[] shape1 = { 4 };
	var tensor = new FloatTensor(_data:data1, _shape:shape1);

	float[] data2 = { 1.15927948f,  1.04719755f,  1.26610367f,  1.67096375f };
	int[] shape2 = { 4 };
	var expectedAcosTensor = new FloatTensor(_data:data2, _shape:shape2);

	var actualAcosTensor = tensor.Acos();

	for (int i = 2; i < actualAcosTensor.Size; i++)
	{
		Assert.AreEqual (expectedAcosTensor.Data[i], actualAcosTensor.Data[i]);
	}
}

[Test]
public void Acos_()
{
	float[] data1 = { 0.4f, 0.5f, 0.3f, -0.1f };
	int[] shape1 = { 4 };
	var tensor = new FloatTensor(_data:data1, _shape:shape1);

	float[] data2 = {  1.15927948f,  1.04719755f,  1.26610367f,  1.67096375f };
	int[] shape2 = { 4 };
	var expectedAcosTensor = new FloatTensor(_data:data2, _shape:shape2);

	tensor.Acos (inline: true );

	for (int i = 2; i < tensor.Size; i++)
	{
		Assert.AreEqual (expectedAcosTensor.Data[i], tensor.Data[i]);
	}
}

[Test]
public void Asin()
{
	float[] data1 = { 0.4f, 0.5f, 0.3f, -0.1f };
	int[] shape1 = { 4 };
	var tensor = new FloatTensor(_data:data1, _shape:shape1);

	float[] data2 = { 0.41151685f,  0.52359878f,  0.30469265f, -0.10016742f };
	int[] shape2 = { 4 };
	var expectedAsinTensor = new FloatTensor(_data:data2, _shape:shape2);

	var actualAsinTensor = tensor.Asin();

	for (int i = 2; i < actualAsinTensor.Size; i++)
	{
		Assert.AreEqual (expectedAsinTensor.Data[i], actualAsinTensor.Data[i]);
	}
}

[Test]
public void Asin_()
{
	float[] data1 = { 0.4f, 0.5f, 0.3f, -0.1f };
	int[] shape1 = { 4 };
	var tensor = new FloatTensor(_data:data1, _shape:shape1);

	float[] data2 = {  0.41151685f,  0.52359878f,  0.30469265f, -0.10016742f };
	int[] shape2 = { 4 };
	var expectedAsinTensor = new FloatTensor(_data:data2, _shape:shape2);

	tensor.Asin (inline: true);

	for (int i = 2; i < tensor.Size; i++)
	{
		Assert.AreEqual (expectedAsinTensor.Data[i], tensor.Data[i]);
	}
}

[Test]
public void Atan()
{
	float[] data1 = { 30, 20, 40, 50 };
	int[] shape1 = { 4 };
	var tensor = new FloatTensor(_data:data1, _shape:shape1);

	float[] data2 = {  1.53747533f,  1.52083793f,  1.54580153f,  1.55079899f };
	int[] shape2 = { 4 };
	var expectedAtanTensor = new FloatTensor(_data:data2, _shape:shape2);

	var actualAtanTensor = tensor.Atan();

	for (int i = 2; i < actualAtanTensor.Size; i++)
	{
		Assert.AreEqual (expectedAtanTensor.Data[i], actualAtanTensor.Data[i]);
	}

}

[Test]
public void Atan_()
{
	float[] data1 = { 30, 20, 40, 50 };
	int[] shape1 = { 4 };
	var tensor = new FloatTensor(_data:data1, _shape:shape1);

	float[] data2 = { 1.53747533f,  1.52083793f,  1.54580153f,  1.55079899f };
	int[] shape2 = { 4 };
	var expectedAtanTensor = new FloatTensor(_data:data2, _shape:shape2);

	tensor.Atan (inline: true);

	for (int i = 2; i < tensor.Size; i++)
	{
		Assert.AreEqual (expectedAtanTensor.Data[i], tensor.Data[i]);
	}
}

[Test]
public void Sin()
{
	float[] data1 = { 0.4f, 0.5f, 0.3f, -0.1f };
	int[] shape1 = { 4 };
	var tensor = new FloatTensor(_data:data1, _shape:shape1);

	float[] data2 = { 0.38941834f,  0.47942554f,  0.29552021f, -0.09983342f };
	int[] shape2 = { 4 };
	var expectedSinTensor = new FloatTensor(_data:data2, _shape:shape2);

	var actualSinTensor = tensor.Sin();

	for (int i = 2; i < actualSinTensor.Size; i++)
	{
		Assert.AreEqual (expectedSinTensor.Data[i], actualSinTensor.Data[i]);
	}
}

[Test]
public void Sin_()
{
	float[] data1 = { 0.4f, 0.5f, 0.3f, -0.1f };
	int[] shape1 = { 4 };
	var tensor = new FloatTensor(_data:data1, _shape:shape1);

	float[] data2 = {  0.38941834f,  0.47942554f,  0.29552021f, -0.09983342f };
	int[] shape2 = { 4 };
	var expectedSinTensor = new FloatTensor(_data:data2, _shape:shape2);

	tensor.Sin (inline: true);

	for (int i = 2; i < tensor.Size; i++)
	{
		Assert.AreEqual (expectedSinTensor.Data[i], tensor.Data[i]);
	}
}

[Test]
public void Cosh()
{
	float[] data1 = { 0.4f, 0.5f, 0.3f, -0.1f };
	int[] shape1 = { 4 };
	var tensor = new FloatTensor(_data:data1, _shape:shape1);

	float[] data2 = {  1.08107237f,  1.12762597f,  1.04533851f,  1.00500417f };
	int[] shape2 = { 4 };
	var expectedCoshTensor = new FloatTensor(_data:data2, _shape:shape2);

	var actualCoshTensor = tensor.Cosh();

	for (int i = 2; i < actualCoshTensor.Size; i++)
	{
		Assert.AreEqual (expectedCoshTensor.Data[i], actualCoshTensor.Data[i]);
	}
}

[Test]
public void Cosh_()
{
	float[] data1 = { 0.4f, 0.5f, 0.3f, -0.1f };
	int[] shape1 = { 4 };
	var tensor = new FloatTensor(_data:data1, _shape:shape1);

	float[] data2 = {  1.08107237f,  1.12762597f,  1.04533851f,  1.00500417f };
	int[] shape2 = { 4 };
	var expectedCoshTensor = new FloatTensor(_data:data2, _shape:shape2);

	tensor.Cosh (inline: true);

	for (int i = 2; i < tensor.Size; i++)
	{
		Assert.AreEqual (expectedCoshTensor.Data[i], tensor.Data[i]);
	}
}
[Test]
public void Create1DTensor()
{
	float[] array = { 1, 2, 3, 4, 5 };
	int[] shape = { 5 };

	var tensor = new FloatTensor(_data:array, _shape:shape);

	Assert.AreEqual(array.Length, tensor.Size);

	for (int i = 0; i < array.Length; i++)
	{
		Assert.AreEqual(array[i], tensor[i]);
	}
}

[Test]
public void Create2DTensor()
{
	float[,] array = { { 1, 2, 3, 4, 5 }, { 6, 7, 8, 9, 10 } };

	float[] data = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
	int[] shape = { 2, 5 };

	var tensor = new FloatTensor(_data:data, _shape:shape);

	Assert.AreEqual(array.GetLength(0), tensor.Shape[0]);
	Assert.AreEqual(array.GetLength(1), tensor.Shape[1]);

	for (int i = 0; i < array.GetLength(0); i++)
	{
		for (int j = 0; j < array.GetLength(1); j++)
		{
			Assert.AreEqual(array[i, j], tensor[i, j]);
		}
	}
}

[Test]
public void Create3DTensor()
{
	float[,,] array = { { { 1, 2 }, { 3, 4 }, { 5, 6 } }, { { 7, 8 }, { 9, 10 }, { 11, 12 } } };

	float[] data = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
	int[] shape = { 2, 3, 2 };

	var tensor = new FloatTensor(_data:data, _shape:shape);

	Assert.AreEqual(array.GetLength(0), tensor.Shape[0]);
	Assert.AreEqual(array.GetLength(1), tensor.Shape[1]);
	Assert.AreEqual(array.GetLength(2), tensor.Shape[2]);

	for (int i = 0; i < array.GetLength(0); i++)
	{
		for (int j = 0; j < array.GetLength(1); j++)
		{
			for (int k = 0; k < array.GetLength(2); k++)
			{
				Assert.AreEqual(array[i, j, k], tensor[i, j, k]);
			}
		}
	}
}

[Test]
public void Transpose2D()
{
	float[] data1 = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
	int[] shape1 = { 2, 5};

	float[] data2 = { 1, 6, 2, 7, 3, 8, 4, 9, 5, 10 };
	int[] shape2 = { 5, 2 };

	var tensor = new FloatTensor(_data:data1, _shape:shape1);
	var transpose = new FloatTensor(_data:data2, _shape:shape2);

	for (int i = 0; i < tensor.Shape[0]; i++)
	{
		for (int j = 0; j < tensor.Shape[1]; j++)
		{
			Assert.AreEqual(tensor[i, j], transpose[j, i]);
		}
	}

	var transposed = tensor.Transpose();
	for (int i = 0; i < tensor.Shape[1]; i++)
	{
		for (int j = 0; j < tensor.Shape[0]; j++)
		{
			Assert.AreEqual(transposed[i, j], transpose[i, j]);
		}
	}
}

[Test]
public void Transpose3D()
{
	float[] data1 = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
	int[] shape1 = { 3, 2, 2};

	float[] data2 = { 1, 5, 9, 3, 7, 11, 2, 6, 10, 4, 8, 12 };
	int[] shape2 = { 2, 2, 3 };

	var tensor = new FloatTensor(_data:data1, _shape:shape1);
	var transpose = new FloatTensor(_data:data2, _shape:shape2);

	for (int i = 0; i < tensor.Shape[0]; i++)
	{
		for (int j = 0; j < tensor.Shape[1]; j++)
		{
			for (int k = 0; k < tensor.Shape [2]; k++)
			{
				Assert.AreEqual (tensor [i, j, k], transpose [k, j, i]);
			}
		}
	}

	var transposed = tensor.Transpose(0, 2);
	for (int i = 0; i < transposed.Shape[0]; i++)
	{
		for (int j = 0; j < transposed.Shape[1]; j++)
		{
			for (int k = 0; k < transposed.Shape [2]; k++)
			{
				Assert.AreEqual(transposed[i, j, k], transpose[i, j, k]);
			}
		}
	}
}

[Test]
public void TransposeNoDimensionsSpecified()
{
	float[] data1 = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
	int[] shape1 = { 3, 2, 2 };

	// Test Tensor with more than 2 dimensions
	var tensor = new FloatTensor(_data:data1, _shape:shape1);
	Assert.That(() => tensor.Transpose(),
	            Throws.TypeOf<InvalidOperationException>());

	// Test tensor with less than 2 dimensions
	float[] data2 = { 1, 2, 3, 4, 5 };
	int[] shape2 = { 5 };
	tensor = new FloatTensor(_data:data2, _shape:shape2);
	Assert.That(() => tensor.Transpose(),
	            Throws.TypeOf<InvalidOperationException>());
}

[Test]
public void TransposeDimensionsOutOfRange()
{
	float[] data1 = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
	int[] shape1 = {3, 2, 2};

	// Test negative dimension indexes
	var tensor = new FloatTensor(_data:data1, _shape:shape1);
	Assert.That(() => tensor.Transpose(-1, 0),
	            Throws.TypeOf<ArgumentOutOfRangeException>());
	Assert.That(() => tensor.Transpose(0, -1),
	            Throws.TypeOf<ArgumentOutOfRangeException>());

	// Test dimension indexes bigger than tensor's shape lenght
	var tensor2 = new FloatTensor(_data:data1, _shape:shape1);
	Assert.That(() => tensor2.Transpose(3, 0),
	            Throws.TypeOf<ArgumentOutOfRangeException>());
	Assert.That(() => tensor2.Transpose(0, 3),
	            Throws.TypeOf<ArgumentOutOfRangeException>());
}

[Test]
public void AddScalar()
{
	float[] data1 = { -1, 0, 0.1f, 1, float.MaxValue, float.MinValue };
	int[] shape1 = {3, 2};
	var tensor1 = new FloatTensor(_data:data1, _shape:shape1);

	float scalar = -100;

	var tensorSum = tensor1.Add (scalar);

	for (int i = 0; i < tensorSum.Size; i++)
	{
		Assert.AreEqual (tensor1.Data [i] + scalar, tensorSum.Data [i]);
	}
}

[Test]
public void AddScalar_()
{
	float[] data1 = { -1, 0, 1, float.MaxValue, float.MinValue };
	int[] shape1 = {5, 1};
	var tensor1 = new FloatTensor(_data:data1, _shape:shape1);

	float[] data2 = { -101, -100, -99, float.MaxValue-100, float.MinValue-100 };
	int[] shape2 = {5, 1};
	var tensor2 = new FloatTensor(_data:data2, _shape:shape2);

	float scalar = -100;

	tensor1.Add (scalar, inline: true);

	for (int i = 0; i < tensor1.Size; i++)
	{
		Assert.AreEqual (tensor1.Data[i], tensor2.Data [i]);
	}
}

[Test]
public void Add()
{
	float[] data1 = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
	int[] shape1 = {2, 5};
	var tensor1 = new FloatTensor(_data:data1, _shape:shape1);

	float[] data2 = { 3, 2, 6, 9, 10, 1, 4, 8, 5, 7};
	int[] shape2 = {2, 5};
	var tensor2 = new FloatTensor(_data:data2, _shape:shape2);

	var tensorSum = tensor1.Add (tensor2);

	for (int i = 0; i < tensorSum.Size; i++)
	{
		Assert.AreEqual (tensor1.Data [i] + tensor2.Data [i], tensorSum.Data [i]);
	}
}

[Test]
public void AddUnequalSizes()
{
	float[] data1 = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
	int[] shape1 = { 2, 5 };
	var tensor1 = new FloatTensor(_data:data1, _shape:shape1);

	float[] data2 = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
	int[] shape2 = { 2, 6 };
	var tensor2 = new FloatTensor(_data:data2, _shape:shape2);

	Assert.That(() => tensor1.Add(tensor2),
	            Throws.TypeOf<InvalidOperationException>());
}

[Test]
public void AddUnequalDimensions()
{
	float[] data1 = { 1, 2, 3, 4 };
	int[] shape1 = { 4 };
	var tensor1 = new FloatTensor(_data:data1, _shape:shape1);

	float[] data2 = { 1, 2, 3, 4 };
	int[] shape2 = { 2, 2 };
	var tensor2 = new FloatTensor(_data:data2, _shape:shape2);

	Assert.That(() => tensor1.Add(tensor2),
	            Throws.TypeOf<InvalidOperationException>());
}

[Test]
public void AddUnequalShapes()
{
	float[] data1 = { 1, 2, 3, 4, 5, 6 };
	int[] shape1 = { 2, 3 };
	var tensor1 = new FloatTensor(_data:data1, _shape:shape1);

	float[] data2 = { 1, 2, 3, 4, 5, 6 };
	int[] shape2 = { 3, 2 };
	var tensor2 = new FloatTensor(_data:data2, _shape:shape2);

	Assert.That(() => tensor1.Add(tensor2),
	            Throws.TypeOf<InvalidOperationException>());
}

[Test]
public void Add_()
{
	float[] data1 = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
	int[] shape1 = {2, 5};
	var tensor1 = new FloatTensor(_data:data1, _shape:shape1);

	float[] data2 = { 3, 2, 6, 9, 10, 1, 4, 8, 5, 7};
	int[] shape2 = {2, 5};
	var tensor2 = new FloatTensor(_data:data2, _shape:shape2);

	float[] data3 = { 4, 4, 9, 13, 15, 7, 11, 16, 14, 17};
	int[] shape3 = {2, 5};
	var tensor3 = new FloatTensor(_data:data3, _shape:shape3);

	tensor1.Add (tensor2, inline: true);

	for (int i = 0; i < tensor1.Size; i++)
	{
		Assert.AreEqual (tensor3.Data[i], tensor1.Data [i]);
	}
}

[Test]
public void AddUnequalSizes_()
{
	float[] data1 = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
	int[] shape1 = { 2, 5 };
	var tensor1 = new FloatTensor(_data:data1, _shape:shape1);

	float[] data2 = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
	int[] shape2 = { 2, 6 };
	var tensor2 = new FloatTensor(_data:data2, _shape:shape2);

	Assert.That(() => tensor1.Add(tensor2, inline: true),
	            Throws.TypeOf<InvalidOperationException>());
}

[Test]
public void AddUnequalDimensions_()
{
	float[] data1 = { 1, 2, 3, 4 };
	int[] shape1 = { 4 };
	var tensor1 = new FloatTensor(_data:data1, _shape:shape1);

	float[] data2 = { 1, 2, 3, 4 };
	int[] shape2 = { 2, 2 };
	var tensor2 = new FloatTensor(_data:data2, _shape:shape2);

	Assert.That(() => tensor1.Add(tensor2, inline: true),
	            Throws.TypeOf<InvalidOperationException>());
}

[Test]
public void AddUnequalShapes_()
{
	float[] data1 = { 1, 2, 3, 4, 5, 6 };
	int[] shape1 = { 2, 3 };
	var tensor1 = new FloatTensor(_data:data1, _shape:shape1);

	float[] data2 = { 1, 2, 3, 4, 5, 6 };
	int[] shape2 = { 3, 2 };
	var tensor2 = new FloatTensor(_data:data2, _shape:shape2);

	Assert.That(() => tensor1.Add(tensor2, inline: true),
	            Throws.TypeOf<InvalidOperationException>());
}

[Test]
public void Abs()
{
	float[] data1 = { -1, 0, 1, float.MaxValue, float.MinValue };
	int[] shape1 = { 5 };
	var tensor1 = new FloatTensor(_data:data1, _shape:shape1);

	float[] data2 = { 1, 0, 1, float.MaxValue, -float.MinValue };
	int[] shape2 = { 5 };
	var tensorAbs = new FloatTensor(_data:data2, _shape:shape2);

	tensor1 = tensor1.Abs();

	for (int i = 0; i < tensor1.Size; i++)
	{
		Assert.AreEqual (tensor1.Data[i], tensorAbs.Data[i]);
	}
}

[Test]
public void Abs_()
{
	float[] data1 = { -1, 0, 1, float.MaxValue, float.MinValue };
	int[] shape1 = { 5 };
	var tensor1 = new FloatTensor(_data:data1, _shape:shape1);

	float[] data2 = { 1, 0, 1, float.MaxValue, -float.MinValue };
	int[] shape2 = { 5 };
	var tensorAbs = new FloatTensor(_data:data2, _shape:shape2);

	tensor1.Abs(inline: true);

	for (int i = 0; i < tensor1.Size; i++)
	{
		Assert.AreEqual (tensor1.Data[i], tensorAbs.Data[i]);
	}
}

[Test]
public void Neg()
{
	float[] data1 = { -1, 0, 1, float.MaxValue, float.MinValue };
	int[] shape1 = { 5 };
	var tensor1 = new FloatTensor(_data:data1, _shape:shape1);

	float[] data2 = { 1, 0, -1, -float.MaxValue, -float.MinValue };
	int[] shape2 = { 5 };
	var tensorNeg = new FloatTensor(_data:data2, _shape:shape2);

	var result = tensor1.Neg ();

	for (int i = 0; i < tensor1.Size; i++)
	{
		Assert.AreEqual (result.Data[i], tensorNeg.Data[i]);
	}
}

[Test]
public void Rsqrt()
{
	float[] data1 = { 1, 2, 3, 4 };
	int[] shape1 = { 4 };
	float[] correct = { 1, (float)0.7071068, (float)0.5773503, (float)0.5};
	var tensor1 = new FloatTensor(_data:data1, _shape:shape1);

	var result = tensor1.Rsqrt();

	for (int i = 2; i < correct.Length; i++)
	{
		Assert.AreEqual (Math.Round(correct[i], 3), Math.Round(result.Data[i], 3));
	}

}

[Test]
public void Sigmoid_()
{
	float[] data1 = { 0.0f };
	int[] shape1 = { 1 };
	var tensor1 = new FloatTensor(_data:data1, _shape:shape1);
	tensor1.Sigmoid(inline: true);
	Assert.AreEqual(tensor1.Data[0], 0.5f);

	float[] data2 = { 0.1f, 0.5f, 1.0f, 2.0f };
	float[] data3 = { -0.1f, -0.5f, -1.0f, -2.0f };
	int[] shape2 = { 4 };
	var tensor2 = new FloatTensor(_data:data2, _shape:shape2);
	var tensor3 = new FloatTensor(_data:data3, _shape:shape2);
	tensor2.Sigmoid(inline: true);
	tensor3.Sigmoid(inline: true);
	var sum = tensor2.Add(tensor3);

	for (int i = 0; i < sum.Size; i++)
	{
		Assert.AreEqual(sum.Data[i], 1.0f);
	}
}

[Test]
public void Sigmoid()
{
	float[] data1 = { 0.0f };
	int[] shape1 = { 1 };
	var tensor1 = new FloatTensor(_data:data1, _shape:shape1).Sigmoid();
	Assert.AreEqual(tensor1.Data[0], 0.5f);

	float[] data2 = { 0.1f, 0.5f, 1.0f, 2.0f };
	float[] data3 = { -0.1f, -0.5f, -1.0f, -2.0f };
	int[] shape2 = { 4 };
	var tensor2 = new FloatTensor(_data:data2, _shape:shape2);
	var tensor3 = new FloatTensor(_data:data3, _shape:shape2);
	var sum = tensor2.Sigmoid().Add(tensor3.Sigmoid());

	for (int i = 0; i < sum.Size; i++)
	{
		Assert.AreEqual(sum.Data[i], 1.0f);
	}
}
[Test]
public void Sign()
{
	float[] data1 = {float.MinValue, -100.0f, -1.0f, -0.0001f, -0.0f, +0.0f, 0.0001f, 1.0f, 10.0f, float.MaxValue};
	int[] shape1 = { 1, 10 };

	var tensor1 = new FloatTensor(_data:data1, _shape:shape1);

	float[] data2 = {-1.0f, -1.0f, -1.0f, -1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 1.0f};
	var tensorSign = new FloatTensor(_data:data2, _shape:shape1);

	var result1 = tensor1.Sign();

	for (int i = 0; i < tensor1.Size; i++)
	{
		Assert.AreEqual (result1.Data[i], tensorSign.Data[i]);
	}
}

[Test]
public void Sign_()
{
	float[] data1 = {float.MinValue, -100.0f, -1.0f, -0.0001f, -0.0f, +0.0f, 0.0001f, 1.0f, 10.0f, float.MaxValue};
	int[] shape1 = { 1, 10 };

	var tensor1 = new FloatTensor(_data:data1, _shape:shape1);

	float[] data2 = {-1.0f, -1.0f, -1.0f, -1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 1.0f};
	var tensorSign_ = new FloatTensor(_data:data2, _shape:shape1);

	tensor1.Sign (inline: true);

	for (int i = 0; i < tensor1.Size; i++)
	{
		Assert.AreEqual (tensor1.Data[i], tensorSign_.Data[i]);
	}
}

[Test]
public void Zero_()
{
	float[] data1 = { -1, 0, 1, float.MaxValue, float.MinValue };
	int[] shape1 = { 5 };
	var tensor1 = new FloatTensor(_data:data1, _shape:shape1);

	float[] data2 = { 0, 0, 0, 0, 0 };
	int[] shape2 = { 5 };
	var tensorZero = new FloatTensor(_data:data2, _shape:shape2);

	tensor1.Zero_ ();

	for (int i = 0; i < tensor1.Size; i++)
	{
		Assert.AreEqual (tensor1.Data[i], tensorZero.Data[i]);
	}
}

[Test]
public void MultiplicationElementwise()
{
	float[] data1 = { float.MinValue, -10, -1.5f, 0, 1.5f, 10, 20, float.MaxValue };
	int[] shape1 = {2, 4};
	var tensor1 = new FloatTensor(_data:data1, _shape:shape1);

	float[] data2 = { float.MinValue, -10, -1.5f, 0, 1.5f, 10, 20, float.MaxValue };
	int[] shape2 = {2, 4};
	var tensor2 = new FloatTensor(_data:data2, _shape:shape2);

	var tensorMult = tensor1.Mul (tensor2);

	for (int i = 0; i < tensorMult.Size; i++)
	{
		float current = tensor1.Data [i] * tensor2.Data [i];
		Assert.AreEqual (tensorMult.Data [i], current);
	}
}

[Test]
public void MultiplicationElementwiseUnequalSizes()
{
	float[] data1 = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
	int[] shape1 = { 2, 5 };
	var tensor1 = new FloatTensor(_data:data1, _shape:shape1);

	float[] data2 = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
	int[] shape2 = { 2, 6 };
	var tensor2 = new FloatTensor(_data:data2, _shape:shape2);

	Assert.That(() => tensor1.Mul(tensor2),
	            Throws.TypeOf<InvalidOperationException>());
}

[Test]
public void MulElementwiseUnequalDimensions()
{
	float[] data1 = { 1, 2, 3, 4 };
	int[] shape1 = { 4 };
	var tensor1 = new FloatTensor(_data:data1, _shape:shape1);

	float[] data2 = { 1, 2, 3, 4 };
	int[] shape2 = { 2, 2 };
	var tensor2 = new FloatTensor(_data:data2, _shape:shape2);

	Assert.That(() => tensor1.Mul(tensor2),
	            Throws.TypeOf<InvalidOperationException>());
}

[Test]
public void MultiplicationElementwiseUnequalShapes()
{
	float[] data1 = { 1, 2, 3, 4, 5, 6 };
	int[] shape1 = { 2, 3 };
	var tensor1 = new FloatTensor(_data:data1, _shape:shape1);

	float[] data2 = { 1, 2, 3, 4, 5, 6 };
	int[] shape2 = { 3, 2 };
	var tensor2 = new FloatTensor(_data:data2, _shape:shape2);
	Assert.That(() => tensor1.Mul(tensor2),
	            Throws.TypeOf<InvalidOperationException>());
}

[Test]
public void MultiplicationElementwise_()
{
	float[] data1 = { float.MinValue, -10, -1.5f, 0, 1.5f, 10, 20, float.MaxValue };
	int[] shape1 = {2, 4};
	var tensor1 = new FloatTensor(_data:data1, _shape:shape1);

	float[] data2 = { float.MinValue, -10, -1.5f, 0, 1.5f, 10, 20, float.MaxValue };
	int[] shape2 = {2, 4};
	var tensor2 = new FloatTensor(_data:data2, _shape:shape2);

	float[] data3 = { float.PositiveInfinity, 100, 2.25f, 0, 2.25f, 100, 400, float.PositiveInfinity };
	int[] shape3 = {2, 4};
	var tensorMult = new FloatTensor(_data:data3, _shape:shape3);

	tensor1.Mul (tensor2, inline: true);

	for (int i = 0; i < tensorMult.Size; i++)
	{
		Assert.AreEqual (tensorMult.Data [i], tensor1.Data [i]);
	}
}

[Test]
public void MultiplicationElementwiseUnequalSizes_()
{
	float[] data1 = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
	int[] shape1 = { 2, 5 };
	var tensor1 = new FloatTensor(_data:data1, _shape:shape1);

	float[] data2 = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
	int[] shape2 = { 2, 6 };
	var tensor2 = new FloatTensor(_data:data2, _shape:shape2);

	Assert.That(() => tensor1.Mul(tensor2, inline: true),
	            Throws.TypeOf<InvalidOperationException>());
}

[Test]
public void MultiplicationElementwisenUnequalDimensions_()
{
	float[] data1 = { 1, 2, 3, 4 };
	int[] shape1 = { 4 };
	var tensor1 = new FloatTensor(_data:data1, _shape:shape1);

	float[] data2 = { 1, 2, 3, 4 };
	int[] shape2 = { 2, 2 };
	var tensor2 = new FloatTensor(_data:data2, _shape:shape2);

	Assert.That(() => tensor1.Mul(tensor2, inline: true),
	            Throws.TypeOf<InvalidOperationException>());
}

[Test]
public void MultiplicationElementwiseUnequalShapes_()
{
	float[] data1 = { 1, 2, 3, 4, 5, 6 };
	int[] shape1 = { 2, 3 };
	var tensor1 = new FloatTensor(_data:data1, _shape:shape1);

	float[] data2 = { 1, 2, 3, 4, 5, 6 };
	int[] shape2 = { 3, 2 };
	var tensor2 = new FloatTensor(_data:data2, _shape:shape2);
	Assert.That(() => tensor1.Mul(tensor2, inline: true),
	            Throws.TypeOf<InvalidOperationException>());
}

[Test]
public void DivisionElementwise_()
{
	float[] data1 = { float.MinValue, -10, -1.5f, 0, 1.5f, 10, 20, float.MaxValue };
	int[] shape1 = {2, 4};
	var tensor1 = new FloatTensor(_data:data1, _shape:shape1);

	float[] data2 = { 1, 1, 1, (float)Double.NaN, 1, 1, 1, 1 };
	int[] shape2 = {2, 4};
	var tensor2 = new FloatTensor(_data:data2, _shape:shape2);

	tensor1.Div (tensor1, inline: true);

	for (int i = 0; i < tensor1.Size; i++)
	{
		Assert.AreEqual (tensor2.Data [i], tensor1.Data [i]);
	}
}

[Test]
public void DivisionElementwise()
{
	float[] data1 = { float.MinValue, -10, -1.5f, 0, 1.5f, 10, 20, float.MaxValue };
	int[] shape1 = {2, 4};
	var tensor1 = new FloatTensor(_data:data1, _shape:shape1);

	float[] data2 = { float.MinValue, -10, -1.5f, 0, 1.5f, 10, 20, float.MaxValue };
	int[] shape2 = {2, 4};
	var tensor2 = new FloatTensor(_data:data2, _shape:shape2);

	var tensorMult = tensor1.Div (tensor2);

	for (int i = 0; i < tensorMult.Size; i++)
	{
		float current = tensor1.Data [i] / tensor2.Data [i];
		Assert.AreEqual (tensorMult.Data [i], current);
	}
}

[Test]
public void DivisionElementwiseUnequalSizes()
{
	float[] data1 = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
	int[] shape1 = { 2, 5 };
	var tensor1 = new FloatTensor(_data:data1, _shape:shape1);

	float[] data2 = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
	int[] shape2 = { 2, 6 };
	var tensor2 = new FloatTensor(_data:data2, _shape:shape2);

	Assert.That(() => tensor1.Div(tensor2),
	            Throws.TypeOf<InvalidOperationException>());
}

[Test]
public void DivisionElementwiseUnequalDimensions()
{
	float[] data1 = { 1, 2, 3, 4 };
	int[] shape1 = { 4 };
	var tensor1 = new FloatTensor(_data:data1, _shape:shape1);

	float[] data2 = { 1, 2, 3, 4 };
	int[] shape2 = { 2, 2 };
	var tensor2 = new FloatTensor(_data:data2, _shape:shape2);

	Assert.That(() => tensor1.Div(tensor2),
	            Throws.TypeOf<InvalidOperationException>());
}

[Test]
public void DivisionElementwiseUnequalShapes()
{
	float[] data1 = { 1, 2, 3, 4, 5, 6 };
	int[] shape1 = { 2, 3 };
	var tensor1 = new FloatTensor(_data:data1, _shape:shape1);

	float[] data2 = { 1, 2, 3, 4, 5, 6 };
	int[] shape2 = { 3, 2 };
	var tensor2 = new FloatTensor(_data:data2, _shape:shape2);

	Assert.That(() => tensor1.Div(tensor2),
	            Throws.TypeOf<InvalidOperationException>());
}

[Test]
public void DivisionScalar()
{
	float[] data1 = { float.MinValue, -10, -1.5f, 0, 1.5f, 10, 20, float.MaxValue };
	int[] shape1 = {2, 4};
	var tensor1 = new FloatTensor(_data:data1, _shape:shape1);

	// Test division by 0
	float scalar = 0;
	var result = tensor1.Div (scalar);
	for (int i = 0; i < tensor1.Size; i++)
	{
		Assert.AreEqual (tensor1.Data [i] / scalar, result.Data [i] );
	}
	// Test division
	float[] data2 = { float.MinValue, -10, -1.5f, 0, 1.5f, 10, 20, float.MaxValue };
	int[] shape2 = {2, 4};
	var tensor2 = new FloatTensor(_data:data1, _shape:shape1);

	scalar = 99;
	tensor1.Div (scalar, inline: true);
	for (int i = 0; i < tensor1.Size; i++)
	{
		Assert.AreEqual (tensor2.Data [i] / scalar, tensor1.Data [i] );
	}
}

[Test]
public void DivisionScalar_()
{
	float[] data1 = { float.MinValue, -10, -1.5f, 0, 1.5f, 10, 20, float.MaxValue };
	int[] shape1 = {2, 4};
	var tensor1 = new FloatTensor(_data:data1, _shape:shape1);
	var tensor2 = new FloatTensor(_data:data1, _shape:shape1);

	// Test multiplication by 0
	float scalar = 0;
	var result = tensor1.Mul (scalar);
	for (int i = 0; i < tensor1.Size; i++)
	{
		Assert.AreEqual (tensor2.Data [i] * scalar, result.Data [i] );
	}
}

[Test]
public void DivisionElementwiseUnequalSizes_()
{
	float[] data1 = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
	int[] shape1 = { 2, 5 };
	var tensor1 = new FloatTensor(_data:data1, _shape:shape1);

	float[] data2 = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
	int[] shape2 = { 2, 6 };
	var tensor2 = new FloatTensor(_data:data2, _shape:shape2);

	Assert.That(() => tensor1.Div(tensor2, inline: true),
	            Throws.TypeOf<InvalidOperationException>());
}

[Test]
public void DivisionElementwiseUnequalDimensions_()
{
	float[] data1 = { 1, 2, 3, 4 };
	int[] shape1 = { 4 };
	var tensor1 = new FloatTensor(_data:data1, _shape:shape1);

	float[] data2 = { 1, 2, 3, 4 };
	int[] shape2 = { 2, 2 };
	var tensor2 = new FloatTensor(_data:data2, _shape:shape2);

	Assert.That(() => tensor1.Div(tensor2, inline: true),
	            Throws.TypeOf<InvalidOperationException>());
}

[Test]
public void DivisionElementwiseUnequalShapes_()
{
	float[] data1 = { 1, 2, 3, 4, 5, 6 };
	int[] shape1 = { 2, 3 };
	var tensor1 = new FloatTensor(_data:data1, _shape:shape1);

	float[] data2 = { 1, 2, 3, 4, 5, 6 };
	int[] shape2 = { 3, 2 };
	var tensor2 = new FloatTensor(_data:data2, _shape:shape2);

	Assert.That(() => tensor1.Div(tensor2, inline: true),
	            Throws.TypeOf<InvalidOperationException>());
}

//        [Test]
//        public void ElementwiseMultiplicationDataOnDifferent()
//        {
//            int[] shape1 = { 2, 3 };
//            var tensor1 = new FloatTensor(shape1);
//            int[] shape2 = { 2, 3 };
//            var tensor2 = new FloatTensor(shape2);
//
//			Assert.That(() => tensor1.Mul(tensor2),
//                Throws.TypeOf<InvalidOperationException>());
//        }

[Test]
public void TensorId()
{
	float[] data = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
	int[] shape = { 2, 3, 2 };

	var tensor1 = new FloatTensor(_data:data, _shape:shape);
	var tensor2 = new FloatTensor(_data:data, _shape:shape);

	Assert.AreNotEqual(tensor1.Id, tensor2.Id);
	Assert.AreEqual(tensor1.Id + 1, tensor2.Id);
}

[Test]
public void MultiplicationScalar()
{
	float[] data1 = { float.MinValue, -10, -1.5f, 0, 1.5f, 10, 20, float.MaxValue };
	int[] shape1 = {2, 4};
	var tensor1 = new FloatTensor(_data:data1, _shape:shape1);
	var tensor2 = new FloatTensor(_data:data1, _shape:shape1);

	// Test multiplication by 0
	float scalar = 0;
	var result = tensor1.Mul (scalar);
	for (int i = 0; i < tensor1.Size; i++)
	{
		Assert.AreEqual (tensor2.Data [i] * scalar, result.Data [i] );
	}

	// Test multiplication by positive
	tensor1 = new FloatTensor(_data:data1, _shape:shape1);
	scalar = 99;
	result = tensor1.Mul (scalar);

	for (int i = 0; i < tensor1.Size; i++)
	{
		Assert.AreEqual (tensor2.Data [i] * scalar, result.Data [i] );
	}

	// Test multiplication by negative
	tensor1 = new FloatTensor(_data:data1, _shape:shape1);
	scalar = -99;
	result = tensor1.Mul (scalar);

	for (int i = 0; i < tensor1.Size; i++)
	{
		Assert.AreEqual (tensor2.Data [i] * scalar, result.Data [i] );
	}

	// Test multiplication by decimal
	tensor1 = new FloatTensor(_data:data1, _shape:shape1);
	scalar = 0.000001f;
	result = tensor1.Mul (scalar);

	for (int i = 0; i < tensor1.Size; i++)
	{
		Assert.AreEqual (tensor2.Data [i] * scalar, result.Data [i] );
	}
}
[Test]
public void Ceil()
{
	float[] data1 = { 5.89221f, -20.11f, 9.0f, 100.4999f, 100.5001f };
	int[] shape1 = { 5 };
	var tensor1 = new FloatTensor(_data:data1, _shape:shape1);

	float[] data2 = { 6, -20, 9, 101, 101 };
	int[] shape2 = { 5 };
	var tensorCeil = new FloatTensor(_data:data2, _shape:shape2);

	var result = tensor1.Ceil ();

	for (int i = 0; i < tensor1.Size; i++)
	{
		Assert.AreEqual (result.Data[i], tensorCeil.Data[i]);
	}
}

[Test]
public void Ceil_()
{
	float[] data1 = { 5.89221f, -20.11f, 9.0f, 100.4999f, 100.5001f };
	int[] shape1 = { 5 };
	var tensor1 = new FloatTensor(_data:data1, _shape:shape1);

	float[] data2 = { 6, -20, 9, 101, 101 };
	int[] shape2 = { 5 };
	var tensorCeil = new FloatTensor(_data:data2, _shape:shape2);

	tensor1.Ceil (inline: true);

	for (int i = 0; i < tensor1.Size; i++)
	{
		Assert.AreEqual (tensor1.Data[i], tensorCeil.Data[i]);
	}
}

[Test]
public void Floor_()
{
	float[] data1 = { 5.89221f, -20.11f, 9.0f, 100.4999f, 100.5001f };
	int[] shape1 = { 5 };
	var tensor1 = new FloatTensor(_data:data1, _shape:shape1);

	float[] data2 = { 5, -21, 9, 100, 100 };
	int[] shape2 = { 5 };
	var tensorFloor = new FloatTensor(_data:data2, _shape:shape2);

	tensor1.Floor(inline: true);

	for (int i = 0; i < tensor1.Size; i++)
	{
		Assert.AreEqual(tensor1.Data[i], tensorFloor.Data[i]);
	}
}

[Test]
public void Floor()
{
	float[] data1 = { 5.89221f, -20.11f, 9.0f, 100.4999f, 100.5001f };
	int[] shape1 = { 5 };
	var tensor1 = new FloatTensor(_data:data1, _shape:shape1);

	float[] data2 = { 5, -21, 9, 100, 100 };
	int[] shape2 = { 5 };
	var tensorFloor = new FloatTensor(_data:data2, _shape:shape2);

	var result = tensor1.Floor(inline: true);

	for (int i = 0; i < tensor1.Size; i++)
	{
		Assert.AreEqual(tensorFloor.Data[i], result.Data[i]);
	}
}

[Test]
public void SubtractElementwise()
{
	float[] data1 = { float.MinValue, -10, -1.5f, 0, 1.5f, 10, 20, float.MaxValue };
	int[] shape1 = {2, 4};
	var tensor1 = new FloatTensor(_data:data1, _shape:shape1);

	float[] data2 = { float.MaxValue, 10, 1.5f, 0, -1.5f, -10, -20, float.MinValue };
	int[] shape2 = {2, 4};
	var tensor2 = new FloatTensor(_data:data2, _shape:shape2);

	var tensor = tensor1.Sub (tensor2);

	for (int i = 0; i < tensor.Size; i++)
	{
		float current = tensor1.Data [i] - tensor2.Data [i];
		Assert.AreEqual (current, tensor.Data [i]);
	}
}

[Test]
public void SubtractElementwiseUnequalSizes()
{
	float[] data1 = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
	int[] shape1 = { 2, 5 };
	var tensor1 = new FloatTensor(_data:data1, _shape:shape1);

	float[] data2 = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
	int[] shape2 = { 2, 6 };
	var tensor2 = new FloatTensor(_data:data2, _shape:shape2);

	Assert.That(() => tensor1.Sub(tensor2),
	            Throws.TypeOf<InvalidOperationException>());
}

[Test]
public void SubtractElementwiseUnequalDimensions()
{
	float[] data1 = { 1, 2, 3, 4 };
	int[] shape1 = { 4 };
	var tensor1 = new FloatTensor(_data:data1, _shape:shape1);

	float[] data2 = { 1, 2, 3, 4 };
	int[] shape2 = { 2, 2 };
	var tensor2 = new FloatTensor(_data:data2, _shape:shape2);

	Assert.That(() => tensor1.Sub(tensor2),
	            Throws.TypeOf<InvalidOperationException>());
}

[Test]
public void SubtractElementwiseUnequalShapes()
{
	float[] data1 = { 1, 2, 3, 4, 5, 6 };
	int[] shape1 = { 2, 3 };
	var tensor1 = new FloatTensor(_data:data1, _shape:shape1);

	float[] data2 = { 1, 2, 3, 4, 5, 6 };
	int[] shape2 = { 3, 2 };
	var tensor2 = new FloatTensor(_data:data2, _shape:shape2);

	Assert.That(() => tensor1.Sub(tensor2),
	            Throws.TypeOf<InvalidOperationException>());
}

[Test]
public void SubtractElementwise_()
{
	float[] data1 = { float.MinValue, -10, -1.5f, 0, 1.5f, 10, 20, float.MaxValue };
	int[] shape1 = {2, 4};
	var tensor1 = new FloatTensor(_data:data1, _shape:shape1);

	float[] data2 = { float.MaxValue, 10, 1.5f, 0, -1.5f, -10, -20, float.MinValue };
	int[] shape2 = {2, 4};
	var tensor2 = new FloatTensor(_data:data2, _shape:shape2);

	float[] data3 = { float.NegativeInfinity, -20, -3, 0, 3, 20, 40, float.MaxValue - float.MinValue };
	int[] shape3 = {2, 4};
	var tensor3 = new FloatTensor(_data:data3, _shape:shape3);

	tensor1.Sub (tensor2, inline: true);

	for (int i = 0; i < tensor1.Size; i++)
	{
		Assert.AreEqual (tensor3.Data [i], tensor1.Data [i]);
	}
}

[Test]
public void SubtractElementwiseUnequalSizes_()
{
	float[] data1 = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
	int[] shape1 = { 2, 5 };
	var tensor1 = new FloatTensor(_data:data1, _shape:shape1);

	float[] data2 = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
	int[] shape2 = { 2, 6 };
	var tensor2 = new FloatTensor(_data:data2, _shape:shape2);

	Assert.That(() => tensor1.Sub(tensor2, inline: true),
	            Throws.TypeOf<InvalidOperationException>());
}

[Test]
public void SubtractElementwiseUnequalDimensions_()
{
	float[] data1 = { 1, 2, 3, 4 };
	int[] shape1 = { 4 };
	var tensor1 = new FloatTensor(_data:data1, _shape:shape1);

	float[] data2 = { 1, 2, 3, 4 };
	int[] shape2 = { 2, 2 };
	var tensor2 = new FloatTensor(_data:data2, _shape:shape2);

	Assert.That(() => tensor1.Sub(tensor2, inline: true),
	            Throws.TypeOf<InvalidOperationException>());
}

[Test]
public void SubtractElementwiseUnequalShapes_()
{
	float[] data1 = { 1, 2, 3, 4, 5, 6 };
	int[] shape1 = { 2, 3 };
	var tensor1 = new FloatTensor(_data:data1, _shape:shape1);

	float[] data2 = { 1, 2, 3, 4, 5, 6 };
	int[] shape2 = { 3, 2 };
	var tensor2 = new FloatTensor(_data:data2, _shape:shape2);

	Assert.That(() => tensor1.Sub(tensor2, inline: true),
	            Throws.TypeOf<InvalidOperationException>());
}

[Test]
public void SubtractScalar()
{
	float[] data1 = { -1, 0, 0.1f, 1, float.MaxValue, float.MinValue };
	int[] shape1 = {3, 2};
	var tensor1 = new FloatTensor(_data:data1, _shape:shape1);

	float scalar = 100;

	var tensorSub = tensor1.Sub (scalar);

	for (int i = 0; i < tensorSub.Size; i++)
	{
		Assert.AreEqual (tensor1.Data [i] - scalar, tensorSub.Data [i]);
	}
}

[Test]
public void SubtractScalar_()
{
	float[] data1 = { -1, 0, 1, float.MaxValue, float.MinValue };
	int[] shape1 = {5, 1};
	var tensor1 = new FloatTensor(_data:data1, _shape:shape1);

	float[] data2 = { -101, -100, -99, float.MaxValue-100, float.MinValue-100 };
	int[] shape2 = {5, 1};
	var tensor2 = new FloatTensor(_data:data2, _shape:shape2);

	float scalar = 100;

	tensor1.Sub (scalar, inline: true);

	for (int i = 0; i < tensor1.Size; i++)
	{
		Assert.AreEqual (tensor1.Data[i], tensor2.Data [i]);
	}
}

//
//        [Test]
//        public void ElementwiseSubtractDataOnDifferent()
//        {
//            int[] shape1 = { 2, 3 };
//            var tensor1 = new FloatTensor(shape1, true);
//            int[] shape2 = { 2, 3 };
//            var tensor2 = new FloatTensor(shape2, false);
//
//			Assert.That(() => tensor1.SubtractElementwise(tensor2),
//                Throws.TypeOf<InvalidOperationException>());
//        }

[Test]
public void GetIndexGetIndeces()
{
	float[] data = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
	int[] shape = { 3, 2, 2 };
	var tensor = new FloatTensor(_data:data, _shape:shape);

	long[] idxs = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
	long[][] idxarrays = new long[][] {
		new long[] { 0, 0, 0 },
		new long[] { 0, 1, 0 },
		new long[] { 2, 0, 1 }
	};

	foreach (long i in idxs)
		Assert.AreEqual(i,tensor.GetIndex(tensor.GetIndices(i)));
	foreach (long[] i in idxarrays)
		Assert.AreEqual(i, tensor.GetIndices(tensor.GetIndex(i)));
}

[Test]
public void AddMatrixMultiplyTest()
{
	float[] base1_data = new float[] { 1, 2, 3, 4 };
	int[] base1_shape = new int[] { 2, 2 };
	var base1 = new FloatTensor(_data:base1_data, _shape:base1_shape);

	float[] base2_data = new float[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
	int[] base2_shape = new int[] { 3, 3 };
	var base2 = new FloatTensor( _data:base2_data,_shape:base2_shape );

	float[] data = new float[] { 1, 2, 3, 4, 5, 6 };
	int[] tensor1_shape = new int[] { 2, 3 };
	int[] tensor2_shape = new int[] { 3, 2 };
	var tensor1 = new FloatTensor(_data:data, _shape:tensor1_shape);
	var tensor2 = new FloatTensor(_data:data, _shape:tensor2_shape);

	base1.AddMatrixMultiply(tensor1, tensor2);
	base2.AddMatrixMultiply(tensor2, tensor1);

	for (int i = 0; i < base1_shape[0]; i++)
	{
		for (int j = 0; j < base1_shape[1]; j++)
		{
			float mm_res = base1_data[i * base1_shape[1] + j];
			for (int k = 0; k < tensor1_shape[1]; k++)
			{
				mm_res += tensor1[i, k] * tensor2[k, j];
			}
			Assert.AreEqual(base1[i, j], mm_res);
		}
	}

	for (int i = 0; i < base2_shape[0]; i++)
	{
		for (int j = 0; j < base2_shape[1]; j++)
		{
			float mm_res = base2_data[i * base2_shape[1] + j];
			for (int k = 0; k < tensor2_shape[1]; k++)
			{
				mm_res += tensor2[i, k] * tensor1[k, j];
			}
			Assert.AreEqual(base2[i, j], mm_res);
		}
	}
}

[Test]
public void Tan()
{
	float[] data1 = { 30, 20, 40, 50 };
	int[] shape1 = { 4 };
	var tensor = new FloatTensor(_data:data1, _shape:shape1);

	float[] data2 = {-6.4053312f, 2.23716094f, -1.11721493f, -0.27190061f};
	int[] shape2 = { 4 };
	var expectedTanTensor = new FloatTensor(_data:data2, _shape:shape2);

	var actualTanTensor = tensor.Tan();

	for (int i = 2; i < actualTanTensor.Size; i++)
	{
		Assert.AreEqual (expectedTanTensor.Data[i], actualTanTensor.Data[i]);
	}
}

[Test]
public void Tan_()
{
	float[] data1 = { 30, 20, 40, 50 };
	int[] shape1 = { 4 };
	var tensor = new FloatTensor(_data:data1, _shape:shape1);

	float[] data2 = {-6.4053312f, 2.23716094f, -1.11721493f, -0.27190061f};
	int[] shape2 = { 4 };
	var expectedTanTensor = new FloatTensor(_data:data2, _shape:shape2);

	tensor.Tan (inline: true);

	for (int i = 2; i < tensor.Size; i++)
	{
		Assert.AreEqual (expectedTanTensor.Data[i], tensor.Data[i]);
	}
}

[Test]
public void AddMatrixVectorProductTest()
{
	float[] base_data = new float[] { 1, 2 };
	int[] base_shape = new int[] { 2 };
	var base_vector = new FloatTensor(_data:base_data, _shape:base_shape);

	float[] data1 = { 1, 2, 3, 4 };
	int[] shape1 = new int[] { 2, 2 };
	var matrix = new FloatTensor(_data:data1, _shape:shape1);

	float[] data2 = new float[] { 5, 6 };
	int[] shape2 = new int[] { 2 };
	var vector = new FloatTensor (_data:data2, _shape:shape2);

	base_vector.AddMatrixVectorProduct(matrix, vector);

	float[] expected_data = new float[] { 18, 41 };
	int[] expected_shape = new int[] { 2 };
	var expected_vector = new FloatTensor(_data:expected_data, _shape:expected_shape);

	for (int i = 0; i < expected_vector.Size; i++)
	{
		Assert.AreEqual (expected_vector.Data[i], base_vector.Data[i]);
	}

}

[Test]
public void Tanh()
{
	float[] data1 = { -0.6366f, 0.2718f, 0.4469f, 1.3122f };
	int[] shape1 = { 4 };
	var tensor = new FloatTensor(_data:data1, _shape:shape1);

	float[] data2 = { -0.562580109f, 0.265298963f, 0.419347495f, 0.86483103f };
	int[] shape2 = { 4 };
	var expectedTanhTensor = new FloatTensor(_data:data2, _shape:shape2);

	var actualTanhTensor = tensor.Tanh();

	for (int i = 2; i < actualTanhTensor.Size; i++)
	{
		Assert.AreEqual (expectedTanhTensor.Data[i], actualTanhTensor.Data[i]);
	}
}

[Test]
public void SizeTensor()
{
	float[] data1 = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
	int[] shape1 = { 2, 3, 2 };
	var tensor = new FloatTensor(_data:data1, _shape:shape1);
	var actualSizeTensor = tensor.SizeTensor();

	float[] data2 = { 2, 3, 2 };
	int[] shape2 = { 3 };
	var expectedSizeTensor = new FloatTensor(_data:data2, _shape:shape2);

	for (int i = 0; i < shape1.Length; i++)
	{
		Assert.AreEqual (actualSizeTensor.Data[i], expectedSizeTensor.Data[i]);
	}
}

[Test]
public void Sqrt()
{
	float[] data1 = { float.MaxValue, float.MinValue, 1f, 4f, 5f, 2.3232f, -30f };
	int[] shape1 = { 7 };
	var tensor = new FloatTensor(_data:data1, _shape:shape1);

	float[] data2 = { float.NaN, float.NaN, 1f, 2f, 2.236068f, 1.524205f, float.NaN };
	int[] shape2 = { 7 };
	var expectedTensor = new FloatTensor(_data:data2, _shape:shape2);

	var actualTensor = tensor.Sqrt();

	for (int i = 2; i < expectedTensor.Size; i++)
	{
		Assert.AreEqual (Math.Round(expectedTensor.Data[i], 3), Math.Round(actualTensor.Data[i], 3));
	}
}

[Test]
public void Sinh()
{
	float[] data1 = { -0.6366f, 0.2718f, 0.4469f, 1.3122f };
	int[] shape1 = { 4 };
	var tensor = new FloatTensor(_data:data1, _shape:shape1);

	float[] data2 = { -0.68048f, 0.27516f, 0.46193f, 1.72255f };
	int[] shape2 = { 4 };
	var expectedSinhTensor = new FloatTensor(_data:data2, _shape:shape2);

	var actualSinhTensor = tensor.Sinh();

	for (int i = 2; i < actualSinhTensor.Size; i++)
	{
		var rounded = Decimal.Round((Decimal)actualSinhTensor.Data[i], 5);
		Assert.AreEqual (expectedSinhTensor.Data[i], rounded);
	}
}

[Test]
public void Sinh_()
{
	float[] data1 = { -0.6366f, 0.2718f, 0.4469f, 1.3122f };
	int[] shape1 = { 4 };
	var tensor = new FloatTensor(_data:data1, _shape:shape1);

	float[] data2 = { -0.68048f, 0.27516f, 0.46193f, 1.72255f };
	int[] shape2 = { 4 };
	var expectedSinhTensor = new FloatTensor(_data:data2, _shape:shape2);

	tensor.Sinh (inline: true);

	for (int i = 2; i < tensor.Size; i++)
	{
		var rounded = Decimal.Round((Decimal)tensor.Data[i], 5);
		Assert.AreEqual (expectedSinhTensor.Data[i], rounded);
	}
}

[Test]
public void Trunc()
{
	float[] data = { -0.323232f, 0.323893f, 0.99999f, 1.2323389f };
	int[] shape = { 4 };
	var tensor = new FloatTensor(_data:data, _shape:shape);

	float[] truncatedData = { -0f, 0f, 0f, 1f };
	var expectedTensor = new FloatTensor(_data:truncatedData, _shape:shape);

	var truncatedTensor = tensor.Trunc();
	for (int i = 2; i < truncatedTensor.Size; i++)
	{
		Assert.AreEqual (expectedTensor.Data[i], truncatedTensor.Data[i]);
	}
}
public void Triu_()
{
	int k = 0;

	// Test tensor with dimension < 2
	float[] data1 = { 1, 2, 3, 4, 5, 6 };
	int[] shape1 = { 6 };
	var tensor1 = new FloatTensor(_data:data1, _shape:shape1);
	Assert.That(() => tensor1.Triu_(k),
	            Throws.TypeOf<InvalidOperationException>());

	// Test tensor with dimension > 2
	float[] data2 = { 1, 2, 3, 4, 5, 6, 7, 8 };
	int[] shape2 = { 2, 2, 2 };
	var tensor2 = new FloatTensor(_data:data2, _shape:shape2);

	Assert.That(() => tensor2.Triu_(k),
	            Throws.TypeOf<InvalidOperationException>());

	// Test dim = 2, k = 0
	k = 0;
	float[] data3 = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
	int[] shape3 = { 3, 3 };
	var tensor3 = new FloatTensor(_data:data3, _shape:shape3);
	tensor3.Triu_(k);
	float[] data3Triu = { 1, 2, 3, 0, 5, 6, 0, 0, 9 };
	var tensor3Triu = new FloatTensor(_data:data3Triu, _shape:shape3);
	for (int i = 0; i < tensor3.Size; i++)
	{
		Assert.AreEqual (tensor3.Data[i], tensor3Triu.Data[i]);
	}

	// Test dim = 2, k = 2
	k = 2;
	float[] data4 = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
	int[] shape4 = { 3, 3 };
	var tensor4 = new FloatTensor(_data:data4, _shape:shape4);
	tensor4.Triu_(k);
	float[] data4Triu = { 0, 0, 3, 0, 0, 0, 0, 0, 0 };
	var tensor4Triu = new FloatTensor(_data:data4Triu, _shape:shape4);
	for (int i = 0; i < tensor4.Size; i++)
	{
		Assert.AreEqual (tensor4.Data[i], tensor4Triu.Data[i]);
	}

	// Test dim = 2, k = -1
	k = -1;
	float[] data5 = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
	int[] shape5 = { 3, 3 };
	var tensor5 = new FloatTensor(_data:data5, _shape:shape5);
	tensor5.Triu_(k);
	float[] data5Triu = { 1, 2, 3, 4, 5, 6, 0, 8, 9 };
	var tensor5Triu = new FloatTensor(_data:data5Triu, _shape:shape5);
	for (int i = 0; i < tensor5.Size; i++)
	{
		Assert.AreEqual (tensor5.Data[i], tensor5Triu.Data[i]);
	}

	// Test dim = 2, k >> ndims
	k = 100;
	float[] data6 = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
	int[] shape6 = { 3, 3 };
	var tensor6 = new FloatTensor(_data:data6, _shape:shape6);
	tensor6.Triu_(k);
	float[] data6Triu = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };
	var tensor6Triu = new FloatTensor(_data:data6Triu, _shape:shape6);
	for (int i = 0; i < tensor6.Size; i++)
	{
		Assert.AreEqual (tensor6.Data[i], tensor6Triu.Data[i]);
	}

	// Test dim = 2, k << ndims
	k = -100;
	float[] data7 = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
	int[] shape7 = { 3, 3 };
	var tensor7 = new FloatTensor(_data:data7, _shape:shape7);
	tensor7.Triu_(k);
	float[] data7Triu = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
	var tensor7Triu = new FloatTensor(_data:data7Triu, _shape:shape7);
	for (int i = 0; i < tensor7.Size; i++)
	{
		Assert.AreEqual (tensor7.Data[i], tensor7Triu.Data[i]);
	}
}

public void IsContiguous()
{
	float[] data = new float[] { 1, 2, 3, 4, 5, 6 };
	int[] shape = new int[] { 2, 3 };
	var tensor = new FloatTensor(_data:data, _shape:shape);
	Assert.AreEqual(tensor.IsContiguous(), true);
	var transposedTensor = tensor.Transpose();
	Assert.AreEqual(transposedTensor.IsContiguous(), false);
}

// TODO: AddMatrixMultiplyTests when implemented on CPU
// TODO: MultiplyDerivative when implemented on CPU
}
}
