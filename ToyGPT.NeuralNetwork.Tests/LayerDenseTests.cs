using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NUnit.Framework;
using ToyGPT.NeuralNetwork.Layers;

namespace ToyGPT.NeuralNetwork.Tests;

internal class LayerDenseTests
{
	[Test]
	public void ForwardTest1()
	{
		var weights = new float[,]
		{
			{ 0.2f, 0.8f, -0.5f, 1.0f },
			{ 0.5f, -0.91f, 0.26f, -0.5f },
			{ -0.26f, -0.27f, 0.17f, 0.87f },
		};
		var biases = new float[] { 2.0f, 3.0f, 0.5f };
		var inputs = new float[,] { { 1.0f, 2.0f, 3.0f, 2.5f } };
		var expected = new float[,] { { 4.8f, 1.21f, 2.385f } };

		var layer = new LinearWeightsTransposedWithBias(weights, biases);

		var output = layer.Forward(inputs);
		Assert.That(output.ToArray(), Is.EqualTo(expected).Within(0.00001f));
	}

	[Test]
	public void ForwardTest2()
	{
		var inputs = new float[,] {
			{ 1.0f, 2.0f, 3.0f, 2.5f },
			{ 2, 5, -1, 2 },
			{ -1.5f, 2.7f, 3.3f, -0.8f },
		};
		var weights = new float[,]
		{
			{ 0.2f, 0.8f, -0.5f, 1.0f },
			{ 0.5f, -0.91f, 0.26f, -0.5f },
			{ -0.26f, -0.27f, 0.17f, 0.87f },
		};
		var biases = new float[] { 2.0f, 3.0f, 0.5f };
		var expected = new float[,] {
			{ 4.8f, 1.21f, 2.385f },
			{ 8.9f, -1.81f, 0.2f },
			{ 1.41f, 1.051f, 0.026f },
		};

		var layer = new LinearWeightsTransposedWithBias(weights, biases);

		var output = layer.Forward(inputs);
		Assert.That(output.ToArray(), Is.EqualTo(expected).Within(0.00001f));
	}

	[Test]
	public void BackwardTest1()
	{
		var inputs = new float[,] {
			{ 1.0f, 2.0f, 3.0f, 2.5f },
			{ 2, 5, -1, 2 },
			{ -1.5f, 2.7f, 3.3f, -0.8f },
		};
		var weights = new float[,]
		{
			{ 0.2f, 0.8f, -0.5f, 1.0f },
			{ 0.5f, -0.91f, 0.26f, -0.5f },
			{ -0.26f, -0.27f, 0.17f, 0.87f },
		};
		var biases = new float[] { 1, 2, 3 };
		var dValues = new float[,] {
			{ 1, 1, 1 },
			{ 2, 2, 2 },
			{ 3, 3, 3 },
		};
		var expectedDInputs = new float[,] {
			{ 0.44f, -0.38f, -0.07f, 1.37f },
			{ 0.88f, -0.76f, -0.14f, 2.74f },
			{ 1.32f, -1.14f, -0.21f, 4.11f },
		};
		var expectedDWeights = new float[,]
		{
			{ 0.5f, 20.1f, 10.9f, 4.1f, },
			{ 0.5f, 20.1f, 10.9f, 4.1f, },
			{ 0.5f, 20.1f, 10.9f, 4.1f, },
		};
		var expectedDBiases = new float[] { 6, 6, 6 };

		var layer = new LinearWeightsTransposedWithBias(weights, biases);
		layer.Forward(inputs);
		layer.Backward(inputs, dValues);
		var dInputs = layer.DInputs.ToArray();
		var dWeights = layer.DWeights.ToArray();
		var dBiases = layer.DBiases.ToArray();

		Assert.That(dInputs, Is.EqualTo(expectedDInputs).Within(0.00001f));
		Assert.That(dWeights, Is.EqualTo(expectedDWeights).Within(0.00001f));
		Assert.That(dBiases, Is.EqualTo(expectedDBiases).Within(0.00001f));
	}

	[Test]
	public void BackwardTest2()
	{
		var inputs = new float[,] {
			{ 1.0f, 2.0f, 3.0f, 2.5f },
			{ 2, 5, -1, 2 },
		};
		var weights = new float[,]
		{
			{ 0.2f, 0.8f, -0.5f, 1.0f },
			{ 0.5f, -0.91f, 0.26f, -0.5f },
			{ -0.26f, -0.27f, 0.17f, 0.87f },
		};
		var biases = new float[] { 1, 2, 3 };
		var dValues = new float[,] {
			{ 1, 0, 1 },
			{ 2, 1, 0.5f },
		};
		var expectedDInputs = new float[,] {
			{ -0.06f, 0.53f, -0.33f, 1.87f, },
			{ 0.77f, 0.555f, -0.655f, 1.935f, },
		};
		var expectedDWeights = new float[,]
		{
			{ 5,  12,   1,   6.5f },
			{ 2,   5,  -1,   2 },
			{ 2,   4.5f,  2.5f,  3.5f },
		};
		var expectedDBiases = new float[] { 3, 1, 1.5f };

		var layer = new LinearWeightsTransposedWithBias(weights, biases);
		layer.Forward(inputs);
		layer.Backward(inputs, dValues);
		var dInputs = layer.DInputs.ToArray();
		var dWeights = layer.DWeights.ToArray();
		var dBiases = layer.DBiases.ToArray();

		Assert.That(dInputs, Is.EqualTo(expectedDInputs).Within(0.00001f));
		Assert.That(dWeights, Is.EqualTo(expectedDWeights).Within(0.00001f));
		Assert.That(dBiases, Is.EqualTo(expectedDBiases).Within(0.00001f));
	}
}
