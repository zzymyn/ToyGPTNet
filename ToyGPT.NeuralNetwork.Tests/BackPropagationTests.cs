using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CommunityToolkit.HighPerformance;
using NUnit.Framework;
using ToyGPT.NeuralNetwork.Activations;
using ToyGPT.NeuralNetwork.Layers;

namespace ToyGPT.NeuralNetwork.Tests;

internal class BackPropagationTests
{
	[Test]
	public void LayerAndReluTest()
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
		var biases = new float[] { 2, 3, 0.5f };

		var layer = new LinearWeightsTransposedWithBias(weights, biases);
		var relu = new ReLU();

		var layerOutput = layer.Forward(inputs);
		var reluOutput = relu.Forward(layerOutput.Span);
		var dRelu = relu.Backward(layerOutput.Span, reluOutput.Span);
		layer.Backward(inputs, dRelu);

		var dWeights = layer.DWeights.Span;
		var dBiases = layer.DBiases.Span;

		{
			var yMax = weights.GetLength(0);
			var xMax = weights.GetLength(1);
			for (int y = 0; y < yMax; ++y)
			{
				for (int x = 0; x < xMax; ++x)
				{
					weights[y, x] += -0.001f * dWeights[y, x];
				}
			}
		}

		{
			var iMax = biases.Length;
			for (int i = 0; i < iMax; ++i)
			{
				biases[i] += -0.001f * dBiases[i];
			}
		}

		var expectedWeights = new float[,]
		{
			{ 0.179515f,   0.742093f,  -0.510153f,   0.971328f },
			{ 0.5003665f, -0.9152577f,  0.2529017f, -0.5021842f },
			{  -0.262746f,  -0.2758402f,  0.1629592f,  0.8636583f },
		};
		var expectedBiases = new float[] { 1.98489f, 2.997739f, 0.497389f };

		Assert.That(weights, Is.EqualTo(expectedWeights).Within(0.00001f));
		Assert.That(biases, Is.EqualTo(expectedBiases).Within(0.00001f));
	}
}
