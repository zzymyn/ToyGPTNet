using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CommunityToolkit.HighPerformance;
using NUnit.Framework;
using ToyGPT.NeuralNetwork.AutoDiff;
using ToyGPT.NeuralNetwork.Layers;
using static ToyGPT.NeuralNetwork.AutoDiff.ExpressionBuilder;

namespace ToyGPT.NeuralNetwork.Tests.AutoDiff;

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

		var vInputs = V(inputs);
		var vWeights = V(weights);
		var vBiases = V(biases);
		var op = ReLU(MatMulMTAddR(vInputs, vWeights, vBiases));
		var ctx = new ExpressionContext();

		var reluOutput = ctx.GetResult(op);
		op.Backward(ctx, reluOutput);

		var dWeights = vWeights.GetGradient(ctx).ToArray();
		var dBiases = vBiases.GetGradient(ctx).ToArray();

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
