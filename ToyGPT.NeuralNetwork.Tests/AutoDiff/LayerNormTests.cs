using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NUnit.Framework;
using ToyGPT.NeuralNetwork.AutoDiff;
using ToyGPT.NeuralNetwork.Layers;
using static ToyGPT.NeuralNetwork.AutoDiff.ExpressionBuilder;

namespace ToyGPT.NeuralNetwork.Tests.AutoDiff;

internal class LayerNormTests
{
	[Test]
	public void ForwardTest1()
	{
		var inputs = new float[] { 0.2f, 0.5f, 1.2f, -1.6f, 0.5f };
		var seed = new float[] { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f };
		var expected = new float[] { 0.04258212f, 0.36194799f, 1.10713502f, -1.87361312f, 0.36194799f };
		var ctx = new ExpressionContext();
		var vInputs = V(inputs);
		var layerNorm = LayerNorm(vInputs);

		var result = ctx.GetResult(layerNorm).ToArray();

		Assert.That(result, Is.EqualTo(expected).Within(0.0001f));
	}
}
