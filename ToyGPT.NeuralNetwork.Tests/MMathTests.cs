using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NUnit.Framework;

namespace ToyGPT.NeuralNetwork.Tests;

public class MMathTests
{
	[Test]
	[TestCase(new float[] { -3.0f, -2.0f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 2.0f, 3.0f }, new float[] { -0.00363739f, -0.04540231f, -0.15880801f, -0.15428599f, 0.0f, 0.34571401f, 0.84119199f, 1.95459769f, 2.99636261f })]
	public void TestGeLU(float[] input, float[] expected)
	{
		var results = new float[input.Length];
		MMath.GeLU(input, results);
		Assert.That(results, Is.EqualTo(expected).Within(1e-5f));
	}

	[Test]
	[TestCase(new float[] { 0.0f, 1.0f, 2.0f, 3.0f }, new float[] { 0.0320586f, 0.08714432f, 0.23688282f, 0.64391426f })]
	[TestCase(new float[] { 0.0f, 1.0f, 2000.0f, 3.0f }, new float[] { 0.0f, 0.0f, 1.0f, 0.0f })]
	public void TestSoftmax(float[] input, float[] expected)
	{
		var results = new float[input.Length];
		MMath.Softmax(input, results);
		Assert.That(results, Is.EqualTo(expected).Within(1e-5f));
	}

	[Test]
	[TestCase(new float[] { 2, 2, 3 }, new float[] { -0.70709087f, -0.70709087f, 1.41418174f })]
	[TestCase(new float[] { -5, 0, 1 }, new float[] { -1.39700038f, 0.50800014f, 0.88900024f })]
	public void TestLayerNormalization(float[] input, float[] expected)
	{
		var results = new float[input.Length];
		var g = new float[input.Length];
		var b = new float[input.Length];
		Array.Fill(g, 1.0f);
		MMath.LayerNormalization(input, g, b, results);
		Assert.That(results, Is.EqualTo(expected).Within(1e-5f));
	}
}
