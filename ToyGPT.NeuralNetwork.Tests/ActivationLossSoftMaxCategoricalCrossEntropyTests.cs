using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using ToyGPT.NeuralNetwork.ActivationLoss;
using ToyGPT.NeuralNetwork.Activations;

namespace ToyGPT.NeuralNetwork.Tests;

internal class ActivationLossSoftMaxCategoricalCrossEntropyTests
{
	[Test]
	public void BackwardTest1()
	{
		var dValues = new float[,] {
			{ 0.7f, 0.1f, 0.2f },
			{ 0.1f, 0.5f, 0.4f },
			{ 0.02f, 0.9f, 0.08f },
		};
		var targets = new int[] { 0, 1, 1 };
		var dInputs = ArrayFactory.NewSameSize(dValues);
		var expected = new float[,]
		{
			{-0.1f       ,  0.03333333f,  0.06666667f},
			{ 0.03333333f, -0.16666667f,  0.13333333f},
			{ 0.00666667f, -0.03333333f,  0.02666667f },
		};
		MMath.DSoftMaxCategoricalCrossEntropy(targets, dValues, dInputs);
		Assert.That(dInputs, Is.EqualTo(expected).Within(0.00001f));
	}
	
	[Test]
	public void BackwardTest2()
	{
		var dValues = new float[,] {
			{ 0.7f, 0.1f, 0.2f },
			{ 0.1f, 0.5f, 0.4f },
		};
		var targets = new int[] { 0, 2 };
		var dInputsA = ArrayFactory.NewSameSize(dValues);
		var dInputsB = ArrayFactory.NewSameSize(dValues);

		MMath.DSoftMaxCategoricalCrossEntropy(targets, dValues, dInputsA);

		var lossDInput = ArrayFactory.NewSameSize(dValues);

		MMath.DCategoricalCrossEntropy(dValues, targets, lossDInput);
		MMath.DSoftmax(dValues, lossDInput, dInputsB);

		Assert.That(dInputsA, Is.EqualTo(dInputsB).Within(0.00001f));
	}
	
	[Test]
	public void BackwardTest3()
	{
		var softmaxR = new float[,] {
			{ 0.7f, 0.1f, 0.2f },
			{ 0.1f, 0.5f, 0.4f },
		};
		var targets = new int[] { 0, 2 };
		var dCce = new float[] { -1.0f, 2.5f };
		var dInputsA = ArrayFactory.NewSameSize(softmaxR);
		var dInputsB = ArrayFactory.NewSameSize(softmaxR);

		MMath.DSoftMaxCategoricalCrossEntropy(targets, softmaxR, dCce, dInputsA);

		var dSoftmaxR = ArrayFactory.NewSameSize(softmaxR);

		MMath.DCategoricalCrossEntropy(softmaxR, targets, dCce, dSoftmaxR);
		MMath.DSoftmax(softmaxR, dSoftmaxR, dInputsB);

		Assert.That(dInputsA, Is.EqualTo(dInputsB).Within(0.00001f));
	}
}
