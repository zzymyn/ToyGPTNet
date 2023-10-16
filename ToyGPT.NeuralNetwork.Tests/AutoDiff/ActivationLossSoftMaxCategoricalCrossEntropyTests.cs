using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using ToyGPT.NeuralNetwork.Activations;
using ToyGPT.NeuralNetwork.AutoDiff;
using static ToyGPT.NeuralNetwork.AutoDiff.ExpressionBuilder;

namespace ToyGPT.NeuralNetwork.Tests.AutoDiff;

internal class ActivationLossSoftMaxCategoricalCrossEntropyTests
{
	[Test]
	public void ForwardTest1()
	{
		var inputs = new float[,]
		{
			{ 0.7f, 0.1f, 0.2f },
			{ 0.1f, 0.5f, 0.4f },
			{ 0.02f, 0.9f, 0.08f },
		};
		var targets = new int[] { 0, 1, 1 };

		var vInputs = V(inputs);
		var softmax = Softmax(vInputs);
		var cce = CategoricalCrossEntropy(softmax, C(targets));

		var ctx = new ExpressionContext();
		var result1 = ctx.GetResult(cce).ToArray();

		var smOut = ArrayFactory.NewSameSize(inputs);
		var result2 = new float[targets.Length];
		MMath.Softmax(inputs, smOut);
		MMath.CategoricalCrossEntropy(smOut, targets, result2);

		Assert.That(result1, Is.EqualTo(result2).Within(0.00001f));
	}

	[Test]
	public void BackwardTest1()
	{
		var inputs = new float[,]
		{
			{ 0.7f, 0.1f, 0.2f },
			{ 0.1f, 0.5f, 0.4f },
			{ 0.02f, 0.9f, 0.08f },
		};
		var targets = new int[] { 0, 1, 1 };
		var seed = new float[] { 1.0f, 1.0f, 1.0f };

		var vInputs = V(inputs);
		var softmax = Softmax(vInputs);
		var cce = CategoricalCrossEntropy(softmax, C(targets));

		var ctx = new ExpressionContext();
		ctx.GetResult(cce).ToArray();
		cce.Backward(ctx, seed);
		var grad1 = vInputs.GetGradient(ctx).ToArray();

		var smOut = ArrayFactory.NewSameSize(inputs);
		var result2 = new float[targets.Length];
		var grad2 = ArrayFactory.NewSameSize(inputs);
		MMath.Softmax(inputs, smOut);
		MMath.CategoricalCrossEntropy(smOut, targets, result2);
		MMath.DSoftMaxCategoricalCrossEntropy(targets, smOut, grad2);

		Assert.That(grad1, Is.EqualTo(grad2).Within(0.00001f));
	}

}
