using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Newtonsoft.Json.Linq;
using NUnit.Framework;

namespace ToyGPT.NeuralNetwork.Tests
{
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
			var loss = new ActivationLossSoftMaxCategoricalCrossEntropy();
			loss.Backward(targets, dValues, dInputs);
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
			
			var loss = new ActivationLossSoftMaxCategoricalCrossEntropy();
			loss.Backward(targets, dValues, dInputsA);

			var loss2 = new LossCategoricalCrossEntropy();
			var lossDInput = ArrayFactory.NewSameSize(dValues);
			var activation = new ActivationSoftMax();
			loss2.Backward(dValues, targets, lossDInput);
			activation.Backward(dValues, lossDInput, dInputsB);

			Assert.That(dInputsA, Is.EqualTo(dInputsB).Within(0.00001f));
		}
	}
}
