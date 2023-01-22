using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NUnit.Framework;

namespace ToyGPT.NeuralNetwork.Tests
{
	internal class LossCategoricalCrossEntropyTests
	{
		[Test]
		public void Test1()
		{
			var inputs = new float[,] {
				{ 0.7f, 0.1f, 0.2f },
				{ 0.1f, 0.5f, 0.4f },
				{ 0.02f, 0.9f, 0.08f },
			};
			var targets = new int[] { 0, 1, 1 };
			var expected = new float[] { 0.35667494f, 0.69314818f, 0.10536052f };
			var actual = ArrayFactory.NewFromWidth(inputs);
			var loss = new LossCategoricalCrossEntropy();
			loss.Forward(inputs, targets, actual);
			Assert.That(actual, Is.EqualTo(expected).Within(0.00001f));
		}

		[Test]
		public void BackwardTest1()
		{
			var inputs = new float[,] {
				{ 0.7f, 0.1f, 0.2f },
				{ 0.1f, 0.5f, 0.4f },
				{ 0.02f, 0.9f, 0.08f },
			};
			var targets = new int[] { 0, 1, 1 };
			var expected = new float[,] {
				{ -0.47619048f, 0, 0},
				{ 0, -0.66666667f, 0 },
				{ 0, -0.37037037f, 0 },
			};
			var actual = ArrayFactory.NewSameSize(inputs);
			var loss = new LossCategoricalCrossEntropy();
			loss.Backward(inputs, targets, actual);
			Assert.That(actual, Is.EqualTo(expected).Within(0.00001f));
		}

		[Test]
		public void BackwardTest2()
		{
			var inputs = new float[,] {
				{ 0.7f, 0.1f, 0.2f },
				{ 0.1f, 0.5f, 0.4f },
			};
			var targets = new int[] { 0, 1 };
			var expected = new float[,] {
				{ -0.71428571f , 0, 0},
				{ 0, -1, 0 },
			};
			var actual = ArrayFactory.NewSameSize(inputs);
			var loss = new LossCategoricalCrossEntropy();
			loss.Backward(inputs, targets, actual);
			Assert.That(actual, Is.EqualTo(expected).Within(0.00001f));
		}
	}
}
