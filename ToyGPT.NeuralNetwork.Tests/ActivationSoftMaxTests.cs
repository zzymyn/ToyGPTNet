using System;
using NUnit.Framework;

namespace ToyGPT.NeuralNetwork.Tests
{
	public class ActivationSoftMaxTests
	{
		[Test]
		public void Test1()
		{
			var values = new float[,] {
				{ 4.8f, 1.21f, 2.385f }
			};
			var expected = new float[,] {
				{ 0.89528266f, 0.02470831f, 0.08000903f }
			};
			var act = new ActivationSoftMax();
			act.Forward(values);
			Assert.That(values, Is.EqualTo(expected).Within(0.00001f));
		}

		[Test]
		public void Test2()
		{
			var values = new float[,] {
				{ 4.8f, 1.21f, 2.385f },
				{ 8.9f, -1.81f, 0.2f },
				{ 1.41f, 1.051f, 0.026f },
			};
			var expected = new float[,] {
				{ 0.895282f, 0.024708f, 0.080009f },
				{ 9.99811129e-1f, 2.23163963e-5f, 1.66554348e-4f },
				{ 5.13097164e-1f, 3.58333899e-1f, 1.28568936e-1f },
			};
			var act = new ActivationSoftMax();
			act.Forward(values);
			Assert.That(values, Is.EqualTo(expected).Within(0.00001f));
		}
	}
}