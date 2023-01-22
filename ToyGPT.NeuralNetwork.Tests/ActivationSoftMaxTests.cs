using System;
using NUnit.Framework;

namespace ToyGPT.NeuralNetwork.Tests
{
	public class ActivationSoftMaxTests
	{
		[Test]
		public void Test1()
		{
			var inputs = new float[,] {
				{ 4.8f, 1.21f, 2.385f }
			};
			var outputs = ArrayFactory.NewSameSize(inputs);
			var expected = new float[,] {
				{ 0.89528266f, 0.02470831f, 0.08000903f }
			};
			var act = new ActivationSoftMax();
			act.Forward(inputs, outputs);
			Assert.That(outputs, Is.EqualTo(expected).Within(0.00001f));
		}

		[Test]
		public void Test2()
		{
			var inputs = new float[,] {
				{ 4.8f, 1.21f, 2.385f },
				{ 8.9f, -1.81f, 0.2f },
				{ 1.41f, 1.051f, 0.026f },
			};
			var outputs = ArrayFactory.NewSameSize(inputs);
			var expected = new float[,] {
				{ 0.895282f, 0.024708f, 0.080009f },
				{ 9.99811129e-1f, 2.23163963e-5f, 1.66554348e-4f },
				{ 5.13097164e-1f, 3.58333899e-1f, 1.28568936e-1f },
			};
			var act = new ActivationSoftMax();
			act.Forward(inputs, outputs);
			Assert.That(outputs, Is.EqualTo(expected).Within(0.00001f));
		}

		[Test]
		public void BackardTest1()
		{
			var inputs = new float[,]
			{
				{ 0.7f, 0.1f, 0.2f },
				{ 0.1f, 0.5f, 0.4f },
				{ 0.02f, 0.9f, 0.08f },
			};
			var dValues = new float[,]
			{
				{ -0.47619048f, 0, 0},
				{ 0, -0.66666667f, 0 },
				{ 0, -0.37037037f, 0 },
			};
			var dInputs = ArrayFactory.NewSameSize(dValues);
			var expected = new float[,]
			{
				{-0.1f       ,  0.03333333f,  0.06666667f},
				{ 0.03333333f, -0.16666667f,  0.13333333f},
				{ 0.00666667f, -0.03333333f,  0.02666667f },
			};
			var act = new ActivationSoftMax();
			act.Backward(inputs, dValues, dInputs);
			Assert.That(dInputs, Is.EqualTo(expected).Within(0.00001f));
		}

		[Test]
		public void BackardTest2()
		{
			var inputs = new float[,]
			{
				{ 0.7f, 0.1f, 0.2f },
				{ 0.1f, 0.5f, 0.4f },
				{ 0.02f, 0.9f, 0.08f },
			};
			var dValues = new float[,]
			{
				{ 0.1f, 0.2f, 0.3f},
				{ -0.1f, -0.2f, -0.3f },
				{ 0.1f, -0.2f, 0.3f },
			};
			var dInputs = ArrayFactory.NewSameSize(dValues);
			var expected = new float[,]
			{
				{-0.035f,    0.005f,    0.03f},
				{0.013f,    0.015f,   -0.028f},
				{0.00508f, -0.0414f,   0.03632f},
			};
			var act = new ActivationSoftMax();
			act.Backward(inputs, dValues, dInputs);
			Assert.That(dInputs, Is.EqualTo(expected).Within(0.00001f));
		}
		
		[Test]
		public void BackardTest3()
		{
			var inputs = new float[,]
			{
				{ 0.2f, 0.7f, 0.1f },
			};
			var dValues = new float[,]
			{
				{ 0.5f, 0.2f, 0.3f},
			};
			var dInputs = ArrayFactory.NewSameSize(dValues);
			var expected = new float[,]
			{
				{0.046f, -0.049f,  0.003f},
			};
			var act = new ActivationSoftMax();
			act.Backward(inputs, dValues, dInputs);
			Assert.That(dInputs, Is.EqualTo(expected).Within(0.00001f));
		}
	}
}