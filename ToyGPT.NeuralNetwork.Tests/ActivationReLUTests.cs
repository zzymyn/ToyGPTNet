using NUnit.Framework;

namespace ToyGPT.NeuralNetwork.Tests
{
	public class ActivationReLUTests
	{
		[Test]
		public void ForwardTest1()
		{
			var inputs = new float[,] { { 0, 2, -1, 3.3f, -2.7f, 1.1f, 2.2f, -100f } };
			var outputs = ArrayFactory.NewSameSize(inputs);
			var expected = new float[,] { { 0, 2, 0, 3.3f, 0, 1.1f, 2.2f, 0 } };
			var act = new ActivationReLU();
			act.Forward(inputs, outputs);
			Assert.That(outputs, Is.EqualTo(expected).Within(0.00001f));
		}

		[Test]
		public void ForwardTest2()
		{
			var inputs = new float[,] {
				{ 0, 2, -1, 3.3f, -2.7f, 1.1f, 2.2f, -100f },
				{ 1, -1, 1, -1, 1, -1, 1, -1 },
			};
			var outputs = ArrayFactory.NewSameSize(inputs);
			var expected = new float[,] {
				{ 0, 2, 0, 3.3f, 0, 1.1f, 2.2f, 0 },
				{ 1, 0, 1, 0, 1, 0, 1, 0 },
			};
			var act = new ActivationReLU();
			act.Forward(inputs, outputs);
			Assert.That(outputs, Is.EqualTo(expected).Within(0.00001f));
		}

		[Test]
		public void BackwardTest1()
		{
			var inputs = new float[,] { { 0, 2, -1, 3.3f, -2.7f, 1.1f, 2.2f, -100f } };
			var dValues = new float[,] { { 1, 2, 3, 4, 5, 6, 7, 8 } };
			var dInputs = ArrayFactory.NewSameSize(inputs);
			var expected = new float[,] { { 0, 2, 0, 4, 0, 6, 7, 0 } };
			var act = new ActivationReLU();
			act.Backward(inputs, dValues, dInputs);
			Assert.That(dInputs, Is.EqualTo(expected).Within(0.00001f));
		}

		[Test]
		public void BackwardTest2()
		{
			var inputs = new float[,]
			{
				{ 1, 2, -3, -4 },
				{ 2, -7, -1, 3 },
				{-1, 2, 5, -1 },
			};
			var dValues = new float[,]
			{
				{ 1, 2, 3, 4 },
				{ 5, 6, 7, 8 },
				{ 9, 10, 11, 12 },
			};
			var dInputs = ArrayFactory.NewSameSize(inputs);
			var expected = new float[,]
			{
				{ 1, 2, 0, 0 },
				{ 5, 0, 0, 8 },
				{ 0, 10, 11, 0 },
			};
			var act = new ActivationReLU();
			act.Backward(inputs, dValues, dInputs);
			Assert.That(dInputs, Is.EqualTo(expected).Within(0.00001f));
		}
	}
}