using NUnit.Framework;

namespace ToyGPT.NeuralNetwork.Tests
{
	public class ActivationReLUTests
	{
		[Test]
		public void Test1()
		{
			var inputs = new float[,] { { 0, 2, -1, 3.3f, -2.7f, 1.1f, 2.2f, -100f } };
			var outputs = new float[1, 8];
			var expected = new float[,] { { 0, 2, 0, 3.3f, 0, 1.1f, 2.2f, 0 } };
			var act = new ActivationReLU();
			act.Forward(inputs, outputs);
			Assert.That(outputs, Is.EqualTo(expected).Within(0.00001f));
		}

		[Test]
		public void Test2()
		{
			var inputs = new float[,] {
				{ 0, 2, -1, 3.3f, -2.7f, 1.1f, 2.2f, -100f },
				{ 1, -1, 1, -1, 1, -1, 1, -1 },
			};
			var outputs = new float[2, 8];
			var expected = new float[,] {
				{ 0, 2, 0, 3.3f, 0, 1.1f, 2.2f, 0 },
				{ 1, 0, 1, 0, 1, 0, 1, 0 },
			};
			var act = new ActivationReLU();
			act.Forward(inputs, outputs);
			Assert.That(outputs, Is.EqualTo(expected).Within(0.00001f));
		}
	}
}