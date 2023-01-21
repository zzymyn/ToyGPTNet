using NUnit.Framework;

namespace ToyGPT.NeuralNetwork.Tests
{
	public class ActivationReLUTests
	{
		[Test]
		public void Test1()
		{
			var values = new float[,] { { 0, 2, -1, 3.3f, -2.7f, 1.1f, 2.2f, -100f } };
			var expected = new float[,] { { 0, 2, 0, 3.3f, 0, 1.1f, 2.2f, 0 } };
			var act = new ActivationReLU();
			act.Forward(values);
			Assert.That(values, Is.EqualTo(expected).Within(0.00001f));
		}
	}
}