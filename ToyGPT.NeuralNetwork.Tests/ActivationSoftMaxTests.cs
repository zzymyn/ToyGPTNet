using NUnit.Framework;

namespace ToyGPT.NeuralNetwork.Tests
{
	public class ActivationSoftMaxTests
	{
		[Test]
		public void Test1()
		{
			var values = new float[,] { { 4.8f, 1.21f, 2.385f } };
			var expected = new float[,] { { 0.895282f, 0.024708f, 0.080009f} };
			var act = new ActivationSoftMax();
			act.Forward(values);
			Assert.That(values, Is.EqualTo(expected).Within(0.00001f));
		}
	}
}