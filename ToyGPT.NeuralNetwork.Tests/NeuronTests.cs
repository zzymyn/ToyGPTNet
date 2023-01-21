using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NUnit.Framework;

namespace ToyGPT.NeuralNetwork.Tests
{
	internal class NeuronTests
	{
		[Test]
		public void Test1()
		{
			var inputs = new float[] { 1.2f, 5.1f, 2.1f };
			var weights = new float[] { 3.1f, 2.1f, 8.7f };
			var bias = 3.0f;
			var expected = 35.7f;
			var actual = Neuron.Forward(inputs, weights, bias);
			Assert.That(actual, Is.EqualTo(expected).Within(0.00001f));
		}
	}
}
