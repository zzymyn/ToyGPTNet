using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NUnit.Framework;

namespace ToyGPT.NeuralNetwork.Tests
{
	internal class AccuracyCategoricalTests
	{
		[Test]
		public void Test1()
		{
			var inputs = new float[,] {
				{ 0.7f, 0.2f, 0.1f },
				{ 0.5f, 0.1f, 0.4f },
				{ 0.02f, 0.9f, 0.08f },
			};
			var targets = new int[] { 0, 1, 1 };
			var expected = 0.6666667f;
			var actual = AccuracyCategorical.Compute(inputs, targets);
			Assert.That(actual, Is.EqualTo(expected).Within(0.00001f));
		}
	}
}
