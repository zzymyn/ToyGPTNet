using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NUnit.Framework;

namespace ToyGPT.NeuralNetwork.Tests
{
	internal class LayerDenseTests
	{
		[Test]
		public void Test1()
		{
			var layer = new LayerDense(4, 3, new float[,]
			{
				{ 0.2f, 0.8f, -0.5f, 1.0f },
				{ 0.5f, -0.91f, 0.26f, -0.5f },
				{ -0.26f, -0.27f, 0.17f, 0.87f },
			}, new float[] { 2.0f, 3.0f, 0.5f });

			var inputs = new float[,] { { 1.0f, 2.0f, 3.0f, 2.5f } };
			var expected = new float[,] { { 4.8f, 1.21f, 2.385f } };
			var output = new float[1, 3];
			layer.Forward(inputs, output);
			Assert.That(output, Is.EqualTo(expected).Within(0.00001f));
		}
		
		[Test]
		public void Test2()
		{
			var layer = new LayerDense(4, 3, new float[,]
			{
				{ 0.2f, 0.8f, -0.5f, 1.0f },
				{ 0.5f, -0.91f, 0.26f, -0.5f },
				{ -0.26f, -0.27f, 0.17f, 0.87f },
			}, new float[] { 2.0f, 3.0f, 0.5f });

			var inputs = new float[,] {
				{ 1.0f, 2.0f, 3.0f, 2.5f },
				{ 2, 5, -1, 2 },
				{ -1.5f, 2.7f, 3.3f, -0.8f },
			};
			var expected = new float[,] {
				{ 4.8f, 1.21f, 2.385f },
				{ 8.9f, -1.81f, 0.2f },
				{ 1.41f, 1.051f, 0.026f },
			};
			var output = new float[3, 3];
			layer.Forward(inputs, output);
			Assert.That(output, Is.EqualTo(expected).Within(0.00001f));
		}
	}
}
