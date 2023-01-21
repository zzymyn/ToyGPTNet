using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CommunityToolkit.HighPerformance;

namespace ToyGPT.NeuralNetwork
{
	public class ActivationSoftMax
	{
		public void Forward(Span2D<float> inputs)
		{
			var sum = 0.0f;

			foreach (ref var a in inputs)
			{
				a = (float)Math.Exp(a);
				sum += a;
			}

			var invSum = 1.0f / sum;

			foreach (ref var a in inputs)
			{
				a *= invSum;
			}
		}
	}
}
