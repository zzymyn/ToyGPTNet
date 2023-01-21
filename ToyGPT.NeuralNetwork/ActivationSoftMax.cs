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
			var rMax = inputs.Height;
			for (int r = 0; r < rMax; ++r)
			{
				var row = inputs.GetRowSpan(r);

				var max = float.MinValue;
				foreach (var a in row)
				{
					if (a > max)
						max = a;
				}

				var sum = 0.0f;

				foreach (ref var a in row)
				{
					a = (float)Math.Exp(a - max);
					sum += a;
				}

				if (sum != 0.0f)
				{
					var invSum = 1.0f / sum;

					foreach (ref var a in row)
					{
						a *= invSum;
					}
				}
			}
		}
	}
}
