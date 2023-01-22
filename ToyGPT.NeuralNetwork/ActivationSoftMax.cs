using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CommunityToolkit.HighPerformance;

namespace ToyGPT.NeuralNetwork
{
	public sealed class ActivationSoftMax
		: IActivation
	{
		public void Forward(ReadOnlySpan2D<float> inputs, Span2D<float> outputs)
		{
			if (inputs.Height != outputs.Height)
				throw new ArgumentException(null, nameof(outputs));
			if (inputs.Width != outputs.Width)
				throw new ArgumentException(null, nameof(inputs));

			var rMax = inputs.Height;
			var iMax = inputs.Width;
			for (int r = 0; r < rMax; ++r)
			{
				var rowIn = inputs.GetRowSpan(r);
				var rowOut = outputs.GetRowSpan(r);

				// because overly large values in the input can cause overflow,
				// we need to subtract the max value from all inputs before
				// computing the exponential:

				var max = float.MinValue;
				foreach (var a in rowIn)
				{
					if (a > max)
						max = a;
				}

				// keeping track of the sum, set each output to e^(x - max):

				var sum = 0.0f;

				for (int i = 0; i < iMax; ++i)
				{
					var a = MathF.Exp(rowIn[i] - max);
					sum += a;
					rowOut[i] = a;
				}

				// after that we need to normalize the output using the sum:

				if (sum != 0.0f)
				{
					var invSum = 1.0f / sum;

					foreach (ref var a in rowOut)
					{
						a *= invSum;
					}
				}
			}
		}
	}
}
