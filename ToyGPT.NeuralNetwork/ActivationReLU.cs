using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CommunityToolkit.HighPerformance;

namespace ToyGPT.NeuralNetwork
{
	public sealed class ActivationReLU
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

				for (int i = 0; i < iMax; ++i)
				{
					var v = rowIn[i];
					rowOut[i] = (v < 0) ? 0 : v;
				}
			}
		}
	}
}
