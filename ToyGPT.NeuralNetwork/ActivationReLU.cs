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
			Validate.ArraysSameSize(inputs, outputs);

			var yMax = inputs.Height;
			var xMax = inputs.Width;
			for (int y = 0; y < yMax; ++y)
			{
				var rowIn = inputs.GetRowSpan(y);
				var rowOut = outputs.GetRowSpan(y);

				for (int x = 0; x < xMax; ++x)
				{
					var v = rowIn[x];
					rowOut[x] = (v < 0) ? 0 : v;
				}
			}
		}

		public void Backward(ReadOnlySpan2D<float> inputs, ReadOnlySpan2D<float> dValues, Span2D<float> dInputs)
		{
			Validate.ArraysSameSize(inputs, dValues);
			Validate.ArraysSameSize(inputs, dInputs);

			var yMax = inputs.Height;
			var xMax = inputs.Width;
			for (int y = 0; y < yMax; ++y)
			{
				var rowIn = inputs.GetRowSpan(y);
				var rowDVal = dValues.GetRowSpan(y);
				var rowDIn = dInputs.GetRowSpan(y);

				for (int x = 0; x < xMax; ++x)
				{
					rowDIn[x] = (rowIn[x] <= 0) ? 0 : rowDVal[x];
				}
			}
		}
	}
}
