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

		public void Backward(ReadOnlySpan2D<float> inputs, ReadOnlySpan2D<float> dValues, Span2D<float> dInputs)
		{
			Validate.ArraysSameSize(inputs, dValues);
			Validate.ArraysSameSize(inputs, dInputs);

			var rMax = inputs.Height;
			var iMax = inputs.Width;
			for (int r = 0; r < rMax; ++r)
			{
				var rowIn = inputs.GetRowSpan(r);
				var rowDVal = dValues.GetRowSpan(r);
				var rowDIn = dInputs.GetRowSpan(r);

				for (int i = 0; i < iMax; ++i)
				{
					rowDIn[i] = (rowIn[i] <= 0) ? 0 : rowDVal[i];
				}
			}
		}
	}
}
