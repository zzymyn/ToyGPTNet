using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CommunityToolkit.HighPerformance;

namespace ToyGPT.NeuralNetwork
{
	public sealed class LossCrossEntropy
		: ILoss
	{
		public void Forward(ReadOnlySpan2D<float> inputs, ReadOnlySpan2D<float> expecteds, Span<float> losses)
		{
			Validate.ArraysSameSize(inputs, expecteds);
			Validate.ArraySize(losses, inputs.Height);

			var rMax = inputs.Height;
			var xMax = inputs.Width;
			for (int r = 0; r < rMax; ++r)
			{
				var inRow = inputs.GetRowSpan(r);
				var exRow = expecteds.GetRowSpan(r);

				var loss = 0.0f;

				for (int x = 0; x < xMax; ++x)
				{
					loss += exRow[x] * MathF.Log(inRow[x]);
				}

				losses[r] = -loss;
			}
		}

		public void Backward(ReadOnlySpan2D<float> inputs, ReadOnlySpan2D<float> expecteds, Span2D<float> dInputs)
		{
			Validate.ArraysSameSize(inputs, dInputs);
			Validate.ArraysSameSize(inputs, expecteds);

			var yMax = dInputs.Height;
			var xMax = dInputs.Width;
			for (int y = 0; y < yMax; ++y)
			{
				var rowIn = inputs.GetRowSpan(y);
				var rowDIn = dInputs.GetRowSpan(y);
				var category = expecteds.GetRowSpan(y);

				for (int x = 0; x < xMax; ++x)
				{
					var dVal = rowIn[x];
					rowDIn[x] =  ((dVal != 0.0f) ? -category[x] / dVal : 0.0f) / yMax;
				}
			}
		}
	}
}
