using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CommunityToolkit.HighPerformance;

namespace ToyGPT.NeuralNetwork.Loss
{
	public static class LossCategoricalCrossEntropy
	{
		public static void Forward(ReadOnlySpan2D<float> inputs, ReadOnlySpan<int> categories, Span<float> losses)
		{
			Validate.ArraySize(categories, inputs.Height);
			Validate.ArraySize(losses, inputs.Height);

			var yMax = inputs.Height;
			var xMax = inputs.Width;
			for (var y = 0; y < yMax; ++y)
			{
				var rowIn = inputs.GetRowSpan(y);
				var category = categories[y];

				if (category < 0 || category >= xMax)
					throw new ArgumentException(null, nameof(categories));

				var v = rowIn[category];

				// because log(0) is undefined, we clamp the vales to be greater than 1e-7
				// to avoid this problem, we also clamp the values to be less than 1 - 1e-7
				// to even out the bias towards 1

				v = Math.Clamp(v, 1e-7f, 1.0f - 1e-7f);

				losses[y] = -MathF.Log(v);
			}
		}

		public static void Backward(ReadOnlySpan2D<float> inputs, ReadOnlySpan<int> categories, Span2D<float> dInputs)
		{
			Validate.ArraySize(categories, inputs.Height);
			Validate.ArraysSameSize(inputs, dInputs);

			var yMax = dInputs.Height;
			var xMax = dInputs.Width;
			for (var y = 0; y < yMax; ++y)
			{
				var rowIn = inputs.GetRowSpan(y);
				var rowDIn = dInputs.GetRowSpan(y);
				var category = categories[y];

				if (category < 0 || category > xMax)
					throw new ArgumentException(null, nameof(categories));

				for (var x = 0; x < xMax; ++x)
				{
					rowDIn[x] = 0.0f;
				}

				var dVal = rowIn[category];

				rowDIn[category] = (dVal != 0.0f ? -1.0f / dVal : 0.0f) / yMax;
			}
		}
	}
}
