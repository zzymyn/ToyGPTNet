using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CommunityToolkit.HighPerformance;

namespace ToyGPT.NeuralNetwork
{
	public sealed class LossCategoricalCrossEntropy
		: ILossCategorical
	{
		public void Forward(ReadOnlySpan2D<float> inputs, ReadOnlySpan<int> categories, Span<float> losses)
		{
			if (inputs.Height != categories.Length)
				throw new ArgumentException(null, nameof(categories));
			if (inputs.Height != losses.Length)
				throw new ArgumentException(null, nameof(losses));

			var rMax = inputs.Height;
			for (int r = 0; r < rMax; ++r)
			{
				var inRow = inputs.GetRowSpan(r);
				var category = categories[r];

				if (category < 0 || category >= inRow.Length)
					throw new ArgumentException(null, nameof(categories));

				var v = inRow[category];

				// because log(0) is undefined, we clamp the vales to be greater than 1e-7
				// to avoid this problem, we also clamp the values to be less than 1 - 1e-7
				// to even out the bias towards 1
				
				v = Math.Clamp(v, 1e-7f, 1.0f - 1e-7f);

				losses[r] = -MathF.Log(v);
			}
		}
	}
}
