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
			if (inputs.Height != expecteds.Height)
				throw new ArgumentException(null, nameof(expecteds));
			if (inputs.Width != expecteds.Width)
				throw new ArgumentException(null, nameof(expecteds));
			if (inputs.Height != losses.Length)
				throw new ArgumentException(null, nameof(losses));

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
	}
}
