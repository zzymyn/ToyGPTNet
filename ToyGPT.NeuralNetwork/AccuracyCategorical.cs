using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CommunityToolkit.HighPerformance;

namespace ToyGPT.NeuralNetwork
{
	public class AccuracyCategorical
	{
		public static float Compute(ReadOnlySpan2D<float> yPreds, ReadOnlySpan<int> yTrues)
		{
			if (yPreds.Height != yTrues.Length)
				throw new ArgumentException(null, nameof(yTrues));

			var rMax = yPreds.Height;
			var xMax = yPreds.Width;

			var correct = 0;
			for (int r = 0; r < rMax; ++r)
			{
				var yPredRow = yPreds.GetRowSpan(r);
				var yTrue = yTrues[r];

				var max = float.MinValue;
				var maxIndex = -1;
				for (int x = 0; x < xMax; ++x)
				{
					var yPred = yPredRow[x];
					if (yPred > max)
					{
						max = yPred;
						maxIndex = x;
					}
				}

				if (maxIndex == yTrue)
					++correct;
			}

			return (float)correct / rMax;
		}
	}
}
