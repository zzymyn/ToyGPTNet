using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CommunityToolkit.HighPerformance;
using ToyGPT.NeuralNetwork.Activations;
using ToyGPT.NeuralNetwork.Loss;

namespace ToyGPT.NeuralNetwork.ActivationLoss
{
	public static class ActivationLossSoftMaxCategoricalCrossEntropy
	{
		public static void Forward(ReadOnlySpan2D<float> inputs, ReadOnlySpan<int> categories, Span2D<float> outputs, Span<float> losses)
		{
			ActivationSoftMax.Forward(inputs, outputs);
			LossCategoricalCrossEntropy.Forward(outputs, categories, losses);
		}

		public static void Backward(ReadOnlySpan<int> categories, ReadOnlySpan2D<float> dValues, Span2D<float> dInputs)
		{
			Validate.ArraysSameSize(dInputs, dValues);
			Validate.ArraySize(categories, dInputs.Height);

			var yMax = dInputs.Height;
			var xMax = dInputs.Width;
			for (var y = 0; y < yMax; ++y)
			{
				var rowDVal = dValues.GetRowSpan(y);
				var rowDIn = dInputs.GetRowSpan(y);
				var category = categories[y];

				if (category < 0 || category > xMax)
					throw new ArgumentException(null, nameof(categories));

				for (var x = 0; x < xMax; ++x)
				{
					var a = rowDVal[x];
					if (x == category)
						a -= 1.0f;
					rowDIn[x] = a / yMax;
				}
			}
		}
	}
}
