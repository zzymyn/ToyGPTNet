using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CommunityToolkit.HighPerformance;
using ToyGPT.NeuralNetwork.Layers;

namespace ToyGPT.NeuralNetwork.Layers
{
	public static class LayerDense
	{
		public static void Forward(ReadOnlySpan2D<float> inputs, ReadOnlySpan2D<float> weights, ReadOnlySpan<float> biases, Span2D<float> outputs)
		{
			Validate.ArraySize(inputs, outputs.Height, weights.Width);
			Validate.ArraySize(outputs, inputs.Height, weights.Height);
			Validate.ArraySize(weights, outputs.Width, inputs.Width);
			Validate.ArraySize(biases, outputs.Width);

			// outputs = mul(inputs, transpose(weights)) + biases

			var yMax = outputs.Height;
			var xMax = outputs.Width;
			var iMax = weights.Width;
			for (int y = 0; y < yMax; ++y)
			{
				var rowIn = inputs.GetRowSpan(y);
				var rowOut = outputs.GetRowSpan(y);

				for (int x = 0; x < xMax; ++x)
				{
					var rowW = weights.GetRowSpan(x);

					var v = biases[x];

					for (int i = 0; i < iMax; ++i)
					{
						v += rowIn[i] * rowW[i];
					}

					rowOut[x] = v;
				}
			}
		}

		public static void Backward(ReadOnlySpan2D<float> inputs, ReadOnlySpan2D<float> weights, ReadOnlySpan2D<float> dValues, Span2D<float> dInputs, Span2D<float> dWeights, Span<float> dBiases)
		{
			Validate.ArraysSameSize(inputs, dInputs);
			Validate.ArraySize(inputs, dValues.Height, weights.Width);
			Validate.ArraySize(dValues, inputs.Height, weights.Height);
			Validate.ArraySize(weights, dValues.Width, inputs.Width);
			Validate.ArraysSameSize(weights, dWeights);
			Validate.ArraySize(dBiases, weights.Height);

			// calculate dInputs:
			// dInputs = mul(dValues, weights);
			{
				dInputs.Clear();

				var yMax = dInputs.Height;
				var xMax = dInputs.Width;
				var iMax = weights.Height;
				for (int y = 0; y < yMax; ++y)
				{
					var rowDIn = dInputs.GetRowSpan(y);
					var rowDVal = dValues.GetRowSpan(y);

					for (int i = 0; i < iMax; ++i)
					{
						var dValue = rowDVal[i];
						var rowWeights = weights.GetRowSpan(i);

						for (int x = 0; x < xMax; ++x)
						{
							rowDIn[x] += dValue * rowWeights[x];
						}
					}
				}
			}

			// calculate dWeights:
			// dWeights = transpose(mul(dValues, transpose(inputs)))
			//          = mul(transpose(dValues), inputs)
			{
				dWeights.Clear();

				var yMax = dWeights.Height;
				var xMax = dWeights.Width;
				var iMax = inputs.Height;
				for (int y = 0; y < yMax; ++y)
				{
					var rowDW = dWeights.GetRowSpan(y);

					for (int i = 0; i < iMax; ++i)
					{
						var dValue = dValues[i, y];
						var rowIn = inputs.GetRowSpan(i);

						for (int x = 0; x < xMax; ++x)
						{
							rowDW[x] += dValue * rowIn[x];
						}
					}
				}
			}

			// calculate dBiases:
			// dBiases = sum-vertical(dValues)
			{
				dBiases.Clear();

				var iMax = dBiases.Length;
				var jMax = inputs.Height;
				for (int j = 0; j < jMax; ++j)
				{
					var rowDVal = dValues.GetRowSpan(j);

					for (int i = 0; i < iMax; ++i)
					{
						dBiases[i] += rowDVal[i];
					}
				}
			}
		}
	}
}
