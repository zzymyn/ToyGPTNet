using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CommunityToolkit.HighPerformance;

namespace ToyGPT.NeuralNetwork
{
	public sealed class LayerDense
	{
		public int InputCount { get; }
		public int NeuronCount { get; }

		public LayerDense(int numInputs, int numOutputs)
		{
			InputCount = numInputs;
			NeuronCount = numOutputs;
		}

		public void Forward(ReadOnlySpan2D<float> inputs, ReadOnlySpan2D<float> weights, ReadOnlySpan<float> biases, Span2D<float> outputs)
		{
			Validate.ArraySize(inputs, InputCount, outputs.Height);
			Validate.ArraySize(outputs, NeuronCount, inputs.Height);
			Validate.ArraySize(weights, InputCount, NeuronCount);
			Validate.ArraySize(biases, NeuronCount);

			var bMax = inputs.Height;
			for (int b = 0; b < bMax; ++b)
			{
				var batchIn = inputs.GetRowSpan(b);
				var batchOut = outputs.GetRowSpan(b);

				for (int i = 0; i < NeuronCount; ++i)
				{
					batchOut[i] = Neuron.Forward(batchIn, weights.GetRowSpan(i), biases[i]);
				}
			}
		}

		public void Backward(ReadOnlySpan2D<float> inputs, ReadOnlySpan2D<float> weights, ReadOnlySpan2D<float> dValues, Span2D<float> dInputs, Span2D<float> dWeights, Span<float> dBiases)
		{
			Validate.ArraysSameSize(inputs, dInputs);
			Validate.ArraySize(inputs, InputCount, dValues.Height);
			Validate.ArraySize(dValues, NeuronCount, inputs.Height);
			Validate.ArraySize(weights, InputCount, NeuronCount);
			Validate.ArraysSameSize(weights, dWeights);
			Validate.ArraySize(dBiases, NeuronCount);

			// calculate dInputs:
			// dInputs = mul(dValues, weights);
			{
				var yMax = dInputs.Height;
				var xMax = dInputs.Width;
				var iMax = NeuronCount;
				for (int y = 0; y < yMax; ++y)
				{
					var rowDIn = dInputs.GetRowSpan(y);

					for (int x = 0; x < xMax; ++x)
					{
						rowDIn[x] = 0;
						for (int i = 0; i < iMax; ++i)
						{
							rowDIn[x] += dValues[y, i] * weights[i, x];
						}
					}
				}
			}

			// calculate dWeights:
			// dWeights = transpose(mul(dValues, transpose(inputs)))
			//          = mul(transpose(dValues), inputs)
			{
				var yMax = dWeights.Height;
				var xMax = dWeights.Width;
				var iMax = inputs.Height;
				for (int y = 0; y < yMax; ++y)
				{
					var rowDW = dWeights.GetRowSpan(y);

					for (int x = 0; x < xMax; ++x)
					{
						rowDW[x] = 0;
						for (int i = 0; i < iMax; ++i)
						{
							rowDW[x] += dValues[i, y] * inputs[i, x];
						}
					}
				}
			}

			// calculate dBiases:
			// dBiases = sum-vertical(dValues)
			{
				var iMax = dBiases.Length;
				var jMax = inputs.Height;
				for (int i = 0; i < iMax; ++i)
				{
					dBiases[i] = 0;
					for (int j = 0; j < jMax; ++j)
					{
						dBiases[i] += dValues[j, i];
					}
				}
			}
		}
	}
}
