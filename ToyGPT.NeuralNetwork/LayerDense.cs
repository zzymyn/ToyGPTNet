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
			if (inputs.Height != outputs.Height)
				throw new ArgumentException(null, nameof(outputs));
			if (inputs.Width != InputCount)
				throw new ArgumentException(null, nameof(inputs));
			if (outputs.Width != NeuronCount)
				throw new ArgumentException(null, nameof(outputs));
			if (weights.Width != InputCount)
				throw new ArgumentException(null, nameof(weights));
			if (weights.Height != NeuronCount)
				throw new ArgumentException(null, nameof(weights));
			if (biases.Length != NeuronCount)
				throw new ArgumentException(null, nameof(biases));

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
	}
}
