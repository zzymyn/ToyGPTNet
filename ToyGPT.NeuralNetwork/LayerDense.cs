using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CommunityToolkit.HighPerformance;

namespace ToyGPT.NeuralNetwork
{
	public class LayerDense
	{
		private readonly Memory2D<float> m_Weights;
		private readonly Memory<float> m_Bias;

		public int InputCount { get; }
		public int NeuronCount { get; }

		public LayerDense(int numInputs, int numOutputs, Memory2D<float> weights, Memory<float> bias)
		{
			if (weights.Width != numInputs)
				throw new ArgumentException(null, nameof(weights));
			if (weights.Height != numOutputs)
				throw new ArgumentException(null, nameof(weights));
			if (bias.Length != numOutputs)
				throw new ArgumentException(null, nameof(bias));

			InputCount = numInputs;
			NeuronCount = numOutputs;
			m_Weights = weights;
			m_Bias = bias;
		}

		public static LayerDense CreateNewRandom(int numInputs, int numOutputs, Random rng)
		{
			var weights = new float[numOutputs, numInputs];
			var bias = new float[numOutputs];

			foreach (ref var w in weights.AsSpan())
			{
				w = (float)(0.1 * rng.NextNormal());
			}

			return new LayerDense(numInputs, numOutputs, weights, bias);
		}

		public void Forward(ReadOnlySpan2D<float> inputs, Span2D<float> outputs)
		{
			if (inputs.Height != outputs.Height)
				throw new ArgumentException(null, nameof(outputs));
			if (inputs.Width != InputCount)
				throw new ArgumentException(null, nameof(inputs));
			if (outputs.Width != NeuronCount)
				throw new ArgumentException(null, nameof(outputs));

			var weights = m_Weights.Span;
			var bias = m_Bias.Span;

			for (int b = 0; b < inputs.Height; ++b)
			{
				var batchIn = inputs.GetRowSpan(b);
				var batchOut = outputs.GetRowSpan(b);

				for (int i = 0; i < NeuronCount; ++i)
				{
					batchOut[i] = Neuron.Forward(batchIn, weights.GetRowSpan(i), bias[i]);
				}
			}
		}
	}
}
