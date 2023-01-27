using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CommunityToolkit.HighPerformance;
using ToyGPT.NeuralNetwork.Activations;

namespace ToyGPT.NeuralNetwork.Loss
{
	public sealed class LossCategoricalCrossEntropyInstance
	{
		private readonly int m_BatchSize;
		private readonly float[] m_Losses;
		private readonly float[,] m_DInputs;

		public LossCategoricalCrossEntropyInstance(int batchSize, int inputSize)
		{
			m_BatchSize = batchSize;
			m_Losses = new float[batchSize];
			m_DInputs = new float[batchSize, inputSize];
		}

		public ReadOnlyMemory<float> Forward(ReadOnlySpan2D<float> inputs, ReadOnlySpan<int> expected)
		{
			LossCategoricalCrossEntropy.Forward(inputs, expected, m_Losses);
			return m_Losses;
		}

		public ReadOnlyMemory2D<float> Backward(ReadOnlySpan2D<float> inputs, ReadOnlySpan<int> expected)
		{
			LossCategoricalCrossEntropy.Backward(inputs, expected, m_DInputs);
			return m_DInputs;
		}

	}
}
