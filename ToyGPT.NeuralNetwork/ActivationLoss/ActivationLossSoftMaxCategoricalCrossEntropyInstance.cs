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
	public sealed class ActivationLossSoftMaxCategoricalCrossEntropyInstance
		: IActivationLossInstance
	{
		private readonly int m_BatchSize;
		private readonly int m_InputSize;
		private readonly float[,] m_SoftMaxOutput;
		private readonly float[] m_Losses;
		private readonly float[,] m_DInputs;

		public ReadOnlyMemory2D<float> ActivationOutput => m_SoftMaxOutput;

		public ActivationLossSoftMaxCategoricalCrossEntropyInstance(int batchSize, int inputSize)
		{
			m_BatchSize = batchSize;
			m_InputSize = inputSize;
			m_SoftMaxOutput = new float[batchSize, inputSize];
			m_Losses = new float[batchSize];
			m_DInputs = new float[batchSize, inputSize];
		}

		public ReadOnlyMemory<float> Forward(ReadOnlySpan2D<float> inputs, ReadOnlySpan<int> expected)
		{
			ActivationLossSoftMaxCategoricalCrossEntropy.Forward(inputs, expected, m_SoftMaxOutput, m_Losses);
			return m_Losses;
		}

		public ReadOnlyMemory2D<float> Backward(ReadOnlySpan<int> expected)
		{
			ActivationLossSoftMaxCategoricalCrossEntropy.Backward(expected, m_SoftMaxOutput, m_DInputs);
			return m_DInputs;
		}
	}
}
