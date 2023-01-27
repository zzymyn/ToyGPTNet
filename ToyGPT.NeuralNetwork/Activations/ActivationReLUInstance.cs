using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CommunityToolkit.HighPerformance;

namespace ToyGPT.NeuralNetwork.Activations
{
	public sealed class ActivationReLUInstance
		: IActivationInstance
	{
		private readonly int m_BatchSize;
		private readonly float[,] m_Outputs;
		private readonly float[,] m_DInputs;

		public ReadOnlyMemory2D<float> Outputs => m_Outputs;

		public ActivationReLUInstance(int batchSize, int inputSize)
		{
			m_BatchSize = batchSize;
			m_Outputs = new float[batchSize, inputSize];
			m_DInputs = new float[batchSize, inputSize];
		}

		public ReadOnlyMemory2D<float> Forward(ReadOnlySpan2D<float> inputs)
		{
			ActivationReLU.Forward(inputs, m_Outputs);
			return m_Outputs;
		}

		public ReadOnlyMemory2D<float> Backward(ReadOnlySpan2D<float> inputs, ReadOnlySpan2D<float> dValues)
		{
			ActivationReLU.Backward(m_Outputs, dValues, m_DInputs);
			return m_DInputs;
		}
	}
}
