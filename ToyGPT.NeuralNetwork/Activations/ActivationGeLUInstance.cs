using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CommunityToolkit.HighPerformance;
using ToyGPT.NeuralNetwork.Steps;

namespace ToyGPT.NeuralNetwork.Activations;

public sealed class ActivationGeLUInstance
	: INeuralNetworkForwardStep
{
	private float[,]? m_Outputs;

	public ReadOnlyMemory2D<float> Outputs => m_Outputs;

	public ActivationGeLUInstance()
	{
	}

	public ReadOnlyMemory2D<float> Forward(ReadOnlySpan2D<float> inputs)
	{
		ArrayFactory.ResizeHeight(ref m_Outputs, inputs.Height, inputs.Width);

		ActivationGeLU.Forward(inputs, m_Outputs);

		return m_Outputs;
	}
}
