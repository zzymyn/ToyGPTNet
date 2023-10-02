using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CommunityToolkit.HighPerformance;

namespace ToyGPT.NeuralNetwork.Activations;

public sealed class ActivationSoftMaxInstance
	: IActivationInstance
{
	private float[,]? m_Outputs;
	private float[,]? m_DInputs;

	public ReadOnlyMemory2D<float> Outputs => m_Outputs;
	public ReadOnlyMemory2D<float> DInputs => m_DInputs;

	public ActivationSoftMaxInstance()
	{
	}

	public ReadOnlyMemory2D<float> Forward(ReadOnlySpan2D<float> inputs)
	{
		ArrayFactory.Resize(ref m_Outputs, inputs.Height, inputs.Width);

		MMath.Softmax(inputs, m_Outputs);

		return m_Outputs;
	}

	public ReadOnlyMemory2D<float> Backward(ReadOnlySpan2D<float> inputs, ReadOnlySpan2D<float> dValues)
	{
		Validate.NotNull(m_Outputs);
		ArrayFactory.Resize(ref m_DInputs, inputs.Height, inputs.Width);

		MMath.DSoftmax(m_Outputs, dValues, m_DInputs);

		return m_DInputs;
	}
}
