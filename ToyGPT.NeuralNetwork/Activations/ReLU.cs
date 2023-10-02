using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CommunityToolkit.HighPerformance;

namespace ToyGPT.NeuralNetwork.Activations;

public sealed class ReLU
	: IActivationInstance
{
	private float[,]? m_Outputs;
	private float[,]? m_DInputs;

	public ReadOnlyMemory2D<float> Outputs => m_Outputs;
	public ReadOnlyMemory2D<float> DInputs => m_DInputs;

	public ReLU()
	{
	}

	public ReadOnlyMemory2D<float> Forward(ReadOnlySpan2D<float> inputs)
	{
		ArrayFactory.Resize(ref m_Outputs, inputs.Height, inputs.Width);
		ArrayFactory.Resize(ref m_DInputs, inputs.Height, inputs.Width);

		var yMax = inputs.Height;

		for (var y = 0; y < yMax; ++y)
		{
			var rowIn = inputs.GetRowSpan(y);
			var rowOut = m_Outputs.GetRowSpan(y);
			MMath.ReLU(rowIn, rowOut);
		}

		return m_Outputs;
	}

	public ReadOnlyMemory2D<float> Backward(ReadOnlySpan2D<float> inputs, ReadOnlySpan2D<float> dValues)
	{
		Validate.NotNull(m_Outputs);
		Validate.NotNull(m_DInputs);

		var yMax = inputs.Height;

		for (var y = 0; y < yMax; ++y)
		{
			var rowIn = m_Outputs.GetRowSpan(y);
			var rowDVal = dValues.GetRowSpan(y);
			var rowDIn = m_DInputs.GetRowSpan(y);

			MMath.DReLU(rowIn, rowDVal, rowDIn);
		}

		return m_DInputs;
	}
}
