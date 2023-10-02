using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CommunityToolkit.HighPerformance;

namespace ToyGPT.NeuralNetwork.Activations;

public sealed class GeLU
{
	private float[,]? m_Outputs;

	public ReadOnlyMemory2D<float> Outputs => m_Outputs;

	public GeLU()
	{
	}

	public ReadOnlyMemory2D<float> Forward(ReadOnlySpan2D<float> inputs)
	{
		ArrayFactory.Resize(ref m_Outputs, inputs.Height, inputs.Width);

		var yMax = inputs.Height;

		for (var y = 0; y < yMax; ++y)
		{
			var rowIn = inputs.GetRowSpan(y);
			var rowOut = m_Outputs.GetRowSpan(y);
			MMath.GeLU(rowIn, rowOut);
		}

		return m_Outputs;
	}
}
