using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CommunityToolkit.HighPerformance;

namespace ToyGPT.NeuralNetwork.Layers;

public class LayerNormalization
{
	private readonly ReadOnlyMemory<float> m_G;
	private readonly ReadOnlyMemory<float> m_B;
	private float[,]? m_Outputs;

	public LayerNormalization(ReadOnlyMemory<float> g, ReadOnlyMemory<float> b)
	{
		m_G = g;
		m_B = b;
	}

	public ReadOnlyMemory2D<float> Outputs => m_Outputs;

	public ReadOnlyMemory2D<float> Forward(ReadOnlyMemory2D<float> inputs)
	{
		ArrayFactory.Resize(ref m_Outputs, inputs.Height, inputs.Width);

		var yMax = inputs.Height;

		for (var y = 0; y < yMax; ++y)
		{
			var rowIn = inputs.Span.GetRowSpan(y);
			var rowOut = m_Outputs.GetRowSpan(y);
			MMath.LayerNormalization(rowIn, m_G.Span, m_B.Span, rowOut);
		}

		return m_Outputs;
	}
}
