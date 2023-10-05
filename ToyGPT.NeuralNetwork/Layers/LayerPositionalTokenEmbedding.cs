using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CommunityToolkit.HighPerformance;

namespace ToyGPT.NeuralNetwork.Layers;

public class LayerPositionalTokenEmbedding
{
	private readonly ReadOnlyMemory2D<float> m_Wte;
	private readonly ReadOnlyMemory2D<float> m_Wpe;
	private float[,]? m_Output;

	public LayerPositionalTokenEmbedding(ReadOnlyMemory2D<float> wte, ReadOnlyMemory2D<float> wpe)
	{
		m_Wte = wte;
		m_Wpe = wpe;
	}

	public ReadOnlyMemory2D<float> Forward(ReadOnlyMemory<int> input)
	{
		ArrayFactory.Resize(ref m_Output, input.Length, m_Wte.Width);

		var yMax = input.Length;
		var xMax = m_Wte.Width;

		for (var y = 0; y < yMax; ++y)
		{
			var out_y = m_Output.GetRowSpan(y);
			var token = input.Span[y];
			var wte_row = m_Wte.Span.GetRowSpan(token);
			var wpe_row = m_Wpe.Span.GetRowSpan(y);

			for (var x = 0; x < xMax; ++x)
			{
				out_y[x] = wte_row[x] + wpe_row[x];
			}
		}

		return m_Output;
	}
}
