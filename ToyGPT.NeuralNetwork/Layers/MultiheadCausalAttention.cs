using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CommunityToolkit.HighPerformance;

namespace ToyGPT.NeuralNetwork.Layers;

public sealed class MultiheadCausalAttention
{
	private readonly int m_HeadCount;
	private float[,]? m_Outputs;

	public MultiheadCausalAttention(int headCount)
	{
		m_HeadCount = headCount;
	}

	public ReadOnlyMemory2D<float> Outputs => m_Outputs;

	public ReadOnlyMemory2D<float> Forward(ReadOnlyMemory2D<float> qs, ReadOnlyMemory2D<float> ks, ReadOnlyMemory2D<float> vs)
	{
		ArrayFactory.Resize(ref m_Outputs, qs.Height, qs.Width);

		var headStep = qs.Width / m_HeadCount;
		var h0 = 0;
		var csaScale = 1.0f / MathF.Sqrt(headStep);
		var tmp = new float[qs.Height, qs.Height];

		for (int head = 0; head < m_HeadCount; ++head)
		{
			var h1 = h0 + headStep;

			var q = qs[.., h0..h1];
			var k = ks[.., h0..h1];
			var v = vs[.., h0..h1];
			var attn = m_Outputs.AsSpan2D()[.., h0..h1];

			MMath.MulMT(q, k, tmp);

			for (int y = 0, yMax = qs.Height; y < yMax; ++y)
			{
				var row = tmp.GetRowSpan(y);
				MMath.CausalAttentionAndSoftmax(row, row, y, csaScale);
			}

			MMath.MulMM(tmp, v.Span, attn);

			h0 = h1;
		}

		return m_Outputs;
	}
}