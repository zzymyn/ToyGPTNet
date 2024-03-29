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

	public ReadOnlyMemory2D<float> Forward(ReadOnlyMemory2D<float> qs, ReadOnlyMemory2D<float> ks, ReadOnlyMemory2D<float> vs, int causalOffset = 0)
	{
		m_Outputs = new float[qs.Height, qs.Width];

		var headStep = qs.Width / m_HeadCount;
		var csaScale = 1.0f / MathF.Sqrt(headStep);

		Parallel.For(0, m_HeadCount, head =>
		{
			var tmp = new float[qs.Height, ks.Height];
			var h0 = head * headStep;
			var h1 = h0 + headStep;

			var q = qs[.., h0..h1];
			var k = ks[.., h0..h1];
			var v = vs[.., h0..h1];
			var attn = m_Outputs.AsSpan2D()[.., h0..h1];

			MMath.MulMT(q, k, tmp, useParallel: false);

			for (int y = 0, yMax = qs.Height; y < yMax; ++y)
			{
				var row = tmp.GetRowSpan(y);
				MMath.CausalAttentionAndSoftmax(row, row, causalOffset + y, csaScale);
			}

			MMath.MulMM(tmp, v.Span, attn);
		});

		return m_Outputs;
	}
}