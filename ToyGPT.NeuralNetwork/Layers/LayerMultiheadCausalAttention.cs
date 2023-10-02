using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CommunityToolkit.HighPerformance;
using ToyGPT.NeuralNetwork.Steps;

namespace ToyGPT.NeuralNetwork.Layers;

public sealed class LayerMultiheadCausalAttention
{
	private readonly int m_HeadCount;
	private float[,]? m_Outputs;

	public LayerMultiheadCausalAttention(int headCount)
	{
		m_HeadCount = headCount;
	}

	public ReadOnlyMemory2D<float> Outputs => m_Outputs;

	public ReadOnlyMemory2D<float> Forward(ReadOnlySpan2D<float> qs, ReadOnlySpan2D<float> ks, ReadOnlySpan2D<float> vs)
	{
		ArrayFactory.ResizeHeight(ref m_Outputs, qs.Height, qs.Width);

		var headStep = qs.Width / m_HeadCount;
		var h0 = 0;
		var csaScale = 1.0f / MathF.Sqrt(headStep);
		var tmp = new float[qs.Height, qs.Height].AsSpan2D();

		for (int head = 0; head < m_HeadCount; ++head)
		{
			var h1 = h0 + headStep;

			var q = qs[.., h0..h1];
			var k = ks[.., h0..h1];
			var v = vs[.., h0..h1];
			var attn = m_Outputs.AsSpan2D()[.., h0..h1];

			MMath.MulMT(q, k, tmp);

			for (int y = 0, yMax = tmp.Height; y < yMax; ++y)
			{
				var row = tmp.GetRowSpan(y);
				MMath.CausalAttentionAndSoftmax(row, row, y, csaScale);
			}

			MMath.MulMM(tmp, v, attn);

			h0 = h1;
		}

		return m_Outputs;
	}
}