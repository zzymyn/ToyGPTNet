using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CommunityToolkit.HighPerformance;

namespace ToyGPT.NeuralNetwork.Layers;

public sealed class MultiheadCausalSelfAttentionWithKvCache
{
	private readonly LinearWeightsTransposedWithBias m_Up;
	private readonly MultiheadCausalAttention m_Attn;
	private readonly LinearWeightsTransposedWithBias m_Down;
	private float[,]? m_Ks;
	private float[,]? m_Vs;

	public MultiheadCausalSelfAttentionWithKvCache(
		LinearWeightsTransposedWithBias up,
		MultiheadCausalAttention attn,
		LinearWeightsTransposedWithBias down
		)
	{
		m_Up = up;
		m_Attn = attn;
		m_Down = down;
	}

	public ReadOnlyMemory2D<float> Outputs => m_Down.Outputs;

	public ReadOnlyMemory2D<float> Forward(ReadOnlyMemory2D<float> inputs)
	{
		var projection = m_Up.Forward(inputs);
		var qvkStep = projection.Width / 3;

		var prevCacheRows = m_Ks?.GetLength(0) ?? 0;

		ArrayFactory.Resize2dPreservingData(ref m_Ks, prevCacheRows + inputs.Height, qvkStep);
		ArrayFactory.Resize2dPreservingData(ref m_Vs, prevCacheRows + inputs.Height, qvkStep);

		var qs = projection[.., 0..qvkStep];
		var ks = projection[.., qvkStep..(qvkStep * 2)];
		var vs = projection[.., (qvkStep * 2)..];

		ks.CopyTo(m_Ks.AsMemory2D()[prevCacheRows.., ..]);
		vs.CopyTo(m_Vs.AsMemory2D()[prevCacheRows.., ..]);

		var attnOut = m_Attn.Forward(qs, m_Ks, m_Vs, causalOffset: prevCacheRows);

		return m_Down.Forward(attnOut);
	}
}
