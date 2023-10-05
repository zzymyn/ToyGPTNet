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
	private ReadOnlyMemory2D<float> m_PrevKVs;

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

	// TODO: need a better way of resetting this on the next prompt?
	public void ClearKvCache()
	{
		m_PrevKVs = default;
	}

	public ReadOnlyMemory2D<float> Forward(ReadOnlyMemory2D<float> inputs)
	{
		var projection = m_Up.Forward(inputs);
		var qvkStep = projection.Width / 3;

		var qs = projection[.., ..qvkStep];
		var kvs = projection[.., qvkStep..];

		var prevCacheRows = m_PrevKVs.Height;
		var newKvs = new float[prevCacheRows + inputs.Height, 2 * qvkStep].AsMemory2D();

		if (prevCacheRows > 0)
		{
			m_PrevKVs.CopyTo(newKvs[..prevCacheRows, ..]);
		}
		kvs.CopyTo(newKvs[prevCacheRows.., ..]);
		m_PrevKVs = newKvs;

		var ks = newKvs[.., ..qvkStep];
		var vs = newKvs[.., qvkStep..];

		var attnOut = m_Attn.Forward(qs, ks, vs, causalOffset: prevCacheRows);

		return m_Down.Forward(attnOut);
	}
}
