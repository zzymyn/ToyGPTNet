using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CommunityToolkit.HighPerformance;

namespace ToyGPT.NeuralNetwork.Layers;

public sealed class MultiheadCausalSelfAttention
{
	private readonly LinearWeightsTransposedWithBias m_Up;
	private readonly MultiheadCausalAttention m_Attn;
	private readonly LinearWeightsTransposedWithBias m_Down;

	public MultiheadCausalSelfAttention(
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

	public ReadOnlyMemory2D<float> Forward(ReadOnlySpan2D<float> inputs)
	{
		var projection = m_Up.Forward(inputs);

		var qvkStep = projection.Width / 3;

		var qs = projection[.., 0..qvkStep];
		var ks = projection[.., qvkStep..(qvkStep * 2)];
		var vs = projection[.., (qvkStep * 2)..];

		var attnOut = m_Attn.Forward(qs.Span, ks.Span, vs.Span);

		return m_Down.Forward(attnOut.Span);
	}
}
