using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CommunityToolkit.HighPerformance;

namespace ToyGPT.NeuralNetwork.Layers;

public class TransformerBlock
{
	private readonly LayerNormalization m_MhaLn;
	private readonly MultiheadCausalSelfAttentionWithKvCache m_Mha;
	private readonly Add m_MhaAdd = new();
	private readonly LayerNormalization m_FfnLn;
	private readonly PositionWiseFeedForward m_Ffn;
	private readonly Add m_FfnAdd = new();

	public TransformerBlock(
		LayerNormalization mhaLn,
		MultiheadCausalSelfAttentionWithKvCache mha,
		LayerNormalization ffnLn,
		PositionWiseFeedForward ffn)
	{
		m_MhaLn = mhaLn;
		m_Mha = mha;
		m_FfnLn = ffnLn;
		m_Ffn = ffn;
	}

	public ReadOnlyMemory2D<float> Forward(ReadOnlyMemory2D<float> inputs)
	{
		var mhaLnOut = m_MhaLn.Forward(inputs);
		var mhaOut = m_Mha.Forward(mhaLnOut);
		var mhaAddOut = m_MhaAdd.Forward(inputs.Span, mhaOut.Span);
		var ffnLnOut = m_FfnLn.Forward(mhaAddOut);
		var ffnOut = m_Ffn.Forward(ffnLnOut);
		return m_FfnAdd.Forward(mhaAddOut.Span, ffnOut.Span);
	}
}
