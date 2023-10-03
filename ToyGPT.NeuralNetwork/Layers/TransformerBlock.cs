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
	private readonly MultiheadCausalSelfAttention m_Mha;
	private readonly Add m_MhaAdd = new();
	private readonly LayerNormalization m_FfnLn;
	private readonly PositionWiseFeedForward m_Ffn;
	private readonly Add m_FfnAdd = new();
	private float[,]? m_Outputs;

	public TransformerBlock(
		LayerNormalization mhaLn,
		MultiheadCausalSelfAttention mha,
		LayerNormalization ffnLn,
		PositionWiseFeedForward ffn)
	{
		m_MhaLn = mhaLn;
		m_Mha = mha;
		m_FfnLn = ffnLn;
		m_Ffn = ffn;
	}

	public ReadOnlyMemory2D<float> Forward(ReadOnlySpan2D<float> inputs)
	{
		ArrayFactory.Resize(ref m_Outputs, inputs.Height, inputs.Width);

		var mhaLnOut = m_MhaLn.Forward(inputs);
		var mhaOut = m_Mha.Forward(mhaLnOut);
		var mhaAddOut = m_MhaAdd.Forward(inputs, mhaOut.Span);
		var ffnLnOut = m_FfnLn.Forward(mhaAddOut.Span);
		var ffnOut = m_Ffn.Forward(ffnLnOut);
		return m_FfnAdd.Forward(mhaAddOut.Span, ffnOut.Span);
	}
}
