using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CommunityToolkit.HighPerformance;

namespace ToyGPT.NeuralNetwork.Layers;

public static class TransformerBlock
{
	public static void Forward(
		ReadOnlySpan2D<float> inputs,
		ReadOnlySpan2D<float> mhaUpWT,
		ReadOnlySpan<float> mhaUpB,
		ReadOnlySpan2D<float> mhaDownWT,
		ReadOnlySpan<float> mhaDownB,
		ReadOnlySpan<float> mhaLnG,
		ReadOnlySpan<float> mhaLnB,
		ReadOnlySpan2D<float> ffnUpWT,
		ReadOnlySpan<float> ffnUpB,
		ReadOnlySpan2D<float> ffnDownWT,
		ReadOnlySpan<float> ffnDownB,
		ReadOnlySpan<float> ffnLnG,
		ReadOnlySpan<float> ffnLnB,
		int headCount,
		Span2D<float> output)
	{
		var tmp = new float[inputs.Height, inputs.Width].AsSpan2D();
		var mhaOut = new float[inputs.Height, inputs.Width].AsSpan2D();

		LayerNormalization.Forward(inputs, mhaLnG, mhaLnB, tmp);
		LayerMultiheadCausalSelfAttention.Forward(tmp, mhaUpWT, mhaUpB, mhaDownWT, mhaDownB, headCount, mhaOut);
		MMath.Add(inputs, mhaOut, mhaOut);

		LayerNormalization.Forward(mhaOut, ffnLnG, ffnLnB, tmp);
		PositionWiseFeedForward.Forward(tmp, ffnUpWT, ffnUpB, ffnDownWT, ffnDownB, output);
		MMath.Add(mhaOut, output, output);
	}
}
