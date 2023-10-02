using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CommunityToolkit.HighPerformance;

namespace ToyGPT.NeuralNetwork.Layers;

public static class LayerNormalization
{
	public static void Forward(
		ReadOnlySpan2D<float> inputs,
		ReadOnlySpan<float> g,
		ReadOnlySpan<float> b,
		Span2D<float> outputs)
	{
		Validate.ArraysSameSize(inputs, outputs);
		Validate.ArraySize(g, inputs.Width);
		Validate.ArraySize(b, inputs.Width);

		var yMax = inputs.Height;

		for (var y = 0; y < yMax; ++y)
		{
			var rowIn = inputs.GetRowSpan(y);
			var rowOut = outputs.GetRowSpan(y);
			MMath.LayerNormalization(rowIn, g, b, rowOut);
		}
	}
}
