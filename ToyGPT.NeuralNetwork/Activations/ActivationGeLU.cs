using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CommunityToolkit.HighPerformance;

namespace ToyGPT.NeuralNetwork.Activations;

public static class ActivationGeLU
{
	public static void Forward(ReadOnlySpan2D<float> inputs, Span2D<float> outputs)
	{
		Validate.ArraysSameSize(inputs, outputs);

		var yMax = inputs.Height;

		for (var y = 0; y < yMax; ++y)
		{
			var rowIn = inputs.GetRowSpan(y);
			var rowOut = outputs.GetRowSpan(y);
			MMath.GeLU(rowIn, rowOut);
		}
	}
}
