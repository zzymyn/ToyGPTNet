using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CommunityToolkit.HighPerformance;
using ToyGPT.NeuralNetwork.Activations;

namespace ToyGPT.NeuralNetwork.Layers;

public static class PositionWiseFeedForward
{
	public static void Forward(ReadOnlySpan2D<float> inputs, ReadOnlySpan2D<float> fcWT, ReadOnlySpan<float> fcB, ReadOnlySpan2D<float> projWT, ReadOnlySpan<float> projB, Span2D<float> outputs)
	{
		var tmp = new float[inputs.Height, fcWT.Height].AsSpan2D();

		LayerDense.ForwardMT(inputs, fcWT, fcB, tmp);
		ActivationGeLU.Forward(tmp, tmp);
		LayerDense.ForwardMT(tmp, projWT, projB, outputs);
	}
}
