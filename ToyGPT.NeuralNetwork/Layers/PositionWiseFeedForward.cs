using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CommunityToolkit.HighPerformance;
using ToyGPT.NeuralNetwork.Activations;
using ToyGPT.NeuralNetwork.Steps;

namespace ToyGPT.NeuralNetwork.Layers;

public sealed class PositionWiseFeedForward
	: ChainedForwardStep<
		LinearWeightsTransposedWithBias,
		ActivationGeLUInstance,
		LinearWeightsTransposedWithBias>
{
	public PositionWiseFeedForward(LinearWeightsTransposedWithBias fc, LinearWeightsTransposedWithBias proj)
		: base(fc, new ActivationGeLUInstance(), proj)
	{
	}

	public static void Forward(ReadOnlySpan2D<float> inputs, ReadOnlySpan2D<float> fcWT, ReadOnlySpan<float> fcB, ReadOnlySpan2D<float> projWT, ReadOnlySpan<float> projB, Span2D<float> outputs)
	{
		var tmp = new float[inputs.Height, fcWT.Height].AsSpan2D();

		LayerDense.ForwardMT(inputs, fcWT, fcB, tmp);
		ActivationGeLU.Forward(tmp, tmp);
		LayerDense.ForwardMT(tmp, projWT, projB, outputs);
	}
}
