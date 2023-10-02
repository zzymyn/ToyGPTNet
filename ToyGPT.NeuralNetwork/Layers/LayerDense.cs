using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CommunityToolkit.HighPerformance;
using ToyGPT.NeuralNetwork.Layers;

namespace ToyGPT.NeuralNetwork.Layers;

public static class LayerDense
{
	public static void ForwardMT(ReadOnlySpan2D<float> inputs, ReadOnlySpan2D<float> weightsT, Span2D<float> outputs)
	{
		Validate.ArraySize(inputs, outputs.Height, weightsT.Width);
		Validate.ArraySize(outputs, inputs.Height, weightsT.Height);
		Validate.ArraySize(weightsT, outputs.Width, inputs.Width);

		// outputs = mul(inputs, weights)
		//         = mul(inputs, transpose(weightsT))
		MMath.MulMT(inputs, weightsT, outputs);
	}

	public static void ForwardMM(ReadOnlySpan2D<float> inputs, ReadOnlySpan2D<float> weights, Span2D<float> outputs)
	{
		Validate.ArraySize(inputs, outputs.Height, weights.Height);
		Validate.ArraySize(outputs, inputs.Height, weights.Width);
		Validate.ArraySize(weights, inputs.Width, outputs.Width);

		// outputs = mul(inputs, weights)
		MMath.MulMM(inputs, weights, outputs);
	}

	public static void ForwardMT(ReadOnlySpan2D<float> inputs, ReadOnlySpan2D<float> weightsT, ReadOnlySpan<float> biases, Span2D<float> outputs)
	{
		Validate.ArraySize(inputs, outputs.Height, weightsT.Width);
		Validate.ArraySize(outputs, inputs.Height, weightsT.Height);
		Validate.ArraySize(weightsT, outputs.Width, inputs.Width);
		Validate.ArraySize(biases, outputs.Width);

		// outputs = mul(inputs, weights) + biases
		//         = mul(inputs, transpose(weightsT)) + biases
		MMath.MulMTAddR(inputs, weightsT, biases, outputs);
	}

	public static void BackwardMT(ReadOnlySpan2D<float> inputs, ReadOnlySpan2D<float> weights, ReadOnlySpan2D<float> dValues, Span2D<float> dInputs, Span2D<float> dWeights)
	{
		Validate.ArraysSameSize(inputs, dInputs);
		Validate.ArraySize(inputs, dValues.Height, weights.Width);
		Validate.ArraySize(dValues, inputs.Height, weights.Height);
		Validate.ArraySize(weights, dValues.Width, inputs.Width);
		Validate.ArraysSameSize(weights, dWeights);

		// calculate dInputs:
		// dInputs = mul(dValues, weights);
		MMath.MulMM(dValues, weights, dInputs);

		// calculate dWeights:
		// dWeights = transpose(mul(dValues, transpose(inputs)))
		//          = mul(transpose(dValues), inputs)
		MMath.MulTM(dValues, inputs, dWeights);
	}

	public static void BackwardMT(ReadOnlySpan2D<float> inputs, ReadOnlySpan2D<float> weights, ReadOnlySpan2D<float> dValues, Span2D<float> dInputs, Span2D<float> dWeights, Span<float> dBiases)
	{
		Validate.ArraysSameSize(inputs, dInputs);
		Validate.ArraySize(inputs, dValues.Height, weights.Width);
		Validate.ArraySize(dValues, inputs.Height, weights.Height);
		Validate.ArraySize(weights, dValues.Width, inputs.Width);
		Validate.ArraysSameSize(weights, dWeights);
		Validate.ArraySize(dBiases, weights.Height);

		// calculate dInputs:
		// dInputs = mul(dValues, weights);
		MMath.MulMM(dValues, weights, dInputs);

		// calculate dWeights:
		// dWeights = transpose(mul(dValues, transpose(inputs)))
		//          = mul(transpose(dValues), inputs)
		MMath.MulTM(dValues, inputs, dWeights);

		// calculate dBiases:
		// dBiases = sum-vertical(dValues)
		MMath.SumColumns(dValues, dBiases);
	}
}
