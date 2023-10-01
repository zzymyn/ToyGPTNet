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
	public static void Forward(ReadOnlySpan2D<float> inputs, ReadOnlySpan2D<float> weightsT, ReadOnlySpan<float> biases, Span2D<float> outputs)
	{
		Validate.ArraySize(inputs, outputs.Height, weightsT.Width);
		Validate.ArraySize(outputs, inputs.Height, weightsT.Height);
		Validate.ArraySize(weightsT, outputs.Width, inputs.Width);
		Validate.ArraySize(biases, outputs.Width);

		// outputs = mul(inputs, weights) + biases
		//         = mul(inputs, transpose(weightsT)) + biases
		MMath.MulMTAddR(inputs, weightsT, biases, outputs);
	}

	public static void Backward(ReadOnlySpan2D<float> inputs, ReadOnlySpan2D<float> weights, ReadOnlySpan2D<float> dValues, Span2D<float> dInputs, Span2D<float> dWeights, Span<float> dBiases)
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