using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CommunityToolkit.HighPerformance;

namespace ToyGPT.NeuralNetwork.Activations;

public static class ActivationReLU
{
	public static void Forward(ReadOnlySpan2D<float> inputs, Span2D<float> outputs)
	{
		Validate.ArraysSameSize(inputs, outputs);

		var yMax = inputs.Height;

		for (var y = 0; y < yMax; ++y)
		{
			var rowIn = inputs.GetRowSpan(y);
			var rowOut = outputs.GetRowSpan(y);
			MMath.ReLU(rowIn, rowOut);
		}
	}

	public static void Backward(ReadOnlySpan2D<float> outputs, ReadOnlySpan2D<float> dValues, Span2D<float> dInputs)
	{
		Validate.ArraysSameSize(outputs, dValues);
		Validate.ArraysSameSize(outputs, dInputs);

		var yMax = outputs.Height;

		for (var y = 0; y < yMax; ++y)
		{
			var rowIn = outputs.GetRowSpan(y);
			var rowDVal = dValues.GetRowSpan(y);
			var rowDIn = dInputs.GetRowSpan(y);

			MMath.DReLU(rowIn, rowDVal, rowDIn);
		}
	}
}
