using System;

namespace ToyGPT.NeuralNetwork
{
	public static class Neuron
	{
		public static float Forward(ReadOnlySpan<float> inputs, ReadOnlySpan<float> weights, float bias)
		{
			if (inputs.Length != weights.Length)
				throw new ArgumentException(null, nameof(weights));
			
			var sum = 0.0f;

			var len = inputs.Length;
			for (int i = 0; i < len; ++i)
			{
				sum += inputs[i] * weights[i];
			}

			return sum + bias;
		}
	}
}