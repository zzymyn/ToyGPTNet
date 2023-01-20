using System;

namespace ToyGPT.NeuralNetwork
{
	public static class Neuron
	{
		public static float RunNeuron(ReadOnlySpan<float> inputs, ReadOnlySpan<float> weights, float bias)
		{
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