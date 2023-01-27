using System;
using CommunityToolkit.HighPerformance;

namespace ToyGPT.NeuralNetwork.ActivationLoss
{
	public interface IActivationLossInstance
	{
		ReadOnlyMemory2D<float> ActivationOutput { get; }

		ReadOnlyMemory<float> Forward(ReadOnlySpan2D<float> inputs, ReadOnlySpan<int> expected);
		ReadOnlyMemory2D<float> Backward(ReadOnlySpan<int> expected);
	}
}