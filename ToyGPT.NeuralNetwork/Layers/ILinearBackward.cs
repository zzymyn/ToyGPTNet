using CommunityToolkit.HighPerformance;

namespace ToyGPT.NeuralNetwork.Layers;

public interface ILinearBackward
{
	ReadOnlyMemory2D<float> DWeights { get; }
	ReadOnlyMemory<float> DBiases { get; }

	ReadOnlyMemory2D<float> Backward(ReadOnlySpan2D<float> inputs, ReadOnlySpan2D<float> dValues);
}