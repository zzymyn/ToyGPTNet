using CommunityToolkit.HighPerformance;

namespace ToyGPT.NeuralNetwork.Layers;

public interface ILinearBackward
{
	ReadOnlyMemory2D<float> DWeights { get; }
	ReadOnlyMemory<float> DBiases { get; }

	ReadOnlyMemory2D<float> Backward(ReadOnlyMemory2D<float> inputs, ReadOnlyMemory2D<float> dValues);
}