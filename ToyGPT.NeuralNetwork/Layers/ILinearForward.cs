using CommunityToolkit.HighPerformance;

namespace ToyGPT.NeuralNetwork.Layers;

public interface ILinearForward
{
	ReadOnlyMemory2D<float> Outputs { get; }

	ReadOnlyMemory2D<float> Forward(ReadOnlyMemory2D<float> inputs);
}