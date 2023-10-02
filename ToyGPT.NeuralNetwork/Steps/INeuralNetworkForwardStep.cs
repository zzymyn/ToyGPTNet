using CommunityToolkit.HighPerformance;

namespace ToyGPT.NeuralNetwork.Steps;

public interface INeuralNetworkForwardStep
{
	ReadOnlyMemory2D<float> Outputs { get; }
	ReadOnlyMemory2D<float> Forward(ReadOnlySpan2D<float> inputs);
}