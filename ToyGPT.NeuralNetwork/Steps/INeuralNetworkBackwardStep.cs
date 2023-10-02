using CommunityToolkit.HighPerformance;

namespace ToyGPT.NeuralNetwork.Steps;

public interface INeuralNetworkBackwardStep
{
	public ReadOnlyMemory2D<float> DInputs { get; }

	ReadOnlyMemory2D<float> Backward(ReadOnlySpan2D<float> inputs, ReadOnlySpan2D<float> dValues);
}