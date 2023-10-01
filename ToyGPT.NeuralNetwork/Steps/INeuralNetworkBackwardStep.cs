using CommunityToolkit.HighPerformance;

namespace ToyGPT.NeuralNetwork.Steps
{
	public interface INeuralNetworkBackwardStep
	{
		ReadOnlyMemory2D<float> Backward(ReadOnlySpan2D<float> inputs, ReadOnlySpan2D<float> dValues);
	}
}