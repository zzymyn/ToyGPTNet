using CommunityToolkit.HighPerformance;

namespace ToyGPT.NeuralNetwork.Activations
{
	public interface IBackwardActivationInstance
	{
		ReadOnlyMemory2D<float> Backward(ReadOnlySpan2D<float> inputs, ReadOnlySpan2D<float> dValues);
	}
}