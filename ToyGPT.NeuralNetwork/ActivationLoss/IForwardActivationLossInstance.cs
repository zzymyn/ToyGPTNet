using CommunityToolkit.HighPerformance;

namespace ToyGPT.NeuralNetwork.ActivationLoss
{
	public interface IForwardActivationLossInstance
	{
		ReadOnlyMemory2D<float> ActivationOutput { get; }

		ReadOnlyMemory<float> Forward(ReadOnlySpan2D<float> inputs, ReadOnlySpan<int> expected);
	}
}