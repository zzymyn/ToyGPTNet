using CommunityToolkit.HighPerformance;

namespace ToyGPT.NeuralNetwork.ActivationLoss;

public interface IBackwardActivationLossInstance
{
	ReadOnlyMemory2D<float> Backward(ReadOnlySpan<int> expected);
}