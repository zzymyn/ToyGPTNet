using CommunityToolkit.HighPerformance;

namespace ToyGPT.NeuralNetwork.Layers;

public interface ILinear
	: ILinearForward
	, ILinearBackward
{
}