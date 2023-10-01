using CommunityToolkit.HighPerformance;

namespace ToyGPT.NeuralNetwork.Layers;

public interface ILayerInstance
	: IForwardLayerInstance
	, IBackwardLayerInstance
{
}