using CommunityToolkit.HighPerformance;

namespace ToyGPT.NeuralNetwork.Layers
{
	public interface ILayerInstance
	{
		ReadOnlyMemory2D<float> Outputs { get; }
		ReadOnlyMemory2D<float> DWeights { get; }
		ReadOnlyMemory<float> DBiases { get; }

		ReadOnlyMemory2D<float> Forward(ReadOnlySpan2D<float> inputs);
		ReadOnlyMemory2D<float> Backward(ReadOnlySpan2D<float> inputs, ReadOnlySpan2D<float> dValues);
	}
}