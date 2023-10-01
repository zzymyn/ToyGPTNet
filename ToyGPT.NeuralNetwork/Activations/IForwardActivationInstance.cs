using CommunityToolkit.HighPerformance;

namespace ToyGPT.NeuralNetwork.Activations
{
	public interface IForwardActivationInstance
	{
		ReadOnlyMemory2D<float> Outputs { get; }

		ReadOnlyMemory2D<float> Forward(ReadOnlySpan2D<float> inputs);
	}
}