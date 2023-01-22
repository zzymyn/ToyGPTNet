using CommunityToolkit.HighPerformance;

namespace ToyGPT.NeuralNetwork
{
	public interface IActivation
	{
		void Forward(ReadOnlySpan2D<float> inputs, Span2D<float> outputs);
	}
}