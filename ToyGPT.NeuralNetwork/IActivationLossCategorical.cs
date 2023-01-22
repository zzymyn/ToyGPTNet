using CommunityToolkit.HighPerformance;

namespace ToyGPT.NeuralNetwork
{
	public interface IActivationLossCategorical
	{
		void Forward(ReadOnlySpan2D<float> inputs, ReadOnlySpan<int> categories, Span2D<float> outputs, Span<float> losses);
	}
}