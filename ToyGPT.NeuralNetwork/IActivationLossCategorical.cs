using CommunityToolkit.HighPerformance;

namespace ToyGPT.NeuralNetwork
{
	public interface IActivationLossCategorical
	{
		void Forward(ReadOnlySpan2D<float> inputs, ReadOnlySpan<int> categories, Span2D<float> outputs, Span<float> losses);
		void Backward(ReadOnlySpan<int> categories, ReadOnlySpan2D<float> dValues, Span2D<float> dInputs);
	}
}