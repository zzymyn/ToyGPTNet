using CommunityToolkit.HighPerformance;

namespace ToyGPT.NeuralNetwork
{
	public interface ILossCategorical
	{
		void Backward(ReadOnlySpan2D<float> inputs, ReadOnlySpan<int> categories, Span2D<float> dInputs);
		void Forward(ReadOnlySpan2D<float> yPreds, ReadOnlySpan<int> yTrues, Span<float> losses);
	}
}