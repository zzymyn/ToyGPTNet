using CommunityToolkit.HighPerformance;

namespace ToyGPT.NeuralNetwork
{
	public interface ILossCategorical
	{
		void Forward(ReadOnlySpan2D<float> yPreds, ReadOnlySpan<int> yTrues, Span<float> losses);
	}
}