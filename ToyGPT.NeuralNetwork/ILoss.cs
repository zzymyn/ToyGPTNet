using CommunityToolkit.HighPerformance;

namespace ToyGPT.NeuralNetwork
{
	public interface ILoss
	{
		void Forward(ReadOnlySpan2D<float> yPreds, ReadOnlySpan2D<float> yTrues, Span<float> losses);
	}
}