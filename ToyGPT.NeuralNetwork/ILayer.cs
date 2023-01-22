using CommunityToolkit.HighPerformance;

namespace ToyGPT.NeuralNetwork
{
	public interface ILayer
	{
		void Forward(ReadOnlySpan2D<float> inputs, ReadOnlySpan2D<float> weights, ReadOnlySpan<float> biases, Span2D<float> outputs);
		void Backward(ReadOnlySpan2D<float> inputs, ReadOnlySpan2D<float> weights, ReadOnlySpan2D<float> dValues, Span2D<float> dInputs, Span2D<float> dWeights, Span<float> dBiases);
	}
}