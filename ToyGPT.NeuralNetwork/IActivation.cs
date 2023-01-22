using CommunityToolkit.HighPerformance;

namespace ToyGPT.NeuralNetwork
{
	public interface IActivation
	{
		void Forward(ReadOnlySpan2D<float> inputs, Span2D<float> outputs);
		void Backward(ReadOnlySpan2D<float> inputs, ReadOnlySpan2D<float> dValues, Span2D<float> dInputs);
	}
}