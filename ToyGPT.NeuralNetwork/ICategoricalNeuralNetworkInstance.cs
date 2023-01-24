using CommunityToolkit.HighPerformance;

namespace ToyGPT.NeuralNetwork
{
	public interface ICategoricalNeuralNetworkInstance
	{
		void Run(ReadOnlySpan2D<float> inputs, Span2D<float> output);
		void Forward(ReadOnlySpan2D<float> inputs, ReadOnlySpan<int> expected, Span2D<float> output, out float avgLoss, out float accuracy);
		void Train(ReadOnlySpan2D<float> inputs, ReadOnlySpan<int> expected, float learningRate, out float avgLoss, out float accuracy);
	}
}