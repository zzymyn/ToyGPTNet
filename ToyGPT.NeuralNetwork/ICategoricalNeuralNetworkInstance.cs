using CommunityToolkit.HighPerformance;

namespace ToyGPT.NeuralNetwork;

public interface ICategoricalNeuralNetworkInstance
{
	ICategoricalNeuralNetworkInstance CopyWithBatchSize(int batchSize);
	ReadOnlyMemory2D<float> Run(ReadOnlyMemory2D<float> inputs);
	void Train(ReadOnlyMemory2D<float> inputs, ReadOnlySpan<int> expected, float learningRate, out float avgLoss, out float accuracy);
}