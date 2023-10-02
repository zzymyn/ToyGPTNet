using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CommunityToolkit.HighPerformance;
using ToyGPT.NeuralNetwork.Steps;

namespace ToyGPT.NeuralNetwork.Layers;

public sealed class LayerDenseInstance
	: ILayerInstance
	, INeuralNetworkStep
{
	private readonly int m_BatchSize;
	private readonly int m_InputSize;
	private readonly int m_OutputSize;
	private readonly ReadOnlyMemory2D<float> m_Weights;
	private readonly ReadOnlyMemory<float> m_Biases;
	private readonly float[,] m_Outputs;
	private readonly float[,] m_DInputs;
	private readonly float[,] m_DWeights;
	private readonly float[] m_DBiases;
	
	public ReadOnlyMemory2D<float> Outputs => m_Outputs;
	public ReadOnlyMemory2D<float> DWeights => m_DWeights;
	public ReadOnlyMemory<float> DBiases => m_DBiases;

	public LayerDenseInstance(int batchSize, int inputSize, int outputSize, ReadOnlyMemory2D<float> weights, ReadOnlyMemory<float> biases)
	{
		Validate.ArraySize(weights, outputSize, inputSize);
		Validate.ArraySize(biases, outputSize);

		m_BatchSize = batchSize;
		m_InputSize = inputSize;
		m_OutputSize = outputSize;
		m_Weights = weights;
		m_Biases = biases;
		m_Outputs = new float[batchSize, outputSize];
		m_DInputs = new float[batchSize, inputSize];
		m_DWeights = new float[outputSize, inputSize];
		m_DBiases = new float[outputSize];
	}

	public ReadOnlyMemory2D<float> Forward(ReadOnlySpan2D<float> inputs)
	{
		LayerDense.ForwardMT(inputs, m_Weights.Span, m_Biases.Span, m_Outputs);
		return m_Outputs;
	}

	public ReadOnlyMemory2D<float> Backward(ReadOnlySpan2D<float> inputs, ReadOnlySpan2D<float> dValues)
	{
		LayerDense.BackwardMT(inputs, m_Weights.Span, dValues, m_DInputs, m_DWeights, m_DBiases);
		return m_DInputs;
	}
}
