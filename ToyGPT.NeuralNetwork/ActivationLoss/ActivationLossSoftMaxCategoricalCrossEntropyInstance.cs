using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CommunityToolkit.HighPerformance;
using ToyGPT.NeuralNetwork.Activations;

namespace ToyGPT.NeuralNetwork.ActivationLoss;

public sealed class ActivationLossSoftMaxCategoricalCrossEntropyInstance
	: IActivationLossInstance
{
	private float[,]? m_SoftMaxOutput;
	private float[]? m_Losses;
	private float[,]? m_DInputs;

	public ReadOnlyMemory2D<float> ActivationOutput => m_SoftMaxOutput;

	public ActivationLossSoftMaxCategoricalCrossEntropyInstance()
	{
		//m_SoftMaxOutput = new float[batchSize, inputSize];
		//m_Losses = new float[batchSize];
		//m_DInputs = new float[batchSize, inputSize];
	}

	public ReadOnlyMemory<float> Forward(ReadOnlySpan2D<float> inputs, ReadOnlySpan<int> expected)
	{
		m_SoftMaxOutput = new float[inputs.Height, inputs.Width];
		m_Losses = new float[inputs.Height];
		m_DInputs = new float[inputs.Height, inputs.Width];

		MMath.Softmax(inputs, m_SoftMaxOutput);
		MMath.CategoricalCrossEntropy(m_SoftMaxOutput, expected, m_Losses);

		return m_Losses;
	}

	public ReadOnlyMemory2D<float> Backward(ReadOnlySpan<int> expected)
	{
		MMath.DSoftMaxCategoricalCrossEntropy(expected, m_SoftMaxOutput, m_DInputs);

		return m_DInputs;
	}
}
