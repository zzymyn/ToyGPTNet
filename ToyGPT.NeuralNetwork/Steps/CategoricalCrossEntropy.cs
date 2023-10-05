using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CommunityToolkit.HighPerformance;
using ToyGPT.NeuralNetwork.Activations;

namespace ToyGPT.NeuralNetwork.Steps;

public sealed class CategoricalCrossEntropy
{
	private float[]? m_Losses;
	private float[,]? m_DInputs;

	public CategoricalCrossEntropy()
	{
	}

	public ReadOnlyMemory<float> Forward(ReadOnlySpan2D<float> inputs, ReadOnlySpan<int> expected)
	{
		m_Losses = new float[inputs.Height];

		MMath.CategoricalCrossEntropy(inputs, expected, m_Losses);

		return m_Losses;
	}

	public ReadOnlyMemory2D<float> Backward(ReadOnlySpan2D<float> inputs, ReadOnlySpan<int> expected)
	{
		m_DInputs = new float[inputs.Height, inputs.Width];

		MMath.DCategoricalCrossEntropy(inputs, expected, m_DInputs);

		return m_DInputs;
	}

}
