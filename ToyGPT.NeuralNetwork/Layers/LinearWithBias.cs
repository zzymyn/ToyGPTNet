using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CommunityToolkit.HighPerformance;
using ToyGPT.NeuralNetwork.Steps;

namespace ToyGPT.NeuralNetwork.Layers;

public sealed class LinearWithBias
	: INeuralNetworkStep
{
	private readonly ReadOnlyMemory2D<float> m_Weights;
	private readonly ReadOnlyMemory<float> m_Biases;
	private float[,]? m_Outputs;
	private float[,]? m_DInputs;
	private float[,]? m_DWeights;
	private float[]? m_DBiases;

	public ReadOnlyMemory2D<float> Outputs => m_Outputs;
	public ReadOnlyMemory2D<float> DInputs => m_DInputs;

	public LinearWithBias(ReadOnlyMemory2D<float> weights, ReadOnlyMemory<float> biases)
	{
		Validate.True(weights.Width == biases.Length);

		m_Weights = weights;
		m_Biases = biases;
	}

	public ReadOnlyMemory2D<float> Forward(ReadOnlySpan2D<float> inputs)
	{
		ArrayFactory.ResizeHeight(ref m_Outputs, inputs.Height, m_Weights.Width);
		ArrayFactory.ResizeHeight(ref m_DInputs, inputs.Height, m_Weights.Height);
		ArrayFactory.ResizeHeight(ref m_DWeights, m_Weights.Height, m_Weights.Width);
		ArrayFactory.Resize(ref m_DBiases, m_Biases.Length);

		MMath.MulMMAddR(inputs, m_Weights.Span, m_Biases.Span, m_Outputs);

		return m_Outputs;
	}

	public ReadOnlyMemory2D<float> Backward(ReadOnlySpan2D<float> inputs, ReadOnlySpan2D<float> dValues)
	{
		// calculate dInputs:
		// dInputs = mul(dValues, transpose(weights));
		//         = mul(dValues, weightsT);
		MMath.MulMT(dValues, m_Weights.Span, m_DInputs);

		// calculate dWeightsT:
		// dWeights  = mul(dValues, transpose(inputs))
		// dWeightsT = transpose(mul(dValues, transpose(inputs)))
		//           = mul(transpose(dValues), inputs)
		MMath.MulMT(dValues, inputs, m_DWeights);

		// calculate dBiases:
		// dBiases = sum-vertical(dValues)
		MMath.SumColumns(dValues, m_DBiases);

		return m_DInputs;
	}
}
