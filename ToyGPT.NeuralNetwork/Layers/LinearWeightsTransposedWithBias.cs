using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CommunityToolkit.HighPerformance;

namespace ToyGPT.NeuralNetwork.Layers;

public sealed class LinearWeightsTransposedWithBias
	: ILinear
{
	private readonly ReadOnlyMemory2D<float> m_WeightsT;
	private readonly ReadOnlyMemory<float> m_Biases;
	private float[,]? m_Outputs;
	private float[,]? m_DInputs;
	private float[,]? m_DWeightsT;
	private float[]? m_DBiases;
	
	public ReadOnlyMemory2D<float> Outputs => m_Outputs;
	public ReadOnlyMemory2D<float> DInputs => m_DInputs;
	public ReadOnlyMemory2D<float> DWeights => m_DWeightsT;
	public ReadOnlyMemory<float> DBiases => m_DBiases;

	public LinearWeightsTransposedWithBias(ReadOnlyMemory2D<float> weightsT, ReadOnlyMemory<float> biases)
	{
		Validate.True(weightsT.Height == biases.Length);

		m_WeightsT = weightsT;
		m_Biases = biases;
	}

	public ReadOnlyMemory2D<float> Forward(ReadOnlyMemory2D<float> inputs)
	{
		m_Outputs = new float[inputs.Height, m_WeightsT.Height];

		MMath.MulMTAddR(inputs, m_WeightsT, m_Biases, m_Outputs);

		return m_Outputs;
	}

	public ReadOnlyMemory2D<float> Backward(ReadOnlyMemory2D<float> inputs, ReadOnlyMemory2D<float> dValues)
	{
		m_DInputs = new float[inputs.Height, m_WeightsT.Width];
		m_DWeightsT = new float[m_WeightsT.Height, m_WeightsT.Width];
		m_DBiases = new float[m_Biases.Length];

		// calculate dInputs:
		// dInputs = mul(dValues, transpose(weights));
		//         = mul(dValues, weightsT);
		MMath.MulMM(dValues.Span, m_WeightsT.Span, m_DInputs);

		// calculate dWeightsT:
		// dWeights  = mul(dValues, transpose(inputs))
		// dWeightsT = transpose(mul(dValues, transpose(inputs)))
		//           = mul(transpose(dValues), inputs)
		MMath.MulTM(dValues.Span, inputs.Span, m_DWeightsT);

		// calculate dBiases:
		// dBiases = sum-vertical(dValues)
		MMath.SumColumns(dValues.Span, m_DBiases);

		return m_DInputs;
	}
}
