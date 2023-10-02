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

	public ReadOnlyMemory2D<float> Forward(ReadOnlySpan2D<float> inputs)
	{
		ArrayFactory.Resize(ref m_Outputs, inputs.Height, m_WeightsT.Height);
		ArrayFactory.Resize(ref m_DInputs, inputs.Height, m_WeightsT.Width);
		ArrayFactory.Resize(ref m_DWeightsT, m_WeightsT.Height, m_WeightsT.Width);
		ArrayFactory.Resize(ref m_DBiases, m_Biases.Length);

		MMath.MulMTAddR(inputs, m_WeightsT.Span, m_Biases.Span, m_Outputs);

		return m_Outputs;
	}

	public ReadOnlyMemory2D<float> Backward(ReadOnlySpan2D<float> inputs, ReadOnlySpan2D<float> dValues)
	{
		// calculate dInputs:
		// dInputs = mul(dValues, transpose(weights));
		//         = mul(dValues, weightsT);
		MMath.MulMM(dValues, m_WeightsT.Span, m_DInputs);

		// calculate dWeightsT:
		// dWeights  = mul(dValues, transpose(inputs))
		// dWeightsT = transpose(mul(dValues, transpose(inputs)))
		//           = mul(transpose(dValues), inputs)
		MMath.MulTM(dValues, inputs, m_DWeightsT);

		// calculate dBiases:
		// dBiases = sum-vertical(dValues)
		MMath.SumColumns(dValues, m_DBiases);

		return m_DInputs;
	}
}
