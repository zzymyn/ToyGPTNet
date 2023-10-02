using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CommunityToolkit.HighPerformance;
using ToyGPT.NeuralNetwork.Steps;

namespace ToyGPT.NeuralNetwork.Layers;

public sealed class LinearWeightsTransposed
	: INeuralNetworkStep
{
	private readonly ReadOnlyMemory2D<float> m_WeightsT;
	private float[,]? m_Outputs;
	private float[,]? m_DInputs;
	private float[,]? m_DWeightsT;
	
	public ReadOnlyMemory2D<float> Outputs => m_Outputs;
	public ReadOnlyMemory2D<float> DInputs => m_DInputs;
	public ReadOnlyMemory2D<float> DWeights => m_DWeightsT;

	public LinearWeightsTransposed(ReadOnlyMemory2D<float> weightsT)
	{
		m_WeightsT = weightsT;
	}

	public ReadOnlyMemory2D<float> Forward(ReadOnlySpan2D<float> inputs)
	{
		ArrayFactory.ResizeHeight(ref m_Outputs, inputs.Height, m_WeightsT.Height);
		ArrayFactory.ResizeHeight(ref m_DInputs, inputs.Height, m_WeightsT.Width);
		ArrayFactory.ResizeHeight(ref m_DWeightsT, m_WeightsT.Height, m_WeightsT.Width);

		MMath.MulMT(inputs, m_WeightsT.Span, m_Outputs);

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

		return m_DInputs;
	}
}
