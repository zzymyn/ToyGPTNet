using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CommunityToolkit.HighPerformance;

namespace ToyGPT.NeuralNetwork.AutoDiff
{
	internal sealed class MatMulMT
		: IExpression<ReadOnlyMemory2D<float>>
	{
		private readonly IExpression<ReadOnlyMemory2D<float>> m_A;
		private readonly IExpression<ReadOnlyMemory2D<float>> m_B;

		public MatMulMT(IExpression<ReadOnlyMemory2D<float>> a, IExpression<ReadOnlyMemory2D<float>> b)
		{
			m_A = a;
			m_B = b;
		}

		public ReadOnlyMemory2D<float> Forward(ExpressionContext context)
		{
			var a = context.GetResult(m_A);
			var b = context.GetResult(m_B);
			var result = new float[a.Height, b.Height];
			MMath.MulMT(a, b, result);
			return result;
		}

		public void Backward(ExpressionContext context, ReadOnlyMemory2D<float> seed)
		{
			var a = context.GetResult(m_A);
			var b = context.GetResult(m_B);

			var dA = new float[a.Height, b.Width];
			var dB = new float[b.Height, b.Width];

			// calculate dInputs:
			// dInputs = mul(dValues, transpose(weights));
			//         = mul(dValues, weightsT);
			MMath.MulMM(seed.Span, b.Span, dA);

			// calculate dWeightsT:
			// dWeights  = mul(dValues, transpose(inputs))
			// dWeightsT = transpose(mul(dValues, transpose(inputs)))
			//           = mul(transpose(dValues), inputs)
			MMath.MulTM(seed.Span, a.Span, dB);

			m_A.Backward(context, dA);
			m_B.Backward(context, dB);
		}
	}
}
