using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CommunityToolkit.HighPerformance;

namespace ToyGPT.NeuralNetwork.AutoDiff
{
	internal sealed class MatMulMTAddR
		: IExpression<ReadOnlyMemory2D<float>>
	{
		private readonly IExpression<ReadOnlyMemory2D<float>> m_A;
		private readonly IExpression<ReadOnlyMemory2D<float>> m_B;
		private readonly IExpression<ReadOnlyMemory<float>> m_C;

		public MatMulMTAddR(IExpression<ReadOnlyMemory2D<float>> a, IExpression<ReadOnlyMemory2D<float>> b, IExpression<ReadOnlyMemory<float>> c)
		{
			m_A = a;
			m_B = b;
			m_C = c;
		}

		public ReadOnlyMemory2D<float> Forward(ExpressionContext context)
		{
			var a = context.GetResult(m_A);
			var b = context.GetResult(m_B);
			var c = context.GetResult(m_C);
			var result = new float[a.Height, b.Height];
			MMath.MulMTAddR(a, b, c, result);
			return result;
		}

		public void Backward(ExpressionContext context, ReadOnlyMemory2D<float> seed)
		{
			var a = context.GetResult(m_A);
			var b = context.GetResult(m_B);
			var c = context.GetResult(m_C);

			var dA = new float[a.Height, b.Width];
			var dB = new float[b.Height, b.Width];
			var dC = new float[c.Length];

			// calculate dInputs:
			// dInputs = mul(dValues, transpose(weights));
			//         = mul(dValues, weightsT);
			MMath.MulMM(seed.Span, b.Span, dA);

			// calculate dWeightsT:
			// dWeights  = mul(dValues, transpose(inputs))
			// dWeightsT = transpose(mul(dValues, transpose(inputs)))
			//           = mul(transpose(dValues), inputs)
			MMath.MulTM(seed.Span, a.Span, dB);

			// calculate dBiases:
			// dBiases = sum-vertical(dValues)
			MMath.SumColumns(seed.Span, dC);

			m_A.Backward(context, dA);
			m_B.Backward(context, dB);
			m_C.Backward(context, dC);
		}
	}
}
