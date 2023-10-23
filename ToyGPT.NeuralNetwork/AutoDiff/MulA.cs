using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ToyGPT.NeuralNetwork.AutoDiff
{
	internal sealed class MulA
		: IExpression<ReadOnlyMemory<float>>
	{
		private readonly IExpression<ReadOnlyMemory<float>> m_A;
		private readonly IExpression<ReadOnlyMemory<float>> m_B;

		public MulA(IExpression<ReadOnlyMemory<float>> a, IExpression<ReadOnlyMemory<float>> b)
		{
			m_A = a;
			m_B = b;
		}

		public ReadOnlyMemory<float> Forward(ExpressionContext context)
		{
			var a = context.GetResult(m_A);
			var b = context.GetResult(m_B);
			var result = new float[a.Length];
			MMath.Scale(a.Span, b.Span, result);
			return result;
		}

		public void Backward(ExpressionContext context, ReadOnlyMemory<float> seed)
		{
			var a = context.GetResult(m_A);
			var b = context.GetResult(m_B);

			var seedA = new float[a.Length];
			var seedB = new float[a.Length];

			MMath.Scale(seed.Span, b.Span, seedA);
			MMath.Scale(seed.Span, a.Span, seedB);

			m_A.Backward(context, seedA);
			m_B.Backward(context, seedB);
		}
	}
}
