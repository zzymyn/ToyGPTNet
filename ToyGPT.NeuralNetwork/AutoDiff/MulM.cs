using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CommunityToolkit.HighPerformance;

namespace ToyGPT.NeuralNetwork.AutoDiff
{
	internal sealed class MulM
		: IExpression<ReadOnlyMemory2D<float>>
	{
		private readonly IExpression<ReadOnlyMemory2D<float>> m_A;
		private readonly IExpression<ReadOnlyMemory2D<float>> m_B;

		public MulM(IExpression<ReadOnlyMemory2D<float>> a, IExpression<ReadOnlyMemory2D<float>> b)
		{
			m_A = a;
			m_B = b;
		}

		public ReadOnlyMemory2D<float> Forward(ExpressionContext context)
		{
			var a = context.GetResult(m_A);
			var b = context.GetResult(m_B);
			var result = new float[a.Height, a.Width];
			MMath.Scale(a.Span, b.Span, result);
			return result;
		}

		public void Backward(ExpressionContext context, ReadOnlyMemory2D<float> seed)
		{
			var a = context.GetResult(m_A);
			var b = context.GetResult(m_B);

			var seedA = new float[a.Height, a.Width];
			var seedB = new float[a.Height, a.Width];

			MMath.Scale(seed.Span, b.Span, seedA);
			MMath.Scale(seed.Span, a.Span, seedB);

			m_A.Backward(context, seedA);
			m_B.Backward(context, seedB);
		}
	}
}
