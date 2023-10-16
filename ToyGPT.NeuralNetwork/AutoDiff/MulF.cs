using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ToyGPT.NeuralNetwork.AutoDiff
{
	internal sealed class MulF
		: IExpression<float>
	{
		private readonly IExpression<float> m_A;
		private readonly IExpression<float> m_B;

		public MulF(IExpression<float> a, IExpression<float> b)
		{
			m_A = a;
			m_B = b;
		}

		public float Forward(ExpressionContext context)
		{
			var a = context.GetResult(m_A);
			var b = context.GetResult(m_B);
			return a * b;
		}

		public void Backward(ExpressionContext context, float seed)
		{
			var a = context.GetResult(m_A);
			var b = context.GetResult(m_B);

			m_A.Backward(context, seed * b);
			m_B.Backward(context, seed * a);
		}
	}
}
