using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ToyGPT.NeuralNetwork.AutoDiff
{
	internal sealed class AddF
		: IExpression<float>
	{
		private readonly IExpression<float> m_A;
		private readonly IExpression<float> m_B;

		public AddF(IExpression<float> a, IExpression<float> b)
		{
			m_A = a;
			m_B = b;
		}

		public float Forward(ExpressionContext context)
		{
			return context.GetResult(m_A) + context.GetResult(m_B);
		}

		public void Backward(ExpressionContext context, float seed)
		{
			m_A.Backward(context, seed);
			m_B.Backward(context, seed);
		}
	}
}
