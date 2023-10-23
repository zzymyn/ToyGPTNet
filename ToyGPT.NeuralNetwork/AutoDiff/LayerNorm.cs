using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ToyGPT.NeuralNetwork.AutoDiff
{
	internal sealed class LayerNormA
		: IExpression<ReadOnlyMemory<float>>
	{
		private readonly IExpression<ReadOnlyMemory<float>> m_A;

		public LayerNormA(IExpression<ReadOnlyMemory<float>> a)
		{
			m_A = a;
		}

		public ReadOnlyMemory<float> Forward(ExpressionContext context)
		{
			var a = context.GetResult(m_A);
			var result = new float[a.Length];
			MMath.LayerNormalization(a.Span, result);
			return result;
		}

		public void Backward(ExpressionContext context, ReadOnlyMemory<float> seed)
		{
			var a = context.GetResult(m_A);
			var dA = new float[a.Length];
			MMath.DLayerNormalization(a.Span, seed.Span, dA);
			m_A.Backward(context, dA);
		}
	}
}
