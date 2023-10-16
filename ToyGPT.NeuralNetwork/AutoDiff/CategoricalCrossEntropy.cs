using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CommunityToolkit.HighPerformance;

namespace ToyGPT.NeuralNetwork.AutoDiff
{
	internal sealed class CategoricalCrossEntropy
		: IExpression<ReadOnlyMemory<float>>
	{
		private readonly IExpression<ReadOnlyMemory2D<float>> m_A;
		private readonly IExpression<ReadOnlyMemory<int>> m_E;
		private readonly ExpressionContextVariable<float[]> m_Outputs = new();

		public CategoricalCrossEntropy(IExpression<ReadOnlyMemory2D<float>> a, IExpression<ReadOnlyMemory<int>> e)
		{
			m_A = a;
			m_E = e;
		}

		public ReadOnlyMemory<float> Forward(ExpressionContext context)
		{
			var inputs = context.GetResult(m_A);
			var expected = context.GetResult(m_E);
			var outputs = new float[inputs.Height];

			MMath.CategoricalCrossEntropy(inputs.Span, expected.Span, outputs);

			context.SetValue(m_Outputs, outputs);

			return outputs;
		}

		public void Backward(ExpressionContext context, ReadOnlyMemory<float> seed)
		{
			if (m_A is Softmax softmax)
			{
				var inputs = context.GetResult(m_A);
				var expected = context.GetResult(m_E);
				var dA = new float[inputs.Height, inputs.Width];
				MMath.DSoftMaxCategoricalCrossEntropy(expected.Span, inputs.Span, seed.Span, dA);
				softmax.BackwardsFromSoftMaxCategoricalCrossEntropy(context, dA);
			}
			else
			{
				var inputs = context.GetResult(m_A);
				var expected = context.GetResult(m_E);
				var dA = new float[inputs.Height, inputs.Width];

				MMath.DCategoricalCrossEntropy(inputs.Span, expected.Span, seed.Span, dA);

				m_A.Backward(context, dA);
			}
		}
	}
}
