using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CommunityToolkit.HighPerformance;

namespace ToyGPT.NeuralNetwork.AutoDiff
{
	internal sealed class Softmax
		: IExpression<ReadOnlyMemory2D<float>>
	{
		private readonly IExpression<ReadOnlyMemory2D<float>> m_A;
		private readonly ExpressionContextVariable<float[,]> m_Outputs = new();

		public Softmax(IExpression<ReadOnlyMemory2D<float>> a)
		{
			m_A = a;
		}

		public ReadOnlyMemory2D<float> Forward(ExpressionContext context)
		{
			var inputs = context.GetResult(m_A);
			var outputs = new float[inputs.Height, inputs.Width];

			MMath.Softmax(inputs.Span, outputs);

			context.SetValue(m_Outputs, outputs);

			return outputs;
		}

		public void Backward(ExpressionContext context, ReadOnlyMemory2D<float> seed)
		{
			var inputs = context.GetResult(m_A);
			var outputs = context.GetValue(m_Outputs);
			var dA = new float[inputs.Height, inputs.Width];

			MMath.DSoftmax(outputs, seed.Span, dA);

			m_A.Backward(context, dA);
		}

		internal void BackwardsFromSoftMaxCategoricalCrossEntropy(ExpressionContext context, ReadOnlyMemory2D<float> dA)
		{
			m_A.Backward(context, dA);
		}
	}
}
