using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CommunityToolkit.HighPerformance;

namespace ToyGPT.NeuralNetwork.AutoDiff
{
	internal sealed class ReLU
		: IExpression<ReadOnlyMemory2D<float>>
	{
		private readonly IExpression<ReadOnlyMemory2D<float>> m_A;
		private readonly ExpressionContextVariable<float[,]> m_Outputs = new();

		public ReLU(IExpression<ReadOnlyMemory2D<float>> a)
		{
			m_A = a;
		}

		public ReadOnlyMemory2D<float> Forward(ExpressionContext context)
		{
			var inputs = context.GetResult(m_A).Span;

			var outputs = new float[inputs.Height, inputs.Width];

			var yMax = inputs.Height;

			for (var y = 0; y < yMax; ++y)
			{
				var rowIn = inputs.GetRowSpan(y);
				var rowOut = outputs.GetRowSpan(y);
				MMath.ReLU(rowIn, rowOut);
			}

			context.SetValue(m_Outputs, outputs);

			return outputs;
		}

		public void Backward(ExpressionContext context, ReadOnlyMemory2D<float> seed)
		{
			var inputs = context.GetResult(m_A).Span;
			var outputs = context.GetValue(m_Outputs).AsSpan2D();

			var dA = new float[inputs.Height, inputs.Width];

			var yMax = inputs.Height;

			for (var y = 0; y < yMax; ++y)
			{
				var rowIn = outputs.GetRowSpan(y);
				var rowDVal = seed.Span.GetRowSpan(y);
				var rowDIn = dA.GetRowSpan(y);

				MMath.DReLU(rowIn, rowDVal, rowDIn);
			}

			m_A.Backward(context, dA);
		}
	}
}
