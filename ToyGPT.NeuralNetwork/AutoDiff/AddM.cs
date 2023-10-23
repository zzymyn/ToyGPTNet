using CommunityToolkit.HighPerformance;

namespace ToyGPT.NeuralNetwork.AutoDiff
{
	internal sealed class AddM
		: IExpression<ReadOnlyMemory2D<float>>
	{
		private readonly IExpression<ReadOnlyMemory2D<float>> m_A;
		private readonly IExpression<ReadOnlyMemory2D<float>> m_B;

		public AddM(IExpression<ReadOnlyMemory2D<float>> a, IExpression<ReadOnlyMemory2D<float>> b)
		{
			m_A = a;
			m_B = b;
		}

		public ReadOnlyMemory2D<float> Forward(ExpressionContext context)
		{
			var a = context.GetResult(m_A);
			var b = context.GetResult(m_B);

			var result = new float[a.Height, a.Width];
			MMath.Add(a.Span, b.Span, result);
			return result;
		}

		public void Backward(ExpressionContext context, ReadOnlyMemory2D<float> seed)
		{
			m_A.Backward(context, seed);
			m_B.Backward(context, seed);
		}
	}
}
