namespace ToyGPT.NeuralNetwork.AutoDiff
{
	internal sealed class AddA
		: IExpression<ReadOnlyMemory<float>>
	{
		private readonly IExpression<ReadOnlyMemory<float>> m_A;
		private readonly IExpression<ReadOnlyMemory<float>> m_B;

		public AddA(IExpression<ReadOnlyMemory<float>> a, IExpression<ReadOnlyMemory<float>> b)
		{
			m_A = a;
			m_B = b;
		}

		public ReadOnlyMemory<float> Forward(ExpressionContext context)
		{
			var a = context.GetResult(m_A);
			var b = context.GetResult(m_B);

			var result = new float[a.Length];
			MMath.Add(a.Span, b.Span, result);
			return result;
		}

		public void Backward(ExpressionContext context, ReadOnlyMemory<float> seed)
		{
			m_A.Backward(context, seed);
			m_B.Backward(context, seed);
		}
	}
}