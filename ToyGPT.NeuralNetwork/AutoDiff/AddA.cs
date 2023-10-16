namespace ToyGPT.NeuralNetwork.AutoDiff
{
	internal sealed class AddA
		: IExpression<ReadOnlyMemory<float>>
	{
		private readonly IExpression<ReadOnlyMemory<float>> m_A;
		private readonly IExpression<ReadOnlyMemory<float>> m_B;
		private float[]? m_Result;

		public ReadOnlyMemory<float> Result => m_Result;

		public AddA(IExpression<ReadOnlyMemory<float>> a, IExpression<ReadOnlyMemory<float>> b)
		{
			m_A = a;
			m_B = b;
		}

		public ReadOnlyMemory<float> Forward(ExpressionContext context)
		{
			var a = context.GetResult(m_A);
			var b = context.GetResult(m_B);

			m_Result = new float[a.Length];
			MMath.Add(a.Span, b.Span, m_Result);
			return m_Result;
		}

		public void Backward(ExpressionContext context, ReadOnlyMemory<float> seed)
		{
			m_A.Backward(context, seed);
			m_B.Backward(context, seed);
		}
	}
}
