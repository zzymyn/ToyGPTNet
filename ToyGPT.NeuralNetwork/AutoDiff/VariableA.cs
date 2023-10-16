namespace ToyGPT.NeuralNetwork.AutoDiff
{
	public sealed class VariableA
		: IExpression<ReadOnlyMemory<float>>
	{
		private readonly ExpressionContextVariable<float[]> m_Gradient = new();

		public Memory<float> Value { get; set; }

		public VariableA(Memory<float> value)
		{
			Value = value;
		}

		public ReadOnlyMemory<float> GetGradient(ExpressionContext context)
		{
			return context.GetValue(m_Gradient);
		}

		public ReadOnlyMemory<float> Forward(ExpressionContext context)
		{
			return Value;
		}

		public void Backward(ExpressionContext context, ReadOnlyMemory<float> seed)
		{
			if (context.TryGetValue(m_Gradient, out var gradient))
			{
				MMath.Add(gradient, seed.Span, gradient);
			}
			else
			{
				context.SetValue(m_Gradient, seed.ToArray());
			}
		}
	}
}
