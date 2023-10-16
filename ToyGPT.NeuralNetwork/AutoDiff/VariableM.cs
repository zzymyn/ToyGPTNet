using CommunityToolkit.HighPerformance;

namespace ToyGPT.NeuralNetwork.AutoDiff
{
	public sealed class VariableM
		: IExpression<ReadOnlyMemory2D<float>>
	{
		private readonly ExpressionContextVariable<float[,]> m_Gradient = new();

		public Memory2D<float> Value { get; set; }

		public VariableM(Memory2D<float> value)
		{
			Value = value;
		}

		public ReadOnlyMemory2D<float> GetGradient(ExpressionContext context)
		{
			return context.GetValue(m_Gradient);
		}

		public ReadOnlyMemory2D<float> Forward(ExpressionContext context)
		{
			return Value;
		}

		public void Backward(ExpressionContext context, ReadOnlyMemory2D<float> seed)
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
