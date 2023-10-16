namespace ToyGPT.NeuralNetwork.AutoDiff
{
	public sealed class ConstantA<T>
		: IExpression<ReadOnlyMemory<T>>
	{
		public Memory<T> Value { get; set; }

		public ConstantA(Memory<T> value)
		{
			Value = value;
		}

		public ReadOnlyMemory<T> Forward(ExpressionContext context)
		{
			return Value;
		}

		public void Backward(ExpressionContext context, ReadOnlyMemory<T> seed)
		{
		}
	}
}
