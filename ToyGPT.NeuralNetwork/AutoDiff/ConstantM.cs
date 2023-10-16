using CommunityToolkit.HighPerformance;

namespace ToyGPT.NeuralNetwork.AutoDiff
{
	public sealed class ConstantM<T>
		: IExpression<ReadOnlyMemory2D<T>>
	{
		public Memory2D<T> Value { get; set; }

		public ConstantM(Memory2D<T> value)
		{
			Value = value;
		}

		public ReadOnlyMemory2D<T> Forward(ExpressionContext context)
		{
			return Value;
		}

		public void Backward(ExpressionContext context, ReadOnlyMemory2D<T> seed)
		{
		}
	}
}
