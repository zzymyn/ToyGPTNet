using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ToyGPT.NeuralNetwork.AutoDiff
{
	public sealed class ConstantF<T>
		: IExpression<T>
	{
		public T Value { get; set; }

		public ConstantF(T value)
		{
			Value = value;
		}

		public T Forward(ExpressionContext context)
		{
			return Value;
		}

		public void Backward(ExpressionContext context, T seed)
		{
		}
	}
}
