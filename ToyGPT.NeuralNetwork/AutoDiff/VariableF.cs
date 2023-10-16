using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Threading.Tasks;

namespace ToyGPT.NeuralNetwork.AutoDiff
{
	public sealed class VariableF
		: IExpression<float>
	{
		private readonly ExpressionContextVariable<float> m_Gradient = new();

		public float Value { get; set; }

		public VariableF(float value)
		{
			Value = value;
		}

		public float GetGradient(ExpressionContext context)
		{
			return context.GetValue(m_Gradient);
		}

		public float Forward(ExpressionContext context)
		{
			return Value;
		}

		public void Backward(ExpressionContext context, float seed)
		{
			if (context.TryGetValue(m_Gradient, out var gradient))
			{
				gradient += seed;
				context.SetValue(m_Gradient, gradient);
			}
			else
			{
				context.SetValue(m_Gradient, seed);
			}
		}
	}
}
