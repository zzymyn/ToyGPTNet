using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ToyGPT.NeuralNetwork.AutoDiff
{
	public sealed class ExpressionContext
	{
		private readonly Dictionary<IExpression, object> m_Results = new();
		private readonly Dictionary<ExpressionContextVariable, object> m_SavedValues = new();

		public void Clear()
		{
			m_Results.Clear();
			m_SavedValues.Clear();
		}

		public T GetResult<T>(IExpression<T> expression)
			where T : notnull
		{
			if (!m_Results.TryGetValue(expression, out var result))
			{
				var resultT = expression.Forward(this);
				m_Results[expression] = resultT;
				return resultT;
			}

			return (T)result;
		}

		public void SetValue<T>(ExpressionContextVariable var, T value)
			where T : notnull
		{
			m_SavedValues[var] = value;
		}

		public bool HasValue(ExpressionContextVariable var)
		{
			return m_SavedValues.ContainsKey(var);
		}

		public T GetValue<T>(ExpressionContextVariable<T> var)
			where T : notnull
		{
			if (!m_SavedValues.TryGetValue(var, out var result))
			{
				throw new InvalidOperationException();
			}

			return (T)result;
		}

		public bool TryGetValue<T>(ExpressionContextVariable<T> var, out T value)
			where T : notnull
		{
			if (!m_SavedValues.TryGetValue(var, out var result))
			{
				value = default!;
				return false;
			}

			value = (T)result;
			return true;
		}
	}

	public abstract class ExpressionContextVariable
	{
	}

	public sealed class ExpressionContextVariable<T>
		: ExpressionContextVariable
		where T : notnull
	{
	}
}
