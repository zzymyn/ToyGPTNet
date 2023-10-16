using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ToyGPT.NeuralNetwork.AutoDiff
{
	public interface IExpression
	{
	}

	public interface IExpression<ResultT>
		: IExpression
	{
		ResultT Forward(ExpressionContext context);
		void Backward(ExpressionContext context, ResultT seed);
	}
}
