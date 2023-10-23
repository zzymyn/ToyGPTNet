using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Threading.Tasks;
using CommunityToolkit.HighPerformance;

namespace ToyGPT.NeuralNetwork.AutoDiff
{
	public static class ExpressionBuilder
	{
		public static VariableF V(float c)
		{
			return new VariableF(c);
		}

		public static VariableA V(Memory<float> c)
		{
			return new VariableA(c);
		}

		public static VariableM V(Memory2D<float> c)
		{
			return new VariableM(c);
		}

		public static ConstantF<T> C<T>(T c)
		{
			return new ConstantF<T>(c);
		}

		public static ConstantA<T> C<T>(T[] c)
		{
			return new ConstantA<T>(c);
		}

		public static ConstantA<T> C<T>(Memory<T> c)
		{
			return new ConstantA<T>(c);
		}

		public static ConstantM<T> C<T>(Memory2D<T> c)
		{
			return new ConstantM<T>(c);
		}

		public static IExpression<float> Add(IExpression<float> a, IExpression<float> b)
		{
			return new AddF(a, b);
		}

		public static IExpression<ReadOnlyMemory<float>> Add(IExpression<ReadOnlyMemory<float>> a, IExpression<ReadOnlyMemory<float>> b)
		{
			return new AddA(a, b);
		}

		public static IExpression<ReadOnlyMemory2D<float>> Add(IExpression<ReadOnlyMemory2D<float>> a, IExpression<ReadOnlyMemory2D<float>> b)
		{
			return new AddM(a, b);
		}

		public static IExpression<float> Mul(IExpression<float> a, IExpression<float> b)
		{
			return new MulF(a, b);
		}

		public static IExpression<ReadOnlyMemory<float>> Mul(IExpression<ReadOnlyMemory<float>> a, IExpression<ReadOnlyMemory<float>> b)
		{
			return new MulA(a, b);
		}

		public static IExpression<ReadOnlyMemory2D<float>> Mul(IExpression<ReadOnlyMemory2D<float>> a, IExpression<ReadOnlyMemory2D<float>> b)
		{
			return new MulM(a, b);
		}

		public static IExpression<ReadOnlyMemory2D<float>> MatMulMT(IExpression<ReadOnlyMemory2D<float>> a, IExpression<ReadOnlyMemory2D<float>> b)
		{
			return new MatMulMT(a, b);
		}

		public static IExpression<ReadOnlyMemory2D<float>> MatMulMTAddR(IExpression<ReadOnlyMemory2D<float>> a, IExpression<ReadOnlyMemory2D<float>> b, IExpression<ReadOnlyMemory<float>> c)
		{
			return new MatMulMTAddR(a, b, c);
		}

		public static IExpression<ReadOnlyMemory2D<float>> ReLU(IExpression<ReadOnlyMemory2D<float>> a)
		{
			return new ReLU(a);
		}

		public static IExpression<ReadOnlyMemory2D<float>> Softmax(IExpression<ReadOnlyMemory2D<float>> a)
		{
			return new Softmax(a);
		}

		public static IExpression<ReadOnlyMemory<float>> CategoricalCrossEntropy(IExpression<ReadOnlyMemory2D<float>> a, IExpression<ReadOnlyMemory<int>> e)
		{
			return new CategoricalCrossEntropy(a, e);
		}

		public static IExpression<ReadOnlyMemory<float>> LayerNorm(IExpression<ReadOnlyMemory<float>> a)
		{
			return new LayerNormA(a);
		}
	}
}
