using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CommunityToolkit.HighPerformance;

namespace ToyGPT.NeuralNetwork;

public static class ArrayFactory
{
	public static T[] NewSameSize<T>(T[] input)
	{
		return new T[input.Length];
	}

	public static T[,] NewSameSize<T>(T[,] input)
	{
		return new T[input.GetLength(0), input.GetLength(1)];
	}

	public static T[,] NewTransposed<T>(T[,] input)
	{
		return new T[input.GetLength(1), input.GetLength(0)];
	}

	public static T[] NewFromWidth<T>(T[,] input)
	{
		return new T[input.GetLength(1)];
	}

	public static T[] NewFromHeight<T>(T[,] input)
	{
		return new T[input.GetLength(0)];
	}

	public static T[] NewFromArea<T>(T[,] input)
	{
		return new T[input.GetLength(0) * input.GetLength(1)];
	}

	public static float[,] NewLayerOutput(float[,] input, float[,] weights)
	{
		return new float[input.GetLength(0), weights.GetLength(0)];
	}

	public static void Resize2dPreservingData<T>([NotNull] ref T[,]? arr, int height, int width)
	{
		var newArr = new T[height, width];

		if (arr != null)
		{
			int minY = Math.Min(arr.GetLength(0), newArr.GetLength(0));
			int minX = Math.Min(arr.GetLength(1), newArr.GetLength(1));

			arr.AsMemory2D()[0..minY, 0..minX].CopyTo(newArr.AsMemory2D()[0..minY, 0..minX]);
		}

		arr = newArr;
	}

	public static void Resize([NotNull] ref float[,]? arr, int height, int width)
	{
		if (arr == null || arr.GetLength(0) != height || arr.GetLength(1) != width)
		{
			arr = new float[height, width];
		}
	}

	public static void Resize([NotNull] ref float[]? arr, int length)
	{
		if (arr == null || arr.Length != length)
		{
			arr = new float[length];
		}
	}
}
