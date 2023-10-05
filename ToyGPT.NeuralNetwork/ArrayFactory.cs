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

	public static Memory2D<T> Resize2DPreservingData<T>([NotNull] ref T[]? buffer, int height, int width)
	{
		if (buffer == null)
		{
			buffer = new T[height * width];
		}
		else
		{
			var newSize = buffer.Length;
			while (newSize < height * width)
			{
				newSize *= 2;
			}

			var newArr = new T[newSize];
			buffer.CopyTo(newArr, 0);

			buffer = newArr;
		}

		return new Memory2D<T>(buffer, height, width);
	}
}
