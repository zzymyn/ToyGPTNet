using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CommunityToolkit.HighPerformance;

namespace ToyGPT.NeuralNetwork
{
	internal static class Validate
	{
		public static void ArraysSameSize<T>(ReadOnlySpan<T> a, ReadOnlySpan<T> b)
		{
			if (a.Length != b.Length)
				throw new ArgumentException($"Arrays aren't the same size: {a.Length} != {b.Length}.");
		}

		public static void ArraysSameSize<T>(ReadOnlySpan2D<T> a, ReadOnlySpan2D<T> b)
		{
			if (a.Height != b.Height || a.Width != b.Width)
			{
				throw new ArgumentException($"Arrays aren't the same size: {a.Height}x{a.Width} != {b.Height}x{b.Width}.");
			}
		}

		public static void ArraySize<T>(Span2D<T> a, int height, int width)
		{
			if (a.Width != width || a.Height != height)
			{
				throw new ArgumentException($"Array is the wrong size: {a.Height}x{a.Width} != {height}x{width}.");
			}
		}

		public static void ArraySize<T>(ReadOnlySpan2D<T> a, int height, int width)
		{
			if (a.Width != width || a.Height != height)
			{
				throw new ArgumentException($"Array is the wrong size: {a.Height}x{a.Width} != {height}x{width}.");
			}
		}

		public static void ArraySize<T>(Span<T> a, int length)
		{
			if (a.Length != length)
			{
				throw new ArgumentException($"Array is the wrong size: {a.Length} != {length}.");
			}
		}

		public static void ArraySize<T>(ReadOnlySpan<T> a, int length)
		{
			if (a.Length != length)
			{
				throw new ArgumentException($"Array is the wrong size: {a.Length} != {length}.");
			}
		}
	}
}
