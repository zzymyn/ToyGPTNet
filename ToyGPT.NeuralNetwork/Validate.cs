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
				throw new ArgumentException(null, nameof(b));
		}

		public static void ArraysSameSize<T>(ReadOnlySpan2D<T> a, ReadOnlySpan2D<T> b)
		{
			if (a.Height != b.Height)
				throw new ArgumentException(null, nameof(b));
			if (a.Width != b.Width)
				throw new ArgumentException(null, nameof(a));
		}

		public static void ArraySize<T>(Span2D<T> a, int width, int height)
		{
			if (a.Width != width)
				throw new ArgumentException(null, nameof(a));
			if (a.Height != height)
				throw new ArgumentException(null, nameof(a));
		}

		public static void ArraySize<T>(ReadOnlySpan2D<T> a, int width, int height)
		{
			if (a.Width != width)
				throw new ArgumentException(null, nameof(a));
			if (a.Height != height)
				throw new ArgumentException(null, nameof(a));
		}

		public static void ArraySize<T>(Span<T> a, int length)
		{
			if (a.Length != length)
				throw new ArgumentException(null, nameof(a));
		}

		public static void ArraySize<T>(ReadOnlySpan<T> a, int length)
		{
			if (a.Length != length)
				throw new ArgumentException(null, nameof(a));
		}
	}
}
