using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using CommunityToolkit.HighPerformance;

namespace ToyGPT.NeuralNetwork;

internal static class Validate
{
	[Conditional("DEBUG")]
	public static void True(bool condition, [CallerArgumentExpression(nameof(condition))] string? message = null)
	{
		if (!condition)
		{
			if (message is null)
			{
				throw new ArgumentException("Condition is false.");
			}
			else
			{
				throw new ArgumentException($"Expected {message}, but wasn't.");
			}
		}
	}

	[Conditional("DEBUG")]
	public static void NotNull([NotNull] object? a, [CallerArgumentExpression(nameof(a))] string? message = null)
	{
		if (a is null)
		{
			if (message is null)
			{
				throw new ArgumentNullException();
			}
			else
			{
				throw new ArgumentNullException(message);
			}
		}
	}

	[Conditional("DEBUG")]
	public static void ArraysSameSize<T>(ReadOnlySpan<T> a, ReadOnlySpan<T> b)
	{
		if (a.Length != b.Length)
			throw new ArgumentException($"Arrays aren't the same size: {a.Length} != {b.Length}.");
	}

	[Conditional("DEBUG")]
	public static void ArraysSameSize<T>(ReadOnlySpan2D<T> a, ReadOnlySpan2D<T> b)
	{
		if (a.Height != b.Height || a.Width != b.Width)
		{
			throw new ArgumentException($"Arrays aren't the same size: {a.Height}x{a.Width} != {b.Height}x{b.Width}.");
		}
	}

	[Conditional("DEBUG")]
	public static void ArraySize<T>(Span2D<T> a, int height, int width)
	{
		if (a.Width != width || a.Height != height)
		{
			throw new ArgumentException($"Array is the wrong size: {a.Height}x{a.Width} != {height}x{width}.");
		}
	}

	[Conditional("DEBUG")]
	public static void ArraySize<T>(ReadOnlySpan2D<T> a, int height, int width)
	{
		if (a.Width != width || a.Height != height)
		{
			throw new ArgumentException($"Array is the wrong size: {a.Height}x{a.Width} != {height}x{width}.");
		}
	}

	[Conditional("DEBUG")]
	public static void ArraySize<T>(Span<T> a, int length)
	{
		if (a.Length != length)
		{
			throw new ArgumentException($"Array is the wrong size: {a.Length} != {length}.");
		}
	}

	[Conditional("DEBUG")]
	public static void ArraySize<T>(ReadOnlySpan<T> a, int length)
	{
		if (a.Length != length)
		{
			throw new ArgumentException($"Array is the wrong size: {a.Length} != {length}.");
		}
	}

	[Conditional("DEBUG")]
	public static void ArraysSameSize<T>(ReadOnlyMemory<T> a, ReadOnlyMemory<T> b)
	{
		if (a.Length != b.Length)
			throw new ArgumentException($"Arrays aren't the same size: {a.Length} != {b.Length}.");
	}

	[Conditional("DEBUG")]
	public static void ArraysSameSize<T>(ReadOnlyMemory2D<T> a, ReadOnlyMemory2D<T> b)
	{
		if (a.Height != b.Height || a.Width != b.Width)
		{
			throw new ArgumentException($"Arrays aren't the same size: {a.Height}x{a.Width} != {b.Height}x{b.Width}.");
		}
	}

	[Conditional("DEBUG")]
	public static void ArraySize<T>(Memory2D<T> a, int height, int width)
	{
		if (a.Width != width || a.Height != height)
		{
			throw new ArgumentException($"Array is the wrong size: {a.Height}x{a.Width} != {height}x{width}.");
		}
	}

	[Conditional("DEBUG")]
	public static void ArraySize<T>(ReadOnlyMemory2D<T> a, int height, int width)
	{
		if (a.Width != width || a.Height != height)
		{
			throw new ArgumentException($"Array is the wrong size: {a.Height}x{a.Width} != {height}x{width}.");
		}
	}

	[Conditional("DEBUG")]
	public static void ArraySize<T>(Memory<T> a, int length)
	{
		if (a.Length != length)
		{
			throw new ArgumentException($"Array is the wrong size: {a.Length} != {length}.");
		}
	}

	[Conditional("DEBUG")]
	public static void ArraySize<T>(ReadOnlyMemory<T> a, int length)
	{
		if (a.Length != length)
		{
			throw new ArgumentException($"Array is the wrong size: {a.Length} != {length}.");
		}
	}

}
