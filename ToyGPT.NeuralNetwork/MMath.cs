using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Numerics;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Text;
using System.Threading.Tasks;
using CommunityToolkit.HighPerformance;

namespace ToyGPT.NeuralNetwork;

public static class MMath
{
	private static int VecSize = Vector<float>.Count;

	/// <summary>
	/// Matrix multiplication:
	/// <code>r = mul(a, b)</code>
	/// Slow, try to use MulMT* instead.
	/// </summary>
	/// <param name="a"></param>
	/// <param name="b"></param>
	/// <param name="r"></param>
	public static void MulMM(ReadOnlySpan2D<float> a, ReadOnlySpan2D<float> b, Span2D<float> r)
	{
		Validate.True(r != a);
		Validate.True(r != b);
		Validate.True(a.Height == r.Height);
		Validate.True(a.Width == b.Height);
		Validate.True(b.Width == r.Width);

		r.Clear();

		var yMax = r.Height;
		var xMax = r.Width;
		var iMax = b.Height;

		for (int y = 0; y < yMax; ++y)
		{
			var r_y = r.GetRowSpan(y);
			var a_y = a.GetRowSpan(y);

			for (int i = 0; i < iMax; ++i)
			{
				var a_y_i = a_y[i];
				var b_i = b.GetRowSpan(i);

				for (int x = 0; x < xMax; ++x)
				{
					r_y[x] += a_y_i * b_i[x];
				}
			}
		}
	}

	/// <summary>
	/// Matrix multiplication:
	/// <code>r = mul(a, b) + c</code>
	/// Slow, try to use MulMT* instead.
	/// </summary>
	/// <param name="a"></param>
	/// <param name="b"></param>
	/// <param name="r"></param>
	public static void MulMMAddR(ReadOnlySpan2D<float> a, ReadOnlySpan2D<float> b, ReadOnlySpan<float> c, Span2D<float> r)
	{
		Validate.True(r != a);
		Validate.True(r != b);
		Validate.True(a.Height == r.Height);
		Validate.True(a.Width == b.Height);
		Validate.True(b.Width == r.Width);
		Validate.True(r.Width == c.Length);

		var yMax = r.Height;
		var xMax = r.Width;
		var iMax = b.Height;

		for (int y = 0; y < yMax; ++y)
		{
			var r_y = r.GetRowSpan(y);

			for (int x = 0; x < xMax; ++x)
			{
				r_y[x] = c[x];
			}
		}

		for (int y = 0; y < yMax; ++y)
		{
			var r_y = r.GetRowSpan(y);
			var a_y = a.GetRowSpan(y);

			for (int i = 0; i < iMax; ++i)
			{
				var a_y_i = a_y[i];
				var b_i = b.GetRowSpan(i);

				for (int x = 0; x < xMax; ++x)
				{
					r_y[x] += a_y_i * b_i[x];
				}
			}
		}
	}

	/// <summary>
	/// Matrix multiplication with transposed b:
	/// <code>r = mul(a, transpose(b))</code>
	/// Significantly faster matrix multiplication than MulMM*.
	/// </summary>
	/// <param name="a"></param>
	/// <param name="b"></param>
	/// <param name="r"></param>
	public static void MulMT(ReadOnlyMemory2D<float> a, ReadOnlyMemory2D<float> b, Memory2D<float> r, bool useParallel = true)
	{
		Validate.True(a.Height == r.Height);
		Validate.True(a.Width == b.Width);
		Validate.True(b.Height == r.Width);

		var yMax = r.Height;

		if (useParallel)
		{
			Parallel.ForEach(Partitioner.Create(0, yMax), yRange =>
			{
				MulMT(a, b, r, yRange.ToValueTuple());
			});
		}
		else
		{
			MulMT(a, b, r, (0, yMax));
		}
	}

	private static void MulMT(ReadOnlyMemory2D<float> a, ReadOnlyMemory2D<float> b, Memory2D<float> r, (int, int) yRange)
	{
		var xMax = r.Width;
		var iMax = b.Width;
		var iVMax = iMax - (iMax % VecSize);

		for (var y = yRange.Item1; y < yRange.Item2; ++y)
		{
			var a_y = a.Span.GetRowSpan(y);
			var r_y = r.Span.GetRowSpan(y);

			for (int x = 0; x < xMax; ++x)
			{
				var b_x = b.Span.GetRowSpan(x);

				var accum = Vector<float>.Zero;
				var r_y_x = 0.0f;

				int i = 0;

				for (; i < iVMax; i += VecSize)
				{
					var a_y_vec = new Vector<float>(a_y[i..]);
					var b_x_vec = new Vector<float>(b_x[i..]);

					var v = Vector.Multiply(a_y_vec, b_x_vec);
					accum = Vector.Add(accum, v);
				}

				r_y_x += Vector.Sum(accum);

				for (; i < iMax; ++i)
				{
					r_y_x += a_y[i] * b_x[i];
				}

				r_y[x] = r_y_x;
			}
		}
	}

	/// <summary>
	/// Matrix multiplication with transposed b and adding c:
	/// <code>r = mul(a, transpose(b)) + c</code>
	/// Significantly faster matrix multiplication than MulMM*.
	/// </summary>
	/// <param name="a"></param>
	/// <param name="b"></param>
	/// <param name="c"></param>
	/// <param name="r"></param>
	public static void MulMTAddR(ReadOnlyMemory2D<float> a, ReadOnlyMemory2D<float> b, ReadOnlyMemory<float> c, Memory2D<float> r, bool useParallel = true)
	{
		Validate.True(a.Height == r.Height);
		Validate.True(a.Width == b.Width);
		Validate.True(b.Height == r.Width);
		Validate.True(r.Width == c.Length);

		var yMax = r.Height;

		if (useParallel)
		{
			Parallel.ForEach(Partitioner.Create(0, yMax), yRange =>
			{
				MulMTAddR(a, b, c, r, yRange.ToValueTuple());
			});
		}
		else
		{
			MulMTAddR(a, b, c, r, (0, yMax));
		}
	}

	private static void MulMTAddR(ReadOnlyMemory2D<float> a, ReadOnlyMemory2D<float> b, ReadOnlyMemory<float> c, Memory2D<float> r, (int, int) yRange)
	{
		var xMax = r.Width;
		var iMax = b.Width;
		var iVMax = iMax - (iMax % VecSize);

		for (var y = yRange.Item1; y < yRange.Item2; ++y)
		{
			var a_y = a.Span.GetRowSpan(y);
			var r_y = r.Span.GetRowSpan(y);

			for (int x = 0; x < xMax; ++x)
			{
				var b_x = b.Span.GetRowSpan(x);

				var accum = Vector<float>.Zero;
				var r_y_x = c.Span[x];

				int i = 0;

				for (; i < iVMax; i += VecSize)
				{
					var a_y_vec = new Vector<float>(a_y[i..]);
					var b_x_vec = new Vector<float>(b_x[i..]);
					var v = Vector.Multiply(a_y_vec, b_x_vec);
					accum = Vector.Add(accum, v);
				}

				r_y_x += Vector.Sum(accum);

				for (; i < iMax; ++i)
				{
					r_y_x += a_y[i] * b_x[i];
				}

				r_y[x] = r_y_x;
			}
		}
	}

	// r = a + b
	public static void AddR(ReadOnlySpan2D<float> a, ReadOnlySpan<float> b, Span2D<float> r)
	{
		Validate.True(r != a);
		Validate.True(a.Height == r.Height);
		Validate.True(a.Width == r.Width);
		Validate.True(a.Width == b.Length);

		var yMax = r.Height;
		var xMax = r.Width;

		for (int y = 0; y < yMax; ++y)
		{
			var a_y = a.GetRowSpan(y);
			var r_y = r.GetRowSpan(y);

			for (int x = 0; x < xMax; ++x)
			{
				r_y[x] = a_y[x] + b[x];
			}
		}
	}

	/// <summary>
	/// Matrix multiplication with transposed a:
	/// <code>r = mul(transpose(a), b)</code>
	/// </summary>
	/// <param name="a"></param>
	/// <param name="b"></param>
	/// <param name="r"></param>
	public static void MulTM(ReadOnlySpan2D<float> a, ReadOnlySpan2D<float> b, Span2D<float> r)
	{
		Validate.True(r != a);
		Validate.True(r != b);
		Validate.True(a.Width == r.Height);
		Validate.True(a.Height == b.Height);
		Validate.True(b.Width == r.Width);

		r.Clear();

		var yMax = r.Height;
		var xMax = r.Width;
		var iMax = b.Height;

		for (int y = 0; y < yMax; ++y)
		{
			var r_y = r.GetRowSpan(y);

			for (int i = 0; i < iMax; ++i)
			{
				var a_i_y = a[i, y];
				var b_i = b.GetRowSpan(i);

				for (int x = 0; x < xMax; ++x)
				{
					r_y[x] += a_i_y * b_i[x];
				}
			}
		}
	}

	public static void SumColumns(ReadOnlySpan2D<float> a, Span<float> r)
	{
		Validate.True(a.Width == r.Length);

		r.Clear();

		var xMax = r.Length;
		var yMax = a.Height;

		for (int y = 0; y < yMax; ++y)
		{
			var a_y = a.GetRowSpan(y);

			for (int x = 0; x < xMax; ++x)
			{
				r[x] += a_y[x];
			}
		}
	}

	public static void Add(ReadOnlySpan2D<float> a, ReadOnlySpan2D<float> b, Span2D<float> r)
	{
		Validate.True(a.Height == b.Height);
		Validate.True(a.Width == b.Width);
		Validate.True(a.Height == r.Height);
		Validate.True(a.Width == r.Width);

		var yMax = a.Height;
		var xMax = a.Width;

		for (int y = 0; y < yMax; ++y)
		{
			var a_y = a.GetRowSpan(y);
			var b_y = b.GetRowSpan(y);
			var r_y = r.GetRowSpan(y);

			for (int x = 0; x < xMax; ++x)
			{
				r_y[x] = a_y[x] + b_y[x];
			}
		}
	}

	public static void Add(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> r)
	{
		Validate.True(a.Length == b.Length);
		Validate.True(a.Length == r.Length);

		var iMax = a.Length;
		for (int i = 0; i < iMax; ++i)
		{
			r[i] = a[i] + b[i];
		}
	}

	public static void Scale(ReadOnlySpan2D<float> a, ReadOnlySpan2D<float> b, Span2D<float> r)
	{
		Validate.True(a.Height == b.Height);
		Validate.True(a.Width == b.Width);
		Validate.True(a.Height == r.Height);
		Validate.True(a.Width == r.Width);

		var yMax = a.Height;
		var xMax = a.Width;

		for (int y = 0; y < yMax; ++y)
		{
			var a_y = a.GetRowSpan(y);
			var b_y = b.GetRowSpan(y);
			var r_y = r.GetRowSpan(y);

			for (int x = 0; x < xMax; ++x)
			{
				r_y[x] = a_y[x] * b_y[x];
			}
		}
	}

	public static void Scale(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> r)
	{
		Validate.True(a.Length == b.Length);
		Validate.True(a.Length == r.Length);

		var iMax = a.Length;
		for (int i = 0; i < iMax; ++i)
		{
			r[i] = a[i] * b[i];
		}
	}

	public static void ReLU(ReadOnlySpan<float> a, Span<float> r)
	{
		Validate.True(a.Length == r.Length);

		var iMax = a.Length;
		for (int i = 0; i < iMax; ++i)
		{
			r[i] = ReLU(a[i]);
		}
	}

	public static void DReLU(ReadOnlySpan<float> r, ReadOnlySpan<float> dR, Span<float> dA)
	{
		var xMax = r.Length;

		for (var x = 0; x < xMax; ++x)
		{
			dA[x] = DReLU(r[x], dR[x]);
		}
	}

	public static void GeLU(ReadOnlySpan<float> a, Span<float> r)
	{
		Validate.True(a.Length == r.Length);

		var iMax = a.Length;
		for (int i = 0; i < iMax; ++i)
		{
			r[i] = GeLU(a[i]);
		}
	}

	public static void LayerNormalization(ReadOnlySpan<float> a, Span<float> r, float eps = 1e-5f)
	{
		Validate.True(a.Length == r.Length);

		var iMax = a.Length;

		var mean = Mean(a);
		var variance = Variance(a, mean);

		var invStd = 1.0f / MathF.Sqrt(variance + eps);

		for (int i = 0; i < iMax; ++i)
		{
			r[i] = (a[i] - mean) * invStd;
		}
	}

	public static void LayerNormalization(ReadOnlySpan<float> x, ReadOnlySpan<float> g, ReadOnlySpan<float> b, Span<float> r, float eps = 1e-5f)
	{
		Validate.True(x.Length == r.Length);
		Validate.True(b.Length == r.Length);
		Validate.True(g.Length == r.Length);

		var iMax = x.Length;

		var mean = Mean(x);
		var variance = Variance(x, mean) + eps;

		var invStd = 1.0f / MathF.Sqrt(variance);

		for (int i = 0; i < iMax; ++i)
		{
			r[i] = g[i] * (x[i] - mean) * invStd + b[i];
		}
	}

	public static void DLayerNormalization(ReadOnlySpan<float> a, ReadOnlySpan<float> dR, Span<float> dA, float eps = 1e-5f)
	{
		Validate.True(a.Length == dA.Length);
		Validate.True(a.Length == dR.Length);

		var xMax = a.Length;

		var mean = Mean(a);
		var variance = Variance(a, mean) + eps;

		// dA[A] = Sum of B in 0..N:
		//  when B == A:
		//   (N - 1) / (N * variance) - (x_A - mean) * (x_B - mean) / (N * variance ^ 3)
		//  when B != A:
		//   -1 / (N * variance) - (x_A - mean) * (x_B - mean) / (N * variance ^ 3)

		var invNV = 1.0f / (xMax * variance);
		var invNV3 = 1.0f / (xMax * variance * variance * variance);

		for (int x = 0; x < xMax; ++x)
		{
			var dA_x = 0.0f;

			for (int x2 = 0; x2 < xMax; ++x2)
			{
				float v;

				if (x == x2)
				{
					v = (xMax - 1.0f) * invNV;
				}
				else
				{
					v = -invNV;
				}

				v += (a[x] - mean) * (a[x2] - mean) * invNV3;
				dA_x += dR[x2] * v;
			}

			dA[x] = dA_x;
		}
	}

	private static float Mean(ReadOnlySpan<float> a)
	{
		var iMax = a.Length;

		var mean = 0.0f;

		for (int i = 0; i < iMax; ++i)
		{
			mean += a[i];
		}

		return mean / iMax;
	}

	private static void DMean(ReadOnlySpan<float> a, float dR, Span<float> dA)
	{
		var iMax = a.Length;

		var dMean = dR / iMax;

		for (int i = 0; i < iMax; ++i)
		{
			dA[i] = dMean;
		}
	}

	private static float Variance(ReadOnlySpan<float> x, float mean)
	{
		var iMax = x.Length;

		var variance = 0.0f;

		for (int i = 0; i < iMax; ++i)
		{
			variance += (x[i] - mean) * (x[i] - mean);
		}

		return variance / iMax;
	}

	public static void Softmax(ReadOnlySpan2D<float> a, Span2D<float> r)
	{
		var yMax = a.Height;
		for (var y = 0; y < yMax; ++y)
		{
			var rowIn = a.GetRowSpan(y);
			var rowOut = r.GetRowSpan(y);

			Softmax(rowIn, rowOut);
		}
	}

	public static void DSoftmax(ReadOnlySpan2D<float> r, ReadOnlySpan2D<float> Dr, Span2D<float> Da)
	{
		var yMax = r.Height;
		for (var y = 0; y < yMax; ++y)
		{
			var rowIn = r.GetRowSpan(y);
			var rowDVal = Dr.GetRowSpan(y);
			var rowDIn = Da.GetRowSpan(y);

			DSoftmax(rowIn, rowDVal, rowDIn);
		}
	}

	public static void Softmax(ReadOnlySpan<float> a, Span<float> r)
	{
		Softmax(a, r, Max(a));
	}

	private static void Softmax(ReadOnlySpan<float> a, Span<float> r, float max)
	{
		Validate.True(a.Length == r.Length);

		// because overly large values in the input can cause overflow,
		// we need to subtract the max value from all inputs before
		// computing the exponential:

		var iMax = a.Length;

		// keeping track of the sum, set each output to e^(x - max):

		var sum = 0.0f;

		for (var i = 0; i < iMax; ++i)
		{
			var v = MathF.Exp(a[i] - max);
			sum += v;
			r[i] = v;
		}

		// after that we need to normalize the output using the sum:

		if (sum != 0.0f)
		{
			var invSum = 1.0f / sum;

			for (var i = 0; i < iMax; ++i)
			{
				r[i] *= invSum;
			}
		}
	}

	public static void CausalAttentionAndSoftmax(ReadOnlySpan<float> a, Span<float> r, int row, float scale, float nInf = -1e10f)
	{
		Validate.True(a.Length == r.Length);

		var iMax = a.Length;
		var max = 0.0f;

		for (var i = 0; i < iMax; ++i)
		{
			var v = a[i];

			if (i > row)
			{
				v = nInf;
			}
			else
			{
				v *= scale;
			}

			if (v > max)
				max = v;

			r[i] = v;
		}

		Softmax(r, r, max);
	}

	public static void DSoftmax(ReadOnlySpan<float> r, ReadOnlySpan<float> Dr, Span<float> Da)
	{
		var xMax = r.Length;

		for (var x0 = 0; x0 < xMax; ++x0)
		{
			var Da_x0 = 0.0f;
			var r_x0 = r[x0];

			for (var x1 = 0; x1 < xMax; ++x1)
			{
				if (x0 == x1)
				{
					Da_x0 += Dr[x1] * (r_x0 - r_x0 * r_x0);
				}
				else
				{
					Da_x0 += Dr[x1] * -r_x0 * r[x1];
				}
			}

			Da[x0] = Da_x0;
		}
	}

	public static void CategoricalCrossEntropy(ReadOnlySpan2D<float> inputs, ReadOnlySpan<int> expected, Span<float> losses)
	{
		Validate.True(inputs.Height == expected.Length);
		Validate.True(inputs.Height == losses.Length);

		var yMax = inputs.Height;
		var xMax = inputs.Width;
		for (var y = 0; y < yMax; ++y)
		{
			var rowIn = inputs.GetRowSpan(y);
			var category = expected[y];

			if (category < 0 || category >= xMax)
				throw new ArgumentException(null, nameof(expected));

			var v = rowIn[category];

			// because log(0) is undefined, we clamp the vales to be greater than 1e-7
			// to avoid this problem, we also clamp the values to be less than 1 - 1e-7
			// to even out the bias towards 1

			v = Math.Clamp(v, 1e-7f, 1.0f - 1e-7f);

			losses[y] = -MathF.Log(v);
		}
	}

	public static void DCategoricalCrossEntropy(ReadOnlySpan2D<float> inputs, ReadOnlySpan<int> expected, Span2D<float> dInputs)
	{
		Validate.True(expected.Length == inputs.Height);
		Validate.True(inputs.Height == dInputs.Height);
		Validate.True(inputs.Width == dInputs.Width);

		var yMax = dInputs.Height;
		var xMax = dInputs.Width;
		for (var y = 0; y < yMax; ++y)
		{
			var rowIn = inputs.GetRowSpan(y);
			var rowDIn = dInputs.GetRowSpan(y);
			var category = expected[y];

			if (category < 0 || category > xMax)
				throw new ArgumentException(null, nameof(expected));

			for (var x = 0; x < xMax; ++x)
			{
				rowDIn[x] = 0.0f;
			}

			var dVal = rowIn[category];

			rowDIn[category] = (dVal != 0.0f ? -1.0f / dVal : 0.0f) / yMax;
		}
	}

	public static void DCategoricalCrossEntropy(ReadOnlySpan2D<float> a, ReadOnlySpan<int> e, ReadOnlySpan<float> dR, Span2D<float> dA)
	{
		Validate.True(e.Length == a.Height);
		Validate.True(a.Height == dA.Height);
		Validate.True(a.Width == dA.Width);
		Validate.True(dR.Length == e.Length);

		var yMax = dA.Height;
		var xMax = dA.Width;
		for (var y = 0; y < yMax; ++y)
		{
			var a_y = a.GetRowSpan(y);
			var dA_y = dA.GetRowSpan(y);
			var category = e[y];

			if (category < 0 || category > xMax)
				throw new ArgumentException(null, nameof(e));

			for (var x = 0; x < xMax; ++x)
			{
				dA_y[x] = 0.0f;
			}

			var a_y_category = a_y[category];

			dA_y[category] = dR[y] * (a_y_category != 0.0f ? -1.0f / a_y_category : 0.0f) / yMax;
		}
	}

	public static void DSoftMaxCategoricalCrossEntropy(ReadOnlySpan<int> expected, ReadOnlySpan2D<float> dValues, Span2D<float> dInputs)
	{
		Validate.True(expected.Length == dInputs.Height);
		Validate.True(dInputs.Height == dValues.Height);
		Validate.True(dInputs.Width == dValues.Width);

		var yMax = dInputs.Height;
		var xMax = dInputs.Width;
		for (var y = 0; y < yMax; ++y)
		{
			var rowDVal = dValues.GetRowSpan(y);
			var rowDIn = dInputs.GetRowSpan(y);
			var category = expected[y];

			if (category < 0 || category > xMax)
				throw new ArgumentException(null, nameof(expected));

			for (var x = 0; x < xMax; ++x)
			{
				var a = rowDVal[x];
				if (x == category)
					a -= 1.0f;
				rowDIn[x] = a / yMax;
			}
		}
	}

	public static void DSoftMaxCategoricalCrossEntropy(ReadOnlySpan<int> e, ReadOnlySpan2D<float> b, ReadOnlySpan<float> dR, Span2D<float> dA)
	{
		Validate.True(e.Length == dA.Height);
		Validate.True(dA.Height == b.Height);
		Validate.True(dA.Width == b.Width);
		Validate.True(dR.Length == e.Length);

		var yMax = dA.Height;
		var xMax = dA.Width;
		for (var y = 0; y < yMax; ++y)
		{
			var b_y = b.GetRowSpan(y);
			var dA_y = dA.GetRowSpan(y);
			var category = e[y];

			if (category < 0 || category > xMax)
				throw new ArgumentException(null, nameof(e));

			for (var x = 0; x < xMax; ++x)
			{
				var a = b_y[x];
				if (x == category)
					a -= 1.0f;
				dA_y[x] = dR[y] * a / yMax;
			}
		}
	}

	public static void CrossEntropy(ReadOnlySpan2D<float> inputs, ReadOnlySpan2D<float> expecteds, Span<float> losses)
	{
		Validate.True(inputs.Height == expecteds.Height);
		Validate.True(inputs.Width == expecteds.Width);
		Validate.True(inputs.Height == losses.Length);

		var rMax = inputs.Height;
		var xMax = inputs.Width;
		for (var r = 0; r < rMax; ++r)
		{
			var inRow = inputs.GetRowSpan(r);
			var exRow = expecteds.GetRowSpan(r);

			var loss = 0.0f;

			for (var x = 0; x < xMax; ++x)
			{
				loss += exRow[x] * MathF.Log(inRow[x]);
			}

			losses[r] = -loss;
		}
	}

	public static void DCrossEntropy(ReadOnlySpan2D<float> inputs, ReadOnlySpan2D<float> expecteds, Span2D<float> dInputs)
	{
		Validate.True(inputs.Height == dInputs.Height);
		Validate.True(inputs.Width == dInputs.Width);
		Validate.True(inputs.Height == expecteds.Height);
		Validate.True(inputs.Width == expecteds.Width);

		var yMax = dInputs.Height;
		var xMax = dInputs.Width;
		for (var y = 0; y < yMax; ++y)
		{
			var rowIn = inputs.GetRowSpan(y);
			var rowDIn = dInputs.GetRowSpan(y);
			var category = expecteds.GetRowSpan(y);

			for (var x = 0; x < xMax; ++x)
			{
				var dVal = rowIn[x];
				rowDIn[x] = (dVal != 0.0f ? -category[x] / dVal : 0.0f) / yMax;
			}
		}
	}

	public static float Max(ReadOnlySpan<float> rowIn)
	{
		var max = float.MinValue;

		foreach (var v in rowIn)
		{
			if (v > max)
				max = v;
		}

		return max;
	}

	/// <summary>
	/// Rectified Linear Unit
	/// </summary>
	/// <param name="a"></param>
	/// <returns></returns>
	public static float ReLU(float a)
	{
		return a < 0 ? 0 : a;
	}

	/// <summary>
	/// Derivative of Rectified Linear Unit
	/// </summary>
	/// <param name="r"></param>
	/// <param name="dR"></param>
	/// <returns></returns>
	public static float DReLU(float r, float dR)
	{
		return r <= 0 ? 0 : dR;
	}

	/// <summary>
	/// Gaussian Error Linear Unit
	/// </summary>
	/// <param name="a"></param>
	/// <returns></returns>
	public static float GeLU(float a)
	{
		return a * 0.5f * (1.0f + MathF.Tanh(0.7978845608f * (a + 0.044715f * a * a * a)));
	}
}
