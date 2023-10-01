using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CommunityToolkit.HighPerformance;

namespace ToyGPT.NeuralNetwork
{
	public static class MMath
	{
		/// <summary>
		/// Matrix multiplication:
		/// <code>r = mul(a, b)</code>
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
		/// Matrix multiplication with transposed b:
		/// <code>r = mul(a, transpose(b))</code>
		/// </summary>
		/// <param name="a"></param>
		/// <param name="b"></param>
		/// <param name="r"></param>
		public static void MulMT(ReadOnlySpan2D<float> a, ReadOnlySpan2D<float> b, Span2D<float> r)
		{
			Validate.True(r != a);
			Validate.True(r != b);
			Validate.True(a.Height == r.Height);
			Validate.True(a.Width == b.Width);
			Validate.True(b.Height == r.Width);

			var yMax = r.Height;
			var xMax = r.Width;
			var iMax = b.Width;

			for (int y = 0; y < yMax; ++y)
			{
				var a_y = a.GetRowSpan(y);
				var r_y = r.GetRowSpan(y);

				for (int x = 0; x < xMax; ++x)
				{
					var b_x = b.GetRowSpan(x);

					var r_y_x = 0.0f; // biases[x];

					for (int i = 0; i < iMax; ++i)
					{
						r_y_x += a_y[i] * b_x[i];
					}

					r_y[x] = r_y_x;
				}
			}
		}

		// r = mul(a, transpose(b)) + c
		public static void MulMTAddR(ReadOnlySpan2D<float> a, ReadOnlySpan2D<float> b, ReadOnlySpan<float> c, Span2D<float> r)
		{
			Validate.True(r != a);
			Validate.True(r != b);
			Validate.True(a.Height == r.Height);
			Validate.True(a.Width == b.Width);
			Validate.True(b.Height == r.Width);
			Validate.True(r.Width == c.Length);

			var yMax = r.Height;
			var xMax = r.Width;
			var iMax = b.Width;

			for (int y = 0; y < yMax; ++y)
			{
				var a_y = a.GetRowSpan(y);
				var r_y = r.GetRowSpan(y);

				for (int x = 0; x < xMax; ++x)
				{
					var b_x = b.GetRowSpan(x);

					var r_y_x = c[x];

					for (int i = 0; i < iMax; ++i)
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

		public static void ReLU(ReadOnlySpan<float> a, Span<float> r)
		{
			Validate.True(a.Length == r.Length);

			var iMax = a.Length;
			for (int i = 0; i < iMax; ++i)
			{
				r[i] = ReLU(a[i]);
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

		public static void LayerNormalization(ReadOnlySpan<float> x, ReadOnlySpan<float> g, ReadOnlySpan<float> b, Span<float> r, float eps = 1e-5f)
		{
			Validate.True(x.Length == r.Length);
			Validate.True(b.Length == r.Length);
			Validate.True(g.Length == r.Length);

			var iMax = x.Length;

			var mean = Mean(x);
			var variance = Variance(x, mean);

			var std = MathF.Sqrt(variance + eps);

			for (int i = 0; i < iMax; ++i)
			{
				r[i] = g[i] * (x[i] - mean) / std + b[i];
			}
		}

		private static float Mean(ReadOnlySpan<float> x)
		{
			var iMax = x.Length;

			var mean = 0.0f;

			for (int i = 0; i < iMax; ++i)
			{
				mean += x[i];
			}

			return mean / iMax;
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

		public static void Softmax(ReadOnlySpan<float> a, Span<float> r)
		{
			Validate.True(a.Length == r.Length);

			// because overly large values in the input can cause overflow,
			// we need to subtract the max value from all inputs before
			// computing the exponential:

			var iMax = a.Length;
			var max = Max(a);

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

		private static float Max(ReadOnlySpan<float> rowIn)
		{
			var max = float.MinValue;

			foreach (var v in rowIn)
			{
				if (v > max)
					max = v;
			}

			return max;
		}

		public static float ReLU(float a)
		{
			return a < 0 ? 0 : a;
		}

		public static float GeLU(float a)
		{
			return a * 0.5f * (1.0f + MathF.Tanh(0.7978845608f * (a + 0.044715f * a * a * a)));
		}
	}
}
