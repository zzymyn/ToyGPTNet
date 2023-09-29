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
	}
}
