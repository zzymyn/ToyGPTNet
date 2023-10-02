using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CommunityToolkit.HighPerformance;
using ToyGPT.NeuralNetwork.Layers;

namespace ToyGPT.NeuralNetwork.Layers;

public static class LayerCausalSelfAttention
{ 
	public static void Forward(ReadOnlySpan2D<float> q, ReadOnlySpan2D<float> k, ReadOnlySpan2D<float> v, Span2D<float> r, Span<float> scratch = default, float nInf = -1e10f)
	{
		Span2D<float> tmp;

		if (scratch.Length >= q.Height * k.Height)
		{
			tmp = scratch[..(q.Height * k.Height)].AsSpan2D(q.Height, k.Height);
		}
		else if (scratch.IsEmpty)
		{
			tmp = new float[q.Height, k.Height];
		}
		else
		{
			throw new ArgumentException("scratch is too small");
		}

		MMath.MulMT(q, k, tmp);

		{
			var yMax = tmp.Height;

			var scale = 1.0f / MathF.Sqrt(q.Width);

			for (var y = 0; y < yMax; ++y)
			{
				var row = tmp.GetRowSpan(y);
				MMath.CausalAttentionAndSoftmax(row, row, y, scale, nInf: nInf);
			}
		}

		MMath.MulMM(tmp, v, r);
	}
}
