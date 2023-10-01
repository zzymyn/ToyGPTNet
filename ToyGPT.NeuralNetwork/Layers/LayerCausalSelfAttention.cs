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
	public static void Forward(ReadOnlySpan2D<float> q, ReadOnlySpan2D<float> k, ReadOnlySpan2D<float> v, Span2D<float> r, float nInf = -1e10f)
	{
		var tmp = new float[q.Height, k.Height].AsSpan2D();

		MMath.MulMT(q, k, tmp);

		{
			var yMax = tmp.Height;
			var xMax = tmp.Width;

			var scale = 1.0f / MathF.Sqrt(xMax);

			for (var y = 0; y < yMax; ++y)
			{
				var row = tmp.GetRowSpan(y);

				for (var x = 0; x < xMax; ++x)
				{
					if (x > y)
					{
						row[x] += nInf;
					}
					else
					{
						row[x] *= scale;
					}
				}

				MMath.Softmax(row, row);
			}
		}

		MMath.MulMM(tmp, v, r);

	}
}
