using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CommunityToolkit.HighPerformance;
using ToyGPT.NeuralNetwork.Layers;

namespace ToyGPT.NeuralNetwork.Layers;

public static class LayerAttention
{
	public static void Forward(ReadOnlySpan2D<float> q, ReadOnlySpan2D<float> k, ReadOnlySpan2D<float> v, Span2D<float> r)
	{
		var tmp = new float[q.Height, k.Height].AsSpan2D();

		MMath.MulMT(q, k, tmp);

		{
			var yMax = tmp.Height;
			var xMax = tmp.Width;

			var scale = 1.0f / MathF.Sqrt(k.Width);

			for (var y = 0; y < yMax; ++y)
			{
				var row = tmp.GetRowSpan(y);

				for (var x = 0; x < xMax; ++x)
				{
					row[x] *= scale;
				}

				MMath.Softmax(row, row);
			}
		}

		MMath.MulMM(tmp, v, r);
	}
}
