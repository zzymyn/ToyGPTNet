using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CommunityToolkit.HighPerformance;

namespace ToyGPT.NeuralNetwork.Layers;

public static class LayerMultiheadCausalSelfAttention
{
	public static void Forward(
		ReadOnlySpan2D<float> inputs,
		ReadOnlySpan2D<float> attnWT,
		ReadOnlySpan<float> attnB,
		ReadOnlySpan2D<float> projWT,
		ReadOnlySpan<float> projB,
		int headCount,
		Span2D<float> outputs,
		float nInf = -1e10f)
	{
		Validate.ArraysSameSize(inputs, outputs);
		Validate.True(inputs.Width == attnWT.Width);
		Validate.True(attnWT.Height == attnB.Length);
		Validate.True(3 * projWT.Height == attnWT.Height);
		Validate.True(projWT.Height == projB.Length);
		Validate.True(projWT.Height % headCount == 0);

		var projection = new float[inputs.Height, attnWT.Height].AsSpan2D();
		var attns = new float[inputs.Height, projWT.Height].AsSpan2D();
		var tmp = new float[inputs.Height, inputs.Height].AsSpan2D();

		LayerDense.ForwardMT(inputs, attnWT, attnB, projection);

		var qvkStep = projWT.Height;
		var headStep = qvkStep / headCount;
		var q0 = 0;
		var k0 = qvkStep;
		var v0 = qvkStep * 2;
		var csaScale = 1.0f / MathF.Sqrt(headStep);

		for (int head = 0; head < headCount; ++head)
		{
			var q1 = q0 + headStep;
			var k1 = k0 + headStep;
			var v1 = v0 + headStep;

			var q = projection[.., q0..q1];
			var k = projection[.., k0..k1];
			var v = projection[.., v0..v1];
			var attn = attns[.., q0..q1];

			LayerDense.ForwardMT(q, k, tmp);

			for (int y = 0, yMax = tmp.Height; y < yMax; ++y)
			{
				var row = tmp.GetRowSpan(y);
				MMath.CausalSelfAttentionAndSoftmax(row, row, y, csaScale, nInf);
			}

			LayerDense.ForwardMM(tmp, v, attn);

			q0 = q1;
			k0 = k1;
			v0 = v1;
		}

		LayerDense.ForwardMT(attns, projWT, projB, outputs);
	}
}
