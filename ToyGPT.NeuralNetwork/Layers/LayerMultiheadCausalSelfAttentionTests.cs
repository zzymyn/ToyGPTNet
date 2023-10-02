using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CommunityToolkit.HighPerformance;

namespace ToyGPT.NeuralNetwork.Layers;

public static class LayerMultiheadCausalSelfAttention
{
	public static void Forward(ReadOnlySpan2D<float> inputs, ReadOnlySpan2D<float> attnWT, ReadOnlySpan<float> attnB, ReadOnlySpan2D<float> projWT, ReadOnlySpan<float> projB, int headCount, Span2D<float> outputs)
	{
		Validate.ArraysSameSize(inputs, outputs);
		Validate.True(inputs.Width == attnWT.Width);
		Validate.True(attnWT.Height == attnB.Length);
		Validate.True(3 * projWT.Height == attnWT.Height);
		Validate.True(projWT.Height == projB.Length);
		Validate.True(projWT.Height % headCount == 0);

		var projection = new float[inputs.Height, attnWT.Height].AsSpan2D();
		var attns = new float[inputs.Height, projWT.Height].AsSpan2D();

		MMath.MulMTAddR(inputs, attnWT, attnB, projection);

		var qvkStep = projWT.Height;
		var headStep = qvkStep / headCount;
		var q0 = 0;
		var k0 = qvkStep;
		var v0 = qvkStep * 2;
		var a0 = 0;

		for (int head = 0; head < headCount; ++head)
		{
			var q1 = q0 + headStep;
			var k1 = k0 + headStep;
			var v1 = v0 + headStep;
			var a1 = a0 + headStep;

			var q = projection[.., q0..q1];
			var k = projection[.., k0..k1];
			var v = projection[.., v0..v1];
			var attn = attns[.., a0..a1];

			LayerCausalSelfAttention.Forward(q, k, v, attn);

			q0 = q1;
			k0 = k1;
			v0 = v1;
			a0 = a1;
		}

		MMath.MulMTAddR(attns, projWT, projB, outputs);
	}
}
