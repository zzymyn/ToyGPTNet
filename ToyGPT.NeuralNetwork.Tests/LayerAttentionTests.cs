using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NUnit.Framework;
using ToyGPT.NeuralNetwork.Layers;

namespace ToyGPT.NeuralNetwork.Tests;

internal class LayerAttentionTests
{
	[Test]
	public void ForwardTest1()
	{
		var q = new float[,]
		{
			{ 0.116f, 0.159f, 0.055f, 0.226f, 0.443f },
			{ 0.180f, 0.397f, 0.142f, 0.106f, 0.175f },
			{ 0.156f, 0.453f, 0.028f, 0.129f, 0.234f },
			{ 0.499f, 0.055f, 0.133f, 0.017f, 0.000f },
			{ 0.089f, 0.290f, 0.240f, 0.228f, 0.153f },
		};

		var expected = new float[,]
		{
			{ 0.20416899f, 0.27249661f, 0.11859655f, 0.14312277f, 0.20524905f },
			{ 0.20571475f, 0.27385849f, 0.11928766f, 0.14177291f, 0.20205442f },
			{ 0.20466852f, 0.27472538f, 0.11882750f, 0.14217771f, 0.20319328f },
			{ 0.21261882f, 0.26815983f, 0.11988719f, 0.13896664f, 0.19737467f },
			{ 0.20504691f, 0.27338890f, 0.11970467f, 0.14237454f, 0.20249782f },
		};

		var outputs = ArrayFactory.NewSameSize(q);

		LayerAttention.Forward(q, q, q, outputs);

		Assert.That(outputs, Is.EqualTo(expected).Within(1e-6f));
	}

}
