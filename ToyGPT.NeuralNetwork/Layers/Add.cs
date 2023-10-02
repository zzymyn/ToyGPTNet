using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CommunityToolkit.HighPerformance;

namespace ToyGPT.NeuralNetwork.Layers;

public sealed class Add
{
	private float[,]? m_Outputs;

	public ReadOnlyMemory2D<float> Outputs => m_Outputs;

	public Add()
	{
	}

	public ReadOnlyMemory2D<float> Forward(ReadOnlySpan2D<float> a, ReadOnlySpan2D<float> b)
	{
		ArrayFactory.Resize(ref m_Outputs, a.Height, a.Width);

		MMath.Add(a, b, m_Outputs);

		return m_Outputs;
	}
}
