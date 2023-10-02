using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CommunityToolkit.HighPerformance;

namespace ToyGPT.NeuralNetwork.Steps;

public class CompoundForwardStep<L, R>
	: INeuralNetworkForwardStep
	where L : INeuralNetworkForwardStep
	where R : INeuralNetworkForwardStep
{
	private readonly L m_Left;
	private readonly R m_Right;

	public CompoundForwardStep(L left, R right)
	{
		m_Left = left;
		m_Right = right;
	}

	public ReadOnlyMemory2D<float> Outputs => m_Right.Outputs;

	public ReadOnlyMemory2D<float> Forward(ReadOnlySpan2D<float> inputs)
	{
		var left = m_Left.Forward(inputs);
		return m_Right.Forward(left.Span);
	}
}
