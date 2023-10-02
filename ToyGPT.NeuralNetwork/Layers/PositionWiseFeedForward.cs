using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CommunityToolkit.HighPerformance;
using ToyGPT.NeuralNetwork.Activations;

namespace ToyGPT.NeuralNetwork.Layers;

public sealed class PositionWiseFeedForward
{
	public LinearWeightsTransposedWithBias m_Up;
	public GeLU m_Activation;
	public LinearWeightsTransposedWithBias m_Down;

	public ReadOnlyMemory2D<float> Outputs => m_Down.Outputs;

	public PositionWiseFeedForward(LinearWeightsTransposedWithBias fc, LinearWeightsTransposedWithBias proj)
	{
		m_Up = fc;
		m_Activation = new GeLU();
		m_Down = proj;
	}

	public ReadOnlyMemory2D<float> Forward(ReadOnlySpan2D<float> inputs)
	{
		var fcOut = m_Up.Forward(inputs);
		var actOut = m_Activation.Forward(fcOut.Span);
		return m_Down.Forward(actOut.Span);
	}
}
