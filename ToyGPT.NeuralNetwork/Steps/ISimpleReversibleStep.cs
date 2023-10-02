using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CommunityToolkit.HighPerformance;

namespace ToyGPT.NeuralNetwork.Steps;

public interface ISimpleReversibleStep
{
	ReadOnlyMemory2D<float> Forward(ReadOnlySpan2D<float> inputs);
	ReadOnlyMemory2D<float> Backward(ReadOnlySpan2D<float> inputs, ReadOnlySpan2D<float> dValues);
}
