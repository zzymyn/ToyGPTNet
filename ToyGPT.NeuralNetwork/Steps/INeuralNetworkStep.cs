using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ToyGPT.NeuralNetwork.Steps
{
	public interface INeuralNetworkStep
		: INeuralNetworkForwardStep
		, INeuralNetworkBackwardStep
	{
	}
}
