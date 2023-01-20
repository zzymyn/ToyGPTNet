using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CommunityToolkit.HighPerformance;

namespace ToyGPT.NeuralNetwork
{
	public class ActivationReLU
	{
		public void Forward(Span2D<float> inputs)
		{
			foreach (ref var a in inputs)
			{
				if (a < 0)
					a = 0;
			}
		}
	}
}
