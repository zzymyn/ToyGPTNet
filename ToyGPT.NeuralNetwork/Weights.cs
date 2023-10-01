using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CommunityToolkit.HighPerformance;

namespace ToyGPT.NeuralNetwork;

public static class Weights
{
	public static float[,] CreateRandomWeights(int numInputs, int numOutputs, Random rng)
	{
		var weights = new float[numOutputs, numInputs];

		foreach (ref var w in weights.AsSpan())
		{
			w = (float)(0.1 * rng.NextNormal());
		}

		return weights;
	}
}
