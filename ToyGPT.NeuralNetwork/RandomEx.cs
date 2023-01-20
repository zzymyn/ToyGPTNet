using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ToyGPT.NeuralNetwork
{
	internal static class RandomEx
	{
		public static double NextNormal(this Random rng)
		{
			var u = rng.NextDouble();
			var v = rng.NextDouble();
			return Math.Sqrt(-2.0 * Math.Log(u)) * Math.Sin(2.0 * Math.PI * v);
		}
	}
}
