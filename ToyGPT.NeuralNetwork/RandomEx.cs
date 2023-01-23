using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ToyGPT.NeuralNetwork
{
	public static class RandomEx
	{
		public static double NextNormal(this Random rng)
		{
			var u = rng.NextDouble();
			var v = rng.NextDouble();

			while (u <= float.Epsilon)
			{
				Debugger.Break();
				u = rng.NextDouble();
			}

			return Math.Sqrt(-2.0 * Math.Log(u)) * Math.Sin(2.0 * Math.PI * v);
		}
	}
}
