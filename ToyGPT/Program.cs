using System.CommandLine;
using CommunityToolkit.HighPerformance;
using ToyGPT.Lib;
using ToyGPT.NeuralNetwork;

namespace ToyGPT
{
	class Program
	{
		const int HiddenLayerNeurons = 8;

		private static async Task Main(string[] args)
		{
			var rootCommand = new RootCommand
			{
			};

			rootCommand.SetHandler(async context =>
			{
				var ct = context.GetCancellationToken();
				var console = context.Console;

				var rng = new Random(42);

				var (data, expected) = CreateSpiral(100, 3, rng);
				
				var weights0 = Weights.CreateRandomWeights(2, HiddenLayerNeurons, rng);
				var biases0 = new float[8];
				var weights1 = Weights.CreateRandomWeights(HiddenLayerNeurons, 3, rng);
				var biases1 = new float[3];

				var nn = new CategoricalNeuralNetworkInstance<LayerDense, ActivationReLU, ActivationLossSoftMaxCategoricalCrossEntropy>(
					(weights0, biases0),
					(weights1, biases1)
					);

				for (long loopCount = 0; true; ++loopCount)
				{
					nn.Train(data, expected, 0.01f, out var b, out var a);
					console.WriteLine($"{loopCount}: {b:0.000000} - {a * 100:#0.000}%");
				}
			});

			await rootCommand.InvokeAsync(args);
		}

		private static void Print(IConsole console, float[] a)
		{
			console.Write("[");
			for (int x = 0; x < a.Length; ++x)
			{
				if (x != 0)
					console.Write(", ");
				console.Write(a[x].ToString());
			}
			console.WriteLine("]");
		}

		private static void Print2D(IConsole console, float[,] a)
		{
			console.Write("[");
			for (int y = 0; y < a.GetLength(0); ++y)
			{
				if (y != 0)
					console.Write(",\n ");
				console.Write("[");
				for (int x = 0; x < a.GetLength(1); ++x)
				{
					if (x != 0)
						console.Write(", ");
					console.Write(a[y, x].ToString());
				}
				console.Write("]");
			}
			console.WriteLine("]");
		}

		private static (float[,] X, int[] y) CreateBlobs(int points, int classes, Random rng)
		{
			var len = points * classes;
			var X = new float[len, 2];
			var y = new int[len];

			for (int i = 0; i < len; ++i)
			{
				var classNumber = i / points;
				var classI = i - classNumber * points;
				X[i, 0] = 3 * classNumber + (float)rng.NextNormal();
				X[i, 1] = (float)rng.NextNormal();
				y[i] = classNumber;
			}

			return (X, y);
		}

		private static (float[,] X, int[] y) CreateSpiral(int points, int classes, Random rng)
		{
			var len = points * classes;
			var X = new float[len, 2];
			var y = new int[len];

			for (int i = 0; i < len; ++i)
			{
				var classNumber = i / points;
				var classI = i - classNumber * points;
				var r = (float)classI / points;
				var t = classNumber * 4 + r * 4 + 0.2f * (float)rng.NextNormal();
				X[i, 0] = r * MathF.Sin(2.5f * t);
				X[i, 1] = r * MathF.Cos(2.5f * t);
				y[i] = classNumber;
			}

			return (X, y);
		}
	}
}