using System.CommandLine;
using CommunityToolkit.HighPerformance;
using ToyGPT.Lib;
using ToyGPT.NeuralNetwork;

namespace ToyGPT
{
	class Program
	{
		private static async Task Main(string[] args)
		{
			var fileArg = new Argument<FileInfo>("file", "The file to read");

			var rootCommand = new RootCommand
			{
				fileArg
			};

			rootCommand.SetHandler(async context =>
			{
				var ct = context.GetCancellationToken();
				var file = context.ParseResult.GetValueForArgument(fileArg);
				var console = context.Console;

				var rng = new Random(42);

				var relu = new ActivationReLU();
				var softmax = new ActivationSoftMax();

				var (X, y) = CreateData(100, 3, rng);

				var layer0 = new LayerDense(2, 3);
				var weights0 = Weights.CreateRandomWeights(2, 3, rng);
				var biases0 = new float[3];

				var layer1 = new LayerDense(3, 3);
				var weights1 = Weights.CreateRandomWeights(3, 3, rng);
				var biases1 = new float[3];

				var values00 = new float[300, layer0.NeuronCount];
				var values01 = new float[300, layer0.NeuronCount];
				var values10 = new float[300, layer1.NeuronCount];
				var values11 = new float[300, layer1.NeuronCount];

				layer0.Forward(X, weights0, biases0, values00);
				relu.Forward(values00, values01);
				layer1.Forward(values01, weights1, biases1, values10);
				softmax.Forward(values10, values11);

				var losses = new float[300];
				var cce = new LossCategoricalCrossEntropy();
				cce.Forward(values11, y, losses);
				var avgLoss = losses.Average();

				Print2D(console, values11);

				//var weights1 = new[] {
				//	new float [] { 0.1f, -0.14f, 0.5f },
				//	new float [] { -0.5f, 0.12f, -0.33f },
				//	new float [] { -0.44f, 0.73f, -0.13f },
				//};
				//var bias1 = new float[] { -1, 2, -0.5f };

				//var values2 = new float[values1.Length][];
				//for (int i = 0; i < values1.Length; ++i)
				//{
				//	values2[i] = new float[weights1.Length];
				//}

				//for (int i = 0; i < values0.Length; ++i)
				//{
				//	var input = values0[i];
				//	for (int bi = 0; bi < bias0.Length; ++bi)
				//	{
				//		values1[i][bi] = Neuron.RunNeuron(input, weights0[bi], bias0[bi]);
				//	}
				//}

				//for (int i = 0; i < values1.Length; ++i)
				//{
				//	var input = values1[i];
				//	for (int bi = 0; bi < bias1.Length; ++bi)
				//	{
				//		values2[i][bi] = Neuron.RunNeuron(input, weights1[bi], bias1[bi]);
				//	}
				//}

				//console.WriteLine($"{string.Join(", ", values1.Select(a => a.ToString()))}");
			});

			await rootCommand.InvokeAsync(args);
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

		private static (float[,] X, int[] y) CreateData(int points, int classes, Random rng)
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