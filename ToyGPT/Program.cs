using System.CommandLine;
using CommunityToolkit.HighPerformance;
using ToyGPT.Lib;
using ToyGPT.NeuralNetwork;

namespace ToyGPT
{
	class Program
	{
		private const int BatchSize = 3;

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

				var activation = new ActivationReLU();

				var X = new float[BatchSize, 4] {
					{ 1, 2, 3, 2.5f },
					{ 2, 5, -1, 2 },
					{ -1.5f, 2.7f, 3.3f, -0.8f },
				};

				var layer0 = LayerDense.CreateNewRandom(4, 3, rng);
				var layer1 = LayerDense.CreateNewRandom(3, 3, rng);

				var values1 = new float[BatchSize, layer0.NeuronCount];
				var values2 = new float[BatchSize, layer1.NeuronCount];

				layer0.Forward(X, values1);
				activation.Forward(values1);
				layer1.Forward(values1, values2);
				activation.Forward(values2);

				console.Write("[");
				for (int y = 0; y < values2.GetLength(0); ++y)
				{
					if (y != 0)
						console.Write(",\n ");
					console.Write("[");
					for (int x = 0; x < values2.GetLength(1); ++x)
					{
						if (x != 0)
							console.Write(", ");
						console.Write(values2[y, x].ToString());
					}
					console.Write("]");
				}
				console.WriteLine("]");

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
	}
}