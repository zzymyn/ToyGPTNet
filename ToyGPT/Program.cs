using System.CommandLine;
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

				var values0 = new[] {
					new float [] { 1, 2, 3, 2.5f },
					new float [] { 2, 5, -1, 2 },
					new float [] { -1.5f, 2.7f, 3.3f, -0.8f },
				};

				var weights0 = new[] {
					new float [] { 0.2f, 0.8f, -0.5f, 1.0f },
					new float [] { 0.5f, -0.91f, 0.26f, -0.5f },
					new float [] { -0.26f, -0.27f, 0.17f, 0.87f },
				};
				var bias0 = new float[] { 2, 3, 0.5f };

				var values1 = new float[values0.Length][];
				for (int i = 0; i < values0.Length; ++i)
				{
					values1[i] = new float[weights0.Length];
				}

				var weights1 = new[] {
					new float [] { 0.1f, -0.14f, 0.5f },
					new float [] { -0.5f, 0.12f, -0.33f },
					new float [] { -0.44f, 0.73f, -0.13f },
				};
				var bias1 = new float[] { -1, 2, -0.5f };

				var values2 = new float[values1.Length][];
				for (int i = 0; i < values1.Length; ++i)
				{
					values2[i] = new float[weights1.Length];
				}

				for (int i = 0; i < values0.Length; ++i)
				{
					var input = values0[i];
					for (int bi = 0; bi < bias0.Length; ++bi)
					{
						values1[i][bi] = Neuron.RunNeuron(input, weights0[bi], bias0[bi]);
					}
				}

				for (int i = 0; i < values1.Length; ++i)
				{
					var input = values1[i];
					for (int bi = 0; bi < bias1.Length; ++bi)
					{
						values2[i][bi] = Neuron.RunNeuron(input, weights1[bi], bias1[bi]);
					}
				}

				console.WriteLine($"{string.Join(", ", values1.Select(a => a.ToString()))}");
			});

			await rootCommand.InvokeAsync(args);
		}
	}
}