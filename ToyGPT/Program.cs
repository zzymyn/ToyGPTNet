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

				var (data, expected) = CreateBlobs(100, 3, rng);
				var relu = new ActivationReLU();
				var softmaxLoss = new ActivationLossSoftMaxCategoricalCrossEntropy();
				var layer0 = new LayerDense(2, HiddenLayerNeurons);
				var layer1 = new LayerDense(HiddenLayerNeurons, 3);

				var weights0 = Weights.CreateRandomWeights(2, HiddenLayerNeurons, rng);
				var biases0 = new float[8];
				var weights1 = Weights.CreateRandomWeights(HiddenLayerNeurons, 3, rng);
				var biases1 = new float[3];
				var values0 = new float[300, HiddenLayerNeurons];
				var values0Act = new float[300, HiddenLayerNeurons];
				var values1 = new float[300, 3];
				var values1Act = new float[300, 3];
				var losses = new float[300];
				var dInputs1Act = new float[300, 3];
				var dInputs1 = new float[300, HiddenLayerNeurons];
				var dWeights1 = new float[3, HiddenLayerNeurons];
				var dBiases1 = new float[3];
				var dInputs0Act = new float[300, HiddenLayerNeurons];
				var dInputs0 = new float[300, 2];
				var dWeights0 = new float[HiddenLayerNeurons, 2];
				var dBiases0 = new float[HiddenLayerNeurons];
				var learningFactor = 0.01f;

				for (long loopCount = 0; true; ++loopCount)
				{
					layer0.Forward(data, weights0, biases0, values0);
					relu.Forward(values0, values0Act);
					layer1.Forward(values0Act, weights1, biases1, values1);
					softmaxLoss.Forward(values1, expected, values1Act, losses);
					var avgLoss = losses.Average();
					var accuracy = AccuracyCategorical.Compute(values1Act, expected);

					console.WriteLine($"{loopCount}: {avgLoss:0.0000} - {accuracy * 100:#0.0}%");

					softmaxLoss.Backward(expected, values1Act, dInputs1Act);
					layer1.Backward(values0Act, weights1, dInputs1Act, dInputs1, dWeights1, dBiases1);
					relu.Backward(values0Act, dInputs1, dInputs0Act);
					layer0.Backward(data, weights0, dInputs0Act, dInputs0, dWeights0, dBiases0);

					for (int y = 0; y < weights0.GetLength(0); ++y)
						for (int x = 0; x < weights0.GetLength(1); ++x)
							weights0[y, x] -= learningFactor * dWeights0[y, x];
					for (int i = 0; i < biases0.Length; ++i)
						biases0[i] -= learningFactor * dBiases0[i];
					for (int y = 0; y < weights1.GetLength(0); ++y)
						for (int x = 0; x < weights1.GetLength(1); ++x)
							weights1[y, x] -= learningFactor * dWeights1[y, x];
					for (int i = 0; i < biases1.Length; ++i)
						biases1[i] -= learningFactor * dBiases1[i];
				}

				//Print2D(console, dWeights0);
				//Print(console, dBiases0);
				//Print2D(console, dWeights1);
				//Print(console, dBiases1);
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