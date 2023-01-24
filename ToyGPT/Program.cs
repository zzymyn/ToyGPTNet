using System.CommandLine;
using System.Diagnostics;
using CommunityToolkit.HighPerformance;
using ToyGPT.Lib;
using ToyGPT.NeuralNetwork;
using Microsoft.Maui.Graphics;
using Microsoft.Maui.Graphics.Skia;

namespace ToyGPT
{
	class Program
	{
		const int HiddenLayerNeurons = 64;

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

				var weights0 = Weights.CreateRandomWeights(2, HiddenLayerNeurons, rng);
				var biases0 = new float[HiddenLayerNeurons];
				var weights1 = Weights.CreateRandomWeights(HiddenLayerNeurons, 3, rng);
				var biases1 = new float[3];
				var testOutput = new float[150, 3];

				var nn = new CategoricalNeuralNetworkInstance<LayerDense, ActivationReLU, ActivationSoftMax, ActivationLossSoftMaxCategoricalCrossEntropy>(
					(weights0, biases0),
					(weights1, biases1)
					);

				var sw = Stopwatch.StartNew();

				var (testX, testY) = CreateSpiral(100, 3, rng);

				for (long loopCount = 0; loopCount <= 25000; ++loopCount)
				{
					ct.ThrowIfCancellationRequested();

					nn.Train(testX, testY, 0.5f, out var avgLoss, out var accuracy);

					if (loopCount % 100 == 0)
					{
						WriteImage(testX, testY, nn, loopCount);
						var ms = sw.ElapsedMilliseconds;
						var loopsPerSecond = ms > 0 ? 1000 * loopCount / sw.ElapsedMilliseconds : 0;
						console.WriteLine($"{loopCount}: {avgLoss:0.000000} - {accuracy * 100:#0.000}% {loopsPerSecond} LPS");
					}
				}
			});

			await rootCommand.InvokeAsync(args);
		}

		const int ImageSize = 600;
		const float AxisXMin = -1.5f;
		const float AxisXMax = 1.5f;
		const float AxisYMin = -1.5f;
		const float AxisYMax = 1.5f;

		private static float XToDraw(float x) => (x - AxisXMin) / (AxisXMax - AxisXMin) * ImageSize;
		private static float YToDraw(float y) => (y - AxisYMin) / (AxisYMax - AxisYMin) * ImageSize;
		private static float DrawToX(float px) => (px / ImageSize) * (AxisXMax - AxisXMin) + AxisXMin;
		private static float DrawToY(float py) => (py / ImageSize) * (AxisYMax - AxisYMin) + AxisYMin;

		private static void WriteImage(float[,] testX, int[] testY, ICategoricalNeuralNetworkInstance nn, long gen)
		{
			var bmp = new SkiaBitmapExportContext(ImageSize, ImageSize, 1.0f);
			var canvas = bmp.Canvas;

			canvas.FillColor = Colors.White;
			canvas.FillRectangle(0, 0, ImageSize, ImageSize);

			var nnIn = new float[ImageSize * ImageSize, 2];
			for (int y = 0; y < ImageSize; ++y)
			{
				for (int x = 0; x < ImageSize; ++x)
				{
					nnIn[y * ImageSize + x, 0] = DrawToX(x);
					nnIn[y * ImageSize + x, 1] = DrawToX(y);
				}
			}

			var nnOut = new float[ImageSize * ImageSize, 3];

			nn.Run(nnIn, nnOut);

			for (int y = 0; y < ImageSize; ++y)
			{
				for (int x = 0; x < ImageSize; ++x)
				{
					canvas.FillColor = new Color(nnOut[y * ImageSize + x, 0], nnOut[y * ImageSize + x, 1], nnOut[y * ImageSize + x, 2]);
					canvas.FillRectangle(x, y, 1, 1);
				}
			}

			canvas.StrokeColor = Colors.Black;
			canvas.DrawLine(XToDraw(AxisXMin), YToDraw(0.0f), XToDraw(AxisXMax), YToDraw(0.0f));
			canvas.DrawLine(XToDraw(0.0f), YToDraw(AxisYMin), XToDraw(0.0f), YToDraw(AxisYMax));

			for (int i = 0; i < testX.GetLength(0); ++i)
			{
				var x = testX[i, 0];
				var y = testX[i, 1];
				var c = testY[i];

				var drawX = XToDraw(x);
				var drawY = YToDraw(y);

				canvas.FillColor = c switch
				{
					0 => Colors.Red,
					1 => Colors.Green,
					_ => Colors.Blue,
				};
				canvas.StrokeColor = Colors.Black;
				canvas.FillCircle(drawX, drawY, 3.0f);
				canvas.DrawCircle(drawX, drawY, 3.0f);
			}

			bmp.WriteToFile($"nn{gen:00000}.png");
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
				var t = classNumber * 4 + r * 4;// + 0.2f * (float)rng.NextNormal();
				X[i, 0] = r * MathF.Sin(2.5f * t);
				X[i, 1] = r * MathF.Cos(2.5f * t);
				y[i] = classNumber;
			}

			return (X, y);
		}
	}
}