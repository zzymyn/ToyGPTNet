using System;
using System.Threading.Tasks;
using System.Windows.Input;
using System.Windows.Media;
using Microsoft.Maui.Graphics;
using Microsoft.Maui.Graphics.Skia;
using SkiaSharp.Views.WPF;
using ToyGPT.NeuralNetwork;
using Color = Microsoft.Maui.Graphics.Color;
using Colors = Microsoft.Maui.Graphics.Colors;

namespace ToyGPTNet.NeuralNetwork.UI.ViewModel
{
	internal class RootViewModel
		: NotifierBase
	{
		private readonly Random m_Rng = new(45);
		private float[,] m_X;
		private int[] m_Y;

		private Task? m_OutputDrawTask;

		private int m_HiddenLayerNeurons = 64;
		private readonly object m_NNLock = new();
		private CategoricalNeuralNetworkInstance m_NeuralNetwork;
		private float[,] m_Weights0;
		private float[] m_Biases0;
		private float[,] m_Weights1;
		private float[] m_Biases1;
		private float m_DataRandomness = 0.02f;
		private int m_TrainingEpochs = 10000;
		private float m_LearningRate = 1.0f;
		private float m_LearningRateDecay = 0.001f;
		private int m_TrainingProgress = 0;
		private bool m_IsTrainingEnabled = true;

		private ImageSource? m_Image;
		public ImageSource? Image
		{
			get => m_Image;
			private set => SetField(ref m_Image, value);
		}

		public float DataRandomness
		{
			get => m_DataRandomness;
			set
			{
				if (SetField(ref m_DataRandomness, value))
				{
					lock (m_NNLock)
					{
						(m_X, m_Y) = CreateSpiral(100, 3, m_Rng);
					}
				}
			}
		}

		public int TrainingEpochs
		{
			get => m_TrainingEpochs;
			set => SetField(ref m_TrainingEpochs, value);
		}

		public float LearningRate
		{
			get => m_LearningRate;
			set => SetField(ref m_LearningRate, value);
		}

		public float LearningRateDecay
		{
			get => m_LearningRateDecay;
			set => SetField(ref m_LearningRateDecay, value);
		}

		public int TrainingProgress
		{
			get => m_TrainingProgress;
			private set => SetField(ref m_TrainingProgress, value);
		}

		public bool IsTrainingEnabled
		{
			get => m_IsTrainingEnabled;
			private set => SetField(ref m_IsTrainingEnabled, value);
		}

		public ICommand ResetCommand => new ActionCommand(Reset);

		public ICommand TrainCommand => new ActionCommand(Train);

		public RootViewModel()
		{
			m_Weights0 = Weights.CreateRandomWeights(2, m_HiddenLayerNeurons, m_Rng);
			m_Biases0 = new float[m_HiddenLayerNeurons];
			m_Weights1 = Weights.CreateRandomWeights(m_HiddenLayerNeurons, 3, m_Rng);
			m_Biases1 = new float[3];

			m_NeuralNetwork = new CategoricalNeuralNetworkInstance(300,
				(m_Weights0, m_Biases0),
				(m_Weights1, m_Biases1)
				);

			(m_X, m_Y) = CreateSpiral(100, 3, m_Rng);

			m_OutputDrawTask = DrawOutputView(600, 600);
		}

		private void Reset()
		{
			lock (m_NNLock)
			{
				m_Weights0 = Weights.CreateRandomWeights(2, m_HiddenLayerNeurons, m_Rng);
				m_Biases0 = new float[m_HiddenLayerNeurons];
				m_Weights1 = Weights.CreateRandomWeights(m_HiddenLayerNeurons, 3, m_Rng);
				m_Biases1 = new float[3];

				m_NeuralNetwork = new CategoricalNeuralNetworkInstance(300,
					(m_Weights0, m_Biases0),
					(m_Weights1, m_Biases1)
					);
			}
		}

		private async void Train()
		{
			IsTrainingEnabled = false;
			TrainingProgress = 0;

			await Task.Run(() =>
			{
				for (int loopCount = 0; loopCount <= m_TrainingEpochs; ++loopCount)
				{
					TrainingProgress = loopCount * 100 / m_TrainingEpochs;

					var lr = m_LearningRate / (1.0f + m_LearningRateDecay * loopCount);
					lock (m_NNLock)
					{
						m_NeuralNetwork.Train(m_X, m_Y, lr, out var avgLoss, out var accuracy);
					}
				}
			});

			IsTrainingEnabled = true;
		}

		internal async Task DrawOutputView(int width, int height)
		{
			while (true)
			{
				await Task.Run(() =>
				{
					float[,] dataX;
					int[] dataY; 
					ICategoricalNeuralNetworkInstance imgNN;
					lock (m_NNLock)
					{
						dataX = m_X;
						dataY = m_Y;
						imgNN = m_NeuralNetwork.CopyWithBatchSize(width * height);
					}

					var bmp = new SkiaBitmapExportContext(width, height, 1.0f);
					var canvas = bmp.Canvas;

					var AxisXMin = -1.5f;
					var AxisXMax = 1.5f;
					var AxisYMin = -1.5f;
					var AxisYMax = 1.5f;

					if (width > height)
					{
						AxisXMin *= (float)width / height;
						AxisXMax *= (float)width / height;
					}
					else
					{
						AxisYMin *= (float)height / width;
						AxisYMax *= (float)height / width;
					}

					float XToDraw(float x) => (x - AxisXMin) / (AxisXMax - AxisXMin) * width;
					float YToDraw(float y) => (y - AxisYMin) / (AxisYMax - AxisYMin) * height;
					float DrawToX(float px) => (px / width) * (AxisXMax - AxisXMin) + AxisXMin;
					float DrawToY(float py) => (py / height) * (AxisYMax - AxisYMin) + AxisYMin;

					canvas.FillColor = Colors.NavajoWhite;
					canvas.FillRectangle(0, 0, width, height);

					var nnIn = new float[width * height, 2];
					for (int y = 0; y < height; ++y)
					{
						for (int x = 0; x < width; ++x)
						{
							nnIn[y * width + x, 0] = DrawToX(x);
							nnIn[y * width + x, 1] = DrawToX(y);
						}
					}

					var nnOut = imgNN.Run(nnIn).Span;

					for (int y = 0; y < height; ++y)
					{
						for (int x = 0; x < width; ++x)
						{
							canvas.FillColor = new Color(nnOut[y * width + x, 0], nnOut[y * width + x, 1], nnOut[y * width + x, 2]);
							canvas.FillRectangle(x, y, 1, 1);
						}
					}

					canvas.StrokeColor = Colors.Black;
					canvas.DrawLine(XToDraw(AxisXMin), YToDraw(0.0f), XToDraw(AxisXMax), YToDraw(0.0f));
					canvas.DrawLine(XToDraw(0.0f), YToDraw(AxisYMin), XToDraw(0.0f), YToDraw(AxisYMax));

					for (int i = 0; i < dataX.GetLength(0); ++i)
					{
						var x = dataX[i, 0];
						var y = dataX[i, 1];
						var c = dataY[i];

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

					var wb = bmp.Bitmap.ToWriteableBitmap();
					wb.Freeze();
					Image = wb;
				});
			}
		}

		private (float[,] X, int[] y) CreateSpiral(int points, int classes, Random rng)
		{
			var len = points * classes;
			var X = new float[len, 2];
			var y = new int[len];

			for (int i = 0; i < len; ++i)
			{
				var classNumber = i / points;
				var classI = i - classNumber * points;
				var r = (float)classI / points;
				var t = classNumber * 4 + r * 4 + m_DataRandomness * (float)rng.NextNormal();
				X[i, 0] = r * MathF.Sin(2.5f * t);
				X[i, 1] = r * MathF.Cos(2.5f * t);
				y[i] = classNumber;
			}

			return (X, y);
		}

	}
}
