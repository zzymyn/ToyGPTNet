using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CommunityToolkit.HighPerformance;

namespace ToyGPT.NeuralNetwork
{
	public sealed class CategoricalNeuralNetworkInstance<TLayer, THiddenActivation, TFinalActivation, TFinalActivationLoss>
		: ICategoricalNeuralNetworkInstance
		where TLayer : ILayer, new()
		where THiddenActivation : IActivation, new()
		where TFinalActivation : IActivation, new()
		where TFinalActivationLoss : IActivationLossCategorical, new()
	{
		private readonly TLayer m_Layer = new();
		private readonly THiddenActivation m_HiddenActivation = new();
		private readonly TFinalActivation m_FinalActivation = new();
		private readonly TFinalActivationLoss m_FinalActivationLoss = new();

		private readonly int m_LayerCount;
		private readonly int m_InputNodeCount;
		private readonly int m_OutputNodeCount;

		public readonly float[][,] m_Weights;
		public readonly float[][] m_Biases;
		public readonly float[][,] m_DWeights;
		public readonly float[][] m_DBiases;

		public int m_LastBatchSize = -1;
		public readonly float[][,] m_Output;
		public readonly float[][,] m_Activation;
		public readonly float[][,] m_DActivation;
		public readonly float[][,] m_DInput;
		public float[] m_Loss;

		public CategoricalNeuralNetworkInstance(params (float[,] Weight, float[] Biases)[] layers)
		{
			if (layers.Length < 1)
			{
				throw new ArgumentException("Must have at least 1 layers");
			}
			for (int i = 0; i < layers.Length; ++i)
			{
				if (layers[i].Weight.GetLength(0) != layers[i].Biases.Length)
				{
					throw new ArgumentException("Layer weight and bias dimensions do not match");
				}
			}
			for (int i = 0; i < layers.Length - 1; ++i)
			{
				if (layers[i].Weight.GetLength(0) != layers[i + 1].Weight.GetLength(1))
				{
					throw new ArgumentException("Layer weight dimensions do not match");
				}
			}

			m_LayerCount = layers.Length;
			m_InputNodeCount = layers[0].Weight.GetLength(1);
			m_OutputNodeCount = layers[^1].Weight.GetLength(0);

			m_Weights = new float[m_LayerCount][,];
			m_Biases = new float[m_LayerCount][];
			m_DWeights = new float[m_LayerCount][,];
			m_DBiases = new float[m_LayerCount][];
			m_Output = new float[m_LayerCount][,];
			m_Activation = new float[m_LayerCount][,];
			m_DActivation = new float[m_LayerCount][,];
			m_DInput = new float[m_LayerCount][,];
			m_Loss = new float[0];

			for (int i = 0; i < m_LayerCount; ++i)
			{
				m_Weights[i] = layers[i].Weight;
				m_Biases[i] = layers[i].Biases;
				m_DWeights[i] = ArrayFactory.NewSameSize(layers[i].Weight);
				m_DBiases[i] = ArrayFactory.NewSameSize(layers[i].Biases);
			}
		}

		public void Run(ReadOnlySpan2D<float> inputs, Span2D<float> output)
		{
			Validate.ArraySize(output, inputs.Height, m_OutputNodeCount);

			CreateInternalArrays(inputs.Height);

			m_Layer.Forward(inputs, m_Weights[0], m_Biases[0], m_Output[0]);

			for (int i = 0; i < m_LayerCount - 1; ++i)
			{
				m_HiddenActivation.Forward(m_Output[i], m_Activation[i]);
				m_Layer.Forward(m_Activation[i], m_Weights[i + 1], m_Biases[i + 1], m_Output[i + 1]);
			}

			m_FinalActivation.Forward(m_Output[^1], output);
		}

		public void Forward(ReadOnlySpan2D<float> inputs, ReadOnlySpan<int> expected, Span2D<float> output, out float avgLoss, out float accuracy)
		{
			Validate.ArraySize(inputs, expected.Length, m_InputNodeCount);
			Validate.ArraySize(expected, inputs.Height);
			Validate.ArraySize(output, inputs.Height, m_OutputNodeCount);

			CreateInternalArrays(inputs.Height);

			DoForward(inputs, expected, output, out avgLoss, out accuracy);
		}

		public void Train(ReadOnlySpan2D<float> inputs, ReadOnlySpan<int> expected, float learningRate, out float avgLoss, out float accuracy)
		{
			Validate.ArraySize(inputs, expected.Length, m_InputNodeCount);
			Validate.ArraySize(expected, inputs.Height);

			CreateInternalArrays(inputs.Height);

			DoForward(inputs, expected, m_Activation[^1], out avgLoss, out accuracy);

			m_FinalActivationLoss.Backward(expected, m_Activation[^1], m_DActivation[^1]);

			for (int i = m_LayerCount - 1; i >= 1; --i)
			{
				m_Layer.Backward(m_Activation[i - 1], m_Weights[i], m_DActivation[i], m_DInput[i], m_DWeights[i], m_DBiases[i]);
				m_HiddenActivation.Backward(m_Activation[i - 1], m_DInput[i], m_DActivation[i - 1]);
			}

			m_Layer.Backward(inputs, m_Weights[0], m_DActivation[0], m_DInput[0], m_DWeights[0], m_DBiases[0]);

			for (int i = 0; i < m_LayerCount; ++i)
			{
				for (int y = 0; y < m_Weights[i].GetLength(0); ++y)
					for (int x = 0; x < m_Weights[i].GetLength(1); ++x)
						m_Weights[i][y, x] -= learningRate * m_DWeights[i][y, x];
				for (int x = 0; x < m_Biases[i].Length; ++x)
					m_Biases[i][x] -= learningRate * m_DBiases[i][x];
			}
		}

		private void DoForward(ReadOnlySpan2D<float> inputs, ReadOnlySpan<int> expected, Span2D<float> output, out float avgLoss, out float accuracy)
		{
			m_Layer.Forward(inputs, m_Weights[0], m_Biases[0], m_Output[0]);

			for (int i = 0; i < m_LayerCount - 1; ++i)
			{
				m_HiddenActivation.Forward(m_Output[i], m_Activation[i]);
				m_Layer.Forward(m_Activation[i], m_Weights[i + 1], m_Biases[i + 1], m_Output[i + 1]);
			}

			m_FinalActivationLoss.Forward(m_Output[^1], expected, output, m_Loss);

			avgLoss = m_Loss.Average();
			accuracy = AccuracyCategorical.Compute(output, expected);
		}

		private void CreateInternalArrays(int batchSize)
		{
			if (batchSize <= 0)
				throw new ArgumentException(null, nameof(batchSize));

			if (m_LastBatchSize == batchSize)
				return;

			m_LastBatchSize = batchSize;

			m_Loss = new float[batchSize];

			for (int i = 0; i < m_LayerCount; ++i)
			{
				m_Output[i] = new float[batchSize, m_Weights[i].GetLength(0)];
				m_Activation[i] = new float[batchSize, m_Weights[i].GetLength(0)];
				m_DActivation[i] = new float[batchSize, m_Weights[i].GetLength(0)];
				m_DInput[i] = new float[batchSize, m_Weights[i].GetLength(1)];
			}
		}
	}
}
