using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CommunityToolkit.HighPerformance;
using ToyGPT.NeuralNetwork.ActivationLoss;
using ToyGPT.NeuralNetwork.Activations;
using ToyGPT.NeuralNetwork.Layers;

namespace ToyGPT.NeuralNetwork;

public sealed class CategoricalNeuralNetworkInstance
	: ICategoricalNeuralNetworkInstance
{
	private readonly int m_BatchSize;
	private readonly int m_LayerCount;
	private readonly int m_InputNodeCount;
	private readonly int m_OutputNodeCount;

	private readonly float[][,] m_Weights;
	private readonly float[][] m_Biases;
	private readonly ILayerInstance[] m_Layers;
	private readonly IActivationInstance[] m_Activations;
	private readonly IActivationLossInstance m_FinalActivationLoss;

	public CategoricalNeuralNetworkInstance(int batchSize, params (float[,] Weight, float[] Biases)[] layers)
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

		m_BatchSize = batchSize;
		m_LayerCount = layers.Length;
		m_InputNodeCount = layers[0].Weight.GetLength(1);
		m_OutputNodeCount = layers[^1].Weight.GetLength(0);

		m_Weights = new float[m_LayerCount][,];
		m_Biases = new float[m_LayerCount][];
		m_Layers = new ILayerInstance[m_LayerCount];
		m_Activations = new IActivationInstance[m_LayerCount];
		m_FinalActivationLoss = new ActivationLossSoftMaxCategoricalCrossEntropyInstance(batchSize, m_OutputNodeCount);

		for (int i = 0; i < m_LayerCount; ++i)
		{
			var inputSize = layers[i].Weight.GetLength(1);
			var outputSize = layers[i].Weight.GetLength(0);

			m_Weights[i] = layers[i].Weight;
			m_Biases[i] = layers[i].Biases;
			m_Layers[i] = new LayerDenseInstance(batchSize, inputSize, outputSize, m_Weights[i], m_Biases[i]);
			m_Activations[i] = (i == m_LayerCount - 1)
				? new ActivationSoftMaxInstance(batchSize, outputSize)
				: new ActivationReLUInstance(batchSize, outputSize);
		}
	}

	public ICategoricalNeuralNetworkInstance CopyWithBatchSize(int batchSize)
	{
		var wbs = new (float[,] Weight, float[] Biases)[m_LayerCount];
		for (int i = 0; i < m_LayerCount; ++i)
		{
			wbs[i] = (m_Weights[i].AsSpan2D().ToArray(), m_Biases[i].AsSpan().ToArray());
		}

		return new CategoricalNeuralNetworkInstance(batchSize, wbs);
	}

	public ReadOnlyMemory2D<float> Run(ReadOnlySpan2D<float> inputs)
	{
		Validate.ArraySize(inputs, m_BatchSize, m_InputNodeCount);

		var a = m_Layers[0].Forward(inputs);

		for (int i = 0; i < m_LayerCount - 1; ++i)
		{
			a = m_Activations[i].Forward(a.Span);
			a = m_Layers[i + 1].Forward(a.Span);
		}

		a = m_Activations[^1].Forward(a.Span);

		return a;
	}

	public void Train(ReadOnlySpan2D<float> inputs, ReadOnlySpan<int> expected, float learningRate, out float avgLoss, out float accuracy)
	{
		Validate.ArraySize(inputs, m_BatchSize, m_InputNodeCount);
		Validate.ArraySize(expected, inputs.Height);

		var a = m_Layers[0].Forward(inputs);

		for (int i = 0; i < m_LayerCount - 1; ++i)
		{
			a = m_Activations[i].Forward(a.Span);
			a = m_Layers[i + 1].Forward(a.Span);
		}

		var loss = m_FinalActivationLoss.Forward(a.Span, expected);

		var b = m_FinalActivationLoss.Backward(expected);

		for (int i = m_LayerCount - 1; i >= 1; --i)
		{
			b = m_Layers[i].Backward(m_Activations[i - 1].Outputs.Span, b.Span);
			b = m_Activations[i - 1].Backward(m_Layers[i - 1].Outputs.Span, b.Span);
		}

		m_Layers[0].Backward(inputs, b.Span);

		{
			avgLoss = 0.0f;

			var iMax = loss.Length;
			if (iMax > 0)
			{
				var row = loss.Span;
				for (int i = 0; i < iMax; ++i)
				{
					avgLoss += row[i];
				}

				avgLoss /= iMax;
			}
		}
		accuracy = AccuracyCategorical.Compute(m_FinalActivationLoss.ActivationOutput.Span, expected);

		for (int i = 0; i < m_LayerCount; ++i)
		{
			var dw = m_Layers[i].DWeights.Span;
			var db = m_Layers[i].DBiases.Span;

			for (int y = 0; y < m_Weights[i].GetLength(0); ++y)
				for (int x = 0; x < m_Weights[i].GetLength(1); ++x)
					m_Weights[i][y, x] -= learningRate * dw[y, x];
			for (int x = 0; x < m_Biases[i].Length; ++x)
				m_Biases[i][x] -= learningRate * db[x];
		}
	}
}
