using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CommunityToolkit.HighPerformance;

namespace ToyGPT.NeuralNetwork
{
	public class ActivationLossSoftMaxCategoricalCrossEntropy
		: IActivationLossCategorical
	{
		private readonly ActivationSoftMax m_SoftMax = new();
		private readonly LossCategoricalCrossEntropy m_Loss = new();

		public void Forward(ReadOnlySpan2D<float> inputs, ReadOnlySpan<int> categories, Span2D<float> outputs, Span<float> losses)
		{
			m_SoftMax.Forward(inputs, outputs);
			m_Loss.Forward(outputs, categories, losses);
		}

		public void Backward(ReadOnlySpan<int> categories, ReadOnlySpan2D<float> dValues, Span2D<float> dInputs)
		{
			Validate.ArraysSameSize(dInputs, dValues);
			Validate.ArraySize(categories, dInputs.Height);

			var yMax = dInputs.Height;
			var xMax = dInputs.Width;
			for (int y = 0; y < yMax; ++y)
			{
				var rowDVal = dValues.GetRowSpan(y);
				var rowDIn = dInputs.GetRowSpan(y);
				var category = categories[y];

				if (category < 0 || category > xMax)
					throw new ArgumentException(null, nameof(categories));

				for (int x = 0; x < xMax; ++x)
				{
					var a = rowDVal[x];
					if (x == category)
						a -= 1.0f;
					rowDIn[x] = a / yMax;
				}
			}
		}
	}
}
