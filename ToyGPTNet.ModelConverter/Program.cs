using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.Serialization;
using System.Xml;
using NumSharp;
using Tensorflow;
using Tensorflow.Checkpoint;
using ToyGPT.Lib.Model;
namespace ToyGPTNet.ModelConverter;

internal class Program
{
	static void Main(string[] args)
	{
		var reader = new CheckpointReader("../../../../Data/124M/model.ckpt");

		var savedTensors = new List<SavedTensor>();

		foreach (var name in reader.VariableToShapeMap.Keys.OrderBy(a => a))
		{
			var tensor = reader.GetTensor(name);
			var value = new NDArray(tensor.ToArray<float>(), new NumSharp.Shape(tensor.shape.as_int_list()));

			value = np.squeeze(value);

			if (name.EndsWith("/w", StringComparison.InvariantCulture))
			{
				// transpose weight matrices because ToyGPT expects that:
				value = np.transpose(value);
			}

			savedTensors.Add(new SavedTensor
			{
				Name = name,
				Shape = value.shape,
				Data = value.ToArray<float>()
			});

			Console.WriteLine($"{name} = {value.shape}");
		}

		var savedData = new SavedData()
		{
			Tensors = savedTensors.ToArray()
		};

		SavedData.WriteBinary("../../../../Data/124M/model.bin", savedData);
	}
}