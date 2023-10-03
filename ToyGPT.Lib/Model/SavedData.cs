using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.Serialization;
using System.Text;
using System.Threading.Tasks;
using System.Xml;
using CommunityToolkit.HighPerformance;

namespace ToyGPT.Lib.Model
{
	[DataContract]
	public class SavedData
	{
		[DataMember]
		public SavedTensor[] Tensors { get; set; } = null!;

		public ReadOnlyMemory<float> LoadArray(string name)
		{
			var tensor = Tensors.FirstOrDefault(t => t.Name == name) ?? throw new ArgumentException(name);

			if (tensor.Shape.Length != 1)
				throw new InvalidOperationException("Tensor must be 1D");

			return tensor.Data.AsMemory();
		}

		public ReadOnlyMemory2D<float> LoadMatrix(string name)
		{
			var tensor = Tensors.FirstOrDefault(t => t.Name == name) ?? throw new ArgumentException(name);

			if (tensor.Shape.Length != 2)
				throw new InvalidOperationException("Tensor must be 2D");

			return tensor.Data.AsMemory().AsMemory2D(tensor.Shape[0], tensor.Shape[1]);
		}

		public static void WriteBinary(string filePath, SavedData savedData)
		{
			var dcs = new DataContractSerializer(typeof(SavedData));
			using var fs = File.Create(filePath);
			using var writer = XmlDictionaryWriter.CreateBinaryWriter(fs);
			dcs.WriteObject(writer, savedData);
		}

		public static SavedData ReadBinary(string filePath)
		{
			var dcs = new DataContractSerializer(typeof(SavedData));
			using var fs = File.OpenRead(filePath);
			using var reader = XmlDictionaryReader.CreateBinaryReader(fs, XmlDictionaryReaderQuotas.Max);
			return (dcs.ReadObject(reader) as SavedData) ?? throw new FileLoadException();
		}
	}

	[DataContract]
	public class SavedTensor
	{
		[DataMember]
		public string Name { get; set; } = null!;

		[DataMember]
		public int[] Shape { get; set; } = null!;

		[DataMember]
		public float[] Data { get; set; } = null!;
	}
}
