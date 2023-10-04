using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CommunityToolkit.HighPerformance;

namespace ToyGPT.Lib.Model
{
	public class CkptReader
	{
		private record struct Entry
		{
			public string Name { get; set; }
			public ulong DataType { get; set; }
			public ulong[] Shape { get; set; }
			public ulong ShardId { get; set; }
			public ulong DataOffset { get; set; }
			public ulong DataSize { get; set; }
			public uint DataChecksum { get; set; }
		}

		private readonly string m_BaseFileName;
		private readonly Dictionary<string, Entry> m_Entries = new();

		public CkptReader(string baseFileName)
		{
			m_BaseFileName = baseFileName;
			using var br = new BinaryReader(File.OpenRead($"{baseFileName}.index"));

			var sb = new StringBuilder();

			while (br.BaseStream.Position < br.BaseStream.Length)
			{
				var shared = ReadVarint(br);
				var nonShared = ReadVarint(br);
				var valueSize = ReadVarint(br);

				sb.Length = (int)shared;
				sb.Append(br.ReadChars((int)nonShared));
				var data = br.ReadBytes((int)valueSize);

				var name = sb.ToString();
				if (name != "")
				{
					if (name[0] == '\0')
					{
						break;
					}
					var entry = ReadEntry(new BinaryReader(new MemoryStream(data)));
					entry.Name = name;
					m_Entries.Add(entry.Name, entry);
				}
			}
		}

		public ReadOnlyMemory<float> LoadArray(string variableName)
		{
			var entry = m_Entries[variableName];

			if (entry.DataType != 1)
				throw new Exception("Not a float array");

			using var br = new BinaryReader(File.OpenRead($"{m_BaseFileName}.data-00000-of-00001"));
			br.BaseStream.Seek((long)entry.DataOffset, SeekOrigin.Begin);
			var data = br.ReadBytes((int)entry.DataSize);
			var result = new float[entry.Shape.Aggregate((a, b) => a * b)];
			Buffer.BlockCopy(data, 0, result, 0, data.Length);
			return result;
		}

		public ReadOnlyMemory2D<float> LoadMatrix(string variableName)
		{
			var entry = m_Entries[variableName];

			if (entry.DataType != 1)
				throw new Exception("Not a float array");

			using var br = new BinaryReader(File.OpenRead($"{m_BaseFileName}.data-00000-of-00001"));
			br.BaseStream.Seek((long)entry.DataOffset, SeekOrigin.Begin);
			var data = br.ReadBytes((int)entry.DataSize);
			var result = new float[entry.Shape[0], entry.Shape[1]];
			Buffer.BlockCopy(data, 0, result, 0, data.Length);
			return result;
		}

		public ReadOnlyMemory2D<float> LoadMatrixT(string variableName)
		{
			var matrix = LoadMatrix(variableName);
			var result = new float[matrix.Width, matrix.Height];
			for (int y = 0; y < matrix.Height; ++y)
			{
				for (int x = 0; x < matrix.Width; ++x)
				{
					result[x, y] = matrix.Span[y, x];
				}
			}
			return result;
		}

		private static Entry ReadEntry(BinaryReader br)
		{
			var entry = new Entry();
			while (br.BaseStream.Position < br.BaseStream.Length)
			{
				var value = ReadNextValue(br, out var field);

				switch (field)
				{
				case 1:
					entry.DataType = (ulong)value;
					break;
				case 2:
					entry.Shape = ReadShape(new BinaryReader(new MemoryStream((byte[])value)))
						.Where(a => a != 1) // squeeze
						.ToArray();
					break;
				case 3:
					entry.ShardId = (ulong)value;
					break;
				case 4:
					entry.DataOffset = (ulong)value;
					break;
				case 5:
					entry.DataSize = (ulong)value;
					break;
				case 6:
					entry.DataChecksum = (uint)value;
					break;
				default:
					break;
				}
			}
			return entry;
		}

		private static List<ulong> ReadShape(BinaryReader br)
		{
			var size = new List<ulong>();
			while (br.BaseStream.Position < br.BaseStream.Length)
			{
				var value = ReadNextValue(br, out var field);

				switch (field)
				{
				case 2:
					size.Add(ReadDim(new BinaryReader(new MemoryStream((byte[])value))));
					break;
				default:
					break;
				}
			}
			return size;
		}

		private static ulong ReadDim(BinaryReader br)
		{
			while (br.BaseStream.Position < br.BaseStream.Length)
			{
				var value = ReadNextValue(br, out var field);

				switch (field)
				{
				case 1:
					return (ulong)value;
				default:
					break;
				}
			}
			throw new Exception("No dim found");
		}

		private static object ReadNextValue(BinaryReader br, out ulong field)
		{
			var fieldAndType = ReadVarint(br);
			field = fieldAndType >> 3;
			var type = fieldAndType & 7;
			switch (type)
			{
			case 0:
				return ReadVarint(br);
			case 1:
				return br.ReadUInt64();
			case 2:
				var len = ReadVarint(br);
				return br.ReadBytes((int)len);
			case 5:
				return br.ReadUInt32();
			default:
				throw new Exception("Unknown type");
			}
		}

		private static ulong ReadVarint(BinaryReader br)
		{
			ulong result = 0;
			int shift = 0;

			while (true)
			{
				byte b = br.ReadByte();
				result |= (ulong)(b & 0x7f) << shift;
				if ((b & 0x80) == 0)
				{
					break;
				}
				shift += 7;
			}

			return result;
		}
	}
}
