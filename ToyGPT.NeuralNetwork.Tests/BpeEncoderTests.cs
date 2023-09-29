using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json.Nodes;
using System.Threading.Tasks;
using NUnit.Framework;
using ToyGPT.NeuralNetwork.Encoders;

namespace ToyGPT.NeuralNetwork.Tests
{
	public class BpeEncoderTests
	{
		[Test]
		public void TestBpe()
		{
			var encoderData = JsonNode.Parse(File.ReadAllText("TestData/encoder.json")) as JsonObject;
			var encoding = encoderData.ToDictionary(a => a.Key, a => (int)a.Value);

			var vocabData = File.ReadAllLines("TestData/vocab.bpe");
			var bpes = vocabData
				.Where(a => !a.StartsWith("#"))
				.Select(a => a.Split(" "))
				.Select(a => (a[0], a[1]))
				.ToList();

			var enc = new BpeEncoder(encoding, bpes);

			var tok = enc.Encode("Not all heroes wear capes.");
		}
	}
}
