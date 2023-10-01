using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json.Nodes;
using System.Threading.Tasks;
using NUnit.Framework;
using ToyGPT.NeuralNetwork.Encoders;

namespace ToyGPT.NeuralNetwork.Tests;

public class BpeEncoderTests
{
	[Test]
	[TestCase("", new int[0])]
	[TestCase("Not all heroes wear capes.", new[] { 3673, 477, 10281, 5806, 1451, 274, 13 })]
	[TestCase("aslk;duhygfapoiuwsenv;auhedsf", new[] { 292, 75, 74, 26, 646, 12114, 70, 69, 41817, 16115, 86, 6248, 85, 26, 559, 704, 28202 })]
	[TestCase("こんにちは", new[] { 46036, 22174, 28618, 2515, 94, 31676 })]
	[TestCase("مرحبًا", new[] { 25405, 26897, 148, 255, 39848, 149, 233, 12919 })]
	public void TestEncoder(string input, int[] output)
	{
		var enc = LoadEncoder();
		var encoded = enc.Encode(input);
		Assert.AreEqual(output, encoded);
	}

	[Test]
	[TestCase(new[] { 3673, 477, 10281, 5806, 1451, 274, 13 }, "Not all heroes wear capes.")]
	[TestCase(new[] { 292, 75, 74, 26, 646, 12114, 70, 69, 41817, 16115, 86, 6248, 85, 26, 559, 704, 28202 }, "aslk;duhygfapoiuwsenv;auhedsf")]
	[TestCase(new[] { 46036, 22174, 28618, 2515, 94, 31676 }, "こんにちは")]
	public void TestDecoder(int[] input, string output)
	{
		var enc = LoadEncoder();
		var decoded = enc.Decode(input);
		Assert.AreEqual(output, decoded);
	}

	private static BpeEncoder LoadEncoder()
	{
		var encoderData = JsonNode.Parse(File.ReadAllText("../../../../Data/124M.encoder.json")) as JsonObject;
		var encoding = encoderData!.ToDictionary(a => a.Key, a => (int)a.Value!);

		var vocabData = File.ReadAllLines("../../../../Data/124M.vocab.bpe");
		var bpes = vocabData
			.Where(a => !a.StartsWith("#"))
			.Select(a => a.Split(" "))
			.Select(a => (a[0], a[1]))
			.ToList();

		var enc = new BpeEncoder(encoding, bpes);
		return enc;
	}
}
