using System.CommandLine;
using System.Diagnostics;
using CommunityToolkit.HighPerformance;
using ToyGPT.Lib;
using ToyGPT.NeuralNetwork;
using Microsoft.Maui.Graphics;
using Microsoft.Maui.Graphics.Skia;
using ToyGPT.Lib.Model;
using ToyGPT.NeuralNetwork.Encoders;
using System.Text.Json.Nodes;
using ToyGPT.NeuralNetwork.Layers;
using System.Runtime.InteropServices;
using ToyGPT.NeuralNetwork.Activations;

namespace ToyGPT;

class Program
{
	private static async Task Main(string[] args)
	{
		var modelDirOption = new Option<DirectoryInfo>("--model-dir", "Directory containing the model files");

		var rootCommand = new RootCommand
		{
			modelDirOption,
		};

		rootCommand.SetHandler(async context =>
		{
			var ct = context.GetCancellationToken();
			var console = context.Console;
			var modelDir = context.ParseResult.GetValueForOption(modelDirOption) ?? new DirectoryInfo(Environment.CurrentDirectory);

			Run(modelDir, ct);
		});

		await rootCommand.InvokeAsync(args);
	}

	private static void Run(DirectoryInfo modelDir, CancellationToken ct)
	{
		Console.WriteLine("Loading...");
		var mr = new CkptReader(Path.Join(modelDir.FullName, "model.ckpt"));
		var hParams = HParams.ReadJson(Path.Join(modelDir.FullName, "hparams.json"));
		var encoder = LoadEncoder(modelDir);

		var random = new Random();

		var wte = mr.LoadMatrix("model/wte");
		var wpe = mr.LoadMatrix("model/wpe");
		var ln_f_b = mr.LoadArray($"model/ln_f/b");
		var ln_f_g = mr.LoadArray($"model/ln_f/g");

		var tokenEmbedding = new LayerPositionalTokenEmbedding(wte, wpe);
		var layers = new List<TransformerBlock>();

		for (int i = 0; i < hParams.n_layer; ++i)
		{
			var attn_c_attn_b = mr.LoadArray($"model/h{i}/attn/c_attn/b");
			var attn_c_attn_w = mr.LoadMatrixT($"model/h{i}/attn/c_attn/w");
			var attn_c_proj_b = mr.LoadArray($"model/h{i}/attn/c_proj/b");
			var attn_c_proj_w = mr.LoadMatrixT($"model/h{i}/attn/c_proj/w");
			var attn_ln_1_b = mr.LoadArray($"model/h{i}/ln_1/b");
			var attn_ln_1_g = mr.LoadArray($"model/h{i}/ln_1/g");
			var attn_ln_2_b = mr.LoadArray($"model/h{i}/ln_2/b");
			var attn_ln_2_g = mr.LoadArray($"model/h{i}/ln_2/g");
			var mlp_c_fc_b = mr.LoadArray($"model/h{i}/mlp/c_fc/b");
			var mlp_c_fc_w = mr.LoadMatrixT($"model/h{i}/mlp/c_fc/w");
			var mlp_c_proj_b = mr.LoadArray($"model/h{i}/mlp/c_proj/b");
			var mlp_c_proj_w = mr.LoadMatrixT($"model/h{i}/mlp/c_proj/w");

			layers.Add(new TransformerBlock(
				new(attn_ln_1_g, attn_ln_1_b),
				new(new(attn_c_attn_w, attn_c_attn_b), new(hParams.n_head), new(attn_c_proj_w, attn_c_proj_b)),
				new(attn_ln_2_g, attn_ln_2_b),
				new(new(mlp_c_fc_w, mlp_c_fc_b), new(mlp_c_proj_w, mlp_c_proj_b))));
		}

		var layerNorm = new LayerNormalization(ln_f_g, ln_f_b);
		var finalOutput = new LinearWeightsTransposed(wte);

		Console.WriteLine("Ready!");

		while (!ct.IsCancellationRequested)
		{
			Console.Write("Prompt: ");
			var text = Console.ReadLine();

			if (text == null)
				break;

			text = text.TrimEnd();

			Console.Write(text);
			var tokens = encoder.Encode(text);

			while (tokens.Count < hParams.n_ctx)
			{
				if (ct.IsCancellationRequested)
					break;

				var x = tokenEmbedding.Forward(CollectionsMarshal.AsSpan(tokens));
				foreach (var layer in layers)
				{
					x = layer.Forward(x.Span);
				}
				x = layerNorm.Forward(x.Span);
				x = finalOutput.Forward(x);

				// greedy sampling:
				var lastRow = x.Span.GetRowSpan(x.Height - 1);

				var options = lastRow.ToArray()
					.Select((a, i) => (a, i))
					.OrderByDescending(a => a.a)
					.Take(3).ToList();

				var bestToken = options[random.Next(options.Count)].i;

				if (bestToken >= hParams.n_vocab - 1)
				{
					break;
				}

				var nextText = encoder.Decode(new[] { bestToken });
				Console.Write(nextText);
				tokens.Add(bestToken);
			}
		}
	}

	private static BpeEncoder LoadEncoder(DirectoryInfo modelDir)
	{
		var encoderData = JsonNode.Parse(File.ReadAllText(Path.Join(modelDir.FullName, "encoder.json"))) as JsonObject;
		var encoding = encoderData!.ToDictionary(a => a.Key, a => (int)a.Value!);

		var vocabData = File.ReadAllLines(Path.Join(modelDir.FullName, "vocab.bpe"));
		var bpes = vocabData
			.Where(a => !a.StartsWith("#"))
			.Select(a => a.Split(" "))
			.Select(a => (a[0], a[1]))
			.ToList();

		var enc = new BpeEncoder(encoding, bpes);
		return enc;
	}
}
