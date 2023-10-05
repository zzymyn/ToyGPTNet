using System.CommandLine;
using System.CommandLine.IO;
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
using System.Net;
using System.Net.Http.Handlers;
using System.Text;

namespace ToyGPT;

class Program
{
	private const string BaseDownloadUrl = @"https://openaipublic.blob.core.windows.net/gpt-2/models";

	private static readonly List<string> Models = new()
	{
		"124M",
		"355M",
		"774M",
		"1558M",
	};

	private static readonly List<string> Files = new()
	{
		"checkpoint",
		"hparams.json",
		"model.ckpt.index",
		"model.ckpt.meta",
		"vocab.bpe",
		"encoder.json",
		"model.ckpt.data-00000-of-00001",
	};

	private static async Task Main(string[] args)
	{
		try
		{
			Console.OutputEncoding = Encoding.UTF8;
		}
		catch (Exception)
		{
		}

		var modelNameOption = new Option<string>(
			"--model",
			parse =>
			{
				if (parse.Tokens.Count == 0)
				{
					return Models[0];
				}
				var v = parse.Tokens[^1].Value;
				if (!Models.Contains(v))
				{
					throw new ArgumentException($"Invalid model name: {v}");
				}
				return v;
			},
			description: "Name of the model to use, one of: [124M, 355M, 774M, 1558M].",
			isDefault: true);

		var rootCommand = new RootCommand
		{
			modelNameOption,
		};

		rootCommand.Description = "ToyGPT - a pure C# implementation of GPT-2";

		rootCommand.SetHandler(async context =>
		{
			var ct = context.GetCancellationToken();

			var modelName = context.ParseResult.GetValueForOption(modelNameOption) ?? "124M";

			context.ExitCode = await Run(modelName, ct);
		});

		await rootCommand.InvokeAsync(args);
	}

	private static async Task<int> Run(string modelName, CancellationToken ct)
	{
		var modelDir = Path.Join("Data", modelName);

		// download files:
		bool? allowDownloads = null;
		bool filesMissing = false;
		foreach (var fileName in Files)
		{
			var filePath = Path.Join(modelDir, fileName);
			if (!File.Exists(filePath))
			{
				var url = $"{BaseDownloadUrl}/{modelName}/{fileName}";

				if (allowDownloads == null)
				{
					Console.WriteLine("The model files are missing. Do you want to download them? [y/n]");
					var key = Console.ReadKey();
					Console.WriteLine();
					allowDownloads = key.Key == ConsoleKey.Y;
				}
				if (allowDownloads == false)
				{
					Console.WriteLine($"Required file {filePath} is missing, download it from:\n{url}\n");
					filesMissing = true;
				}
				else
				{
					Console.WriteLine($"Downloading {url}...");
					await DownloadFile(url, filePath, ct);
				}
			}
		}

		if (filesMissing)
		{
			return 1;
		}

		Console.WriteLine("Loading...");
		var mr = new CkptReader(Path.Join(modelDir, "model.ckpt"));
		var hParams = HParams.ReadJson(Path.Join(modelDir, "hparams.json"));
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

			if (text == "")
				continue;

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

				// top-3 sampling:
				var options = x.Span.GetRowSpan(x.Height - 1).ToArray()
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

		return 0;
	}

	private static async Task DownloadFile(string url, string filePath, CancellationToken ct)
	{
		var dir = Path.GetDirectoryName(filePath);
		if (!string.IsNullOrEmpty(dir) && !Directory.Exists(dir))
		{
			Directory.CreateDirectory(dir);
		}

		var handler = new HttpClientHandler();
		var ph = new ProgressMessageHandler(handler);

		var clearLine = $"\r{new string(' ', 60)}\r";

		Console.Write($"{clearLine} {GenerateConsoleProgressBar(10, 0.0f)} Starting download...");

		var startTime = Stopwatch.StartNew();

		ph.HttpReceiveProgress += (_, e) =>
		{
			var aMb = (float)e.BytesTransferred / 1024 / 1024;
			var totalSeconds = startTime.Elapsed.TotalSeconds;
			var aMbPerSec = aMb / totalSeconds;

			if (e.TotalBytes != null)
			{
				var f = (float)e.BytesTransferred / e.TotalBytes.Value;
				var bMb = (float)e.TotalBytes.Value / 1024 / 1024;
				Console.Write($"{clearLine} {GenerateConsoleProgressBar(10, f)} {100 * f:0.0}% {aMb:0.0} MB of {bMb:0.0} MB ({aMbPerSec:0.0} MB/s)");
			}
			else
			{
				Console.Write($"{clearLine} {GenerateConsoleProgressBar(10, 0.0f)} ?% {aMb:0.0} MB of unknown ({aMbPerSec:0.0} MB/s)");
			}
		};

		var client = new HttpClient(ph);

		using (var s = await client.GetStreamAsync(url, ct))
		using (var fs = File.Create($"{filePath}.download"))
		{
			await s.CopyToAsync(fs, ct);
		}

		Console.Write($"{clearLine} done");
		Console.WriteLine();

		File.Move($"{filePath}.download", filePath);
	}

	private static string GenerateConsoleProgressBar(int width, float progress)
	{
		var chars = new char[width + 2];
		chars[0] = '▐';
		var progressChars = "▖▌▙█";

		for (int i = 0; i < width; ++i)
		{
			var t0 = (float)i / width;
			var t1 = (float)(i + 1) / width;
			var subT = (progress - t0) / (t1 - t0);

			if (subT <= 0.0f)
			{
				chars[i + 1] = ' ';
			}
			else
			{
				chars[i + 1] = progressChars[Math.Min((int)(subT * progressChars.Length), progressChars.Length - 1)];
			}
		}

		chars[^1] = '▌';

		return new string(chars);
	}

	private static BpeEncoder LoadEncoder(string modelDir)
	{
		var encoderData = JsonNode.Parse(File.ReadAllText(Path.Join(modelDir, "encoder.json"))) as JsonObject;
		var encoding = encoderData!.ToDictionary(a => a.Key, a => (int)a.Value!);

		var vocabData = File.ReadAllLines(Path.Join(modelDir, "vocab.bpe"));
		var bpes = vocabData
			.Where(a => !a.StartsWith("#"))
			.Select(a => a.Split(" "))
			.Select(a => (a[0], a[1]))
			.ToList();

		var enc = new BpeEncoder(encoding, bpes);
		return enc;
	}
}