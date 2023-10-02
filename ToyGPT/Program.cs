using System.CommandLine;
using System.Diagnostics;
using CommunityToolkit.HighPerformance;
using ToyGPT.Lib;
using ToyGPT.NeuralNetwork;
using Microsoft.Maui.Graphics;
using Microsoft.Maui.Graphics.Skia;

namespace ToyGPT;

class Program
{
	private static async Task Main(string[] args)
	{
		var rootCommand = new RootCommand
		{
		};

		rootCommand.SetHandler(async context =>
		{
			var ct = context.GetCancellationToken();
			var console = context.Console;

		});

		await rootCommand.InvokeAsync(args);
	}
}