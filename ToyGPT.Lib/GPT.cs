using System.CommandLine;
using System.Runtime.CompilerServices;
using System.Threading.Channels;
using System.Linq;

namespace ToyGPT.Lib;

public class GPT
{
	private readonly IConsole? m_Console;
	private readonly CancellationToken m_Ct;

	public GPT(IConsole? console, CancellationToken ct)
	{
		m_Console = console;
		m_Ct = ct;
	}

	public async Task LoadFile(FileInfo file)
	{
		using var sr = file.OpenText();

		var tokenizer = new CharacterLevelTokenizer(m_Ct);
		await tokenizer.Load(ReadFile(file));

		var tensor = await tokenizer.Encode(ReadFile(file)).ToArrayAsync(m_Ct);

		var trainLen = tensor.Length * 90 / 100;
		
		var trainingData = (ReadOnlyMemory<int>)tensor.AsMemory(0..trainLen);
		var validationData = (ReadOnlyMemory<int>)tensor.AsMemory(trainLen..);

	}

	private async IAsyncEnumerable<char> ReadFile(FileInfo file)
	{
		using var sr = file.OpenText();
		while (true)
		{
			var buffer = new char[1024];
			var bytesRead = await sr.ReadAsync(buffer.AsMemory(), m_Ct);

			if (bytesRead <= 0)
				break;
			for (int i = 0; i < bytesRead; ++i)
			{
				yield return buffer[i];
			}
		}
	}
}