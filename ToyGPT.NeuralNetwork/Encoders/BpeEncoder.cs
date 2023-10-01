using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

namespace ToyGPT.NeuralNetwork.Encoders;

/// <summary>
/// Byte Pair Encoding from GPT2
/// </summary>
/// <remarks>
/// At a high level, encoding works following these steps:
/// <list type="number">
/// <item>Split the text into "words" using a Regex, see <see cref="MakePat"/>.</item>
/// <item>For each word:
/// <list type="number">
/// <item>Convert each word to UTF8.</item>
/// <item>Encode the UTF8 bytes to an encoding I'm calling "minicode" (see <see cref="BytesToMinicode"/>) for some reason (GPT2 does this but I'm still not sure exactly why).</item>
/// <item>Map each minicode char to a token using the provided encoding dictionary.</item>
/// <item>Merge token pairs using the provided bytePairs list, see <see cref="MergeTokenPairs(List{int})"/>.</item>
/// </list></item>
/// </list>
/// </remarks>
public partial class BpeEncoder
	: IEncoder
{
	private record struct RankedToken(int Token, int Rank);

	private readonly Regex m_Pat = MakePat();

	private readonly Dictionary<byte, int> m_ByteToToken = new();
	private readonly Dictionary<int, byte[]> m_TokenToBytes = new();

	private readonly Dictionary<(int, int), RankedToken> m_TokenPairs = new();

	private readonly Dictionary<string, int[]> m_Cache = new();

	public BpeEncoder(Dictionary<string, int> encoding, List<(string, string)> bytePairs)
	{
		var byteEncoder = BytesToMinicode();
		var byteDecoder = byteEncoder.ToDictionary(x => x.Value, x => x.Key);

		foreach (var (b, c) in byteEncoder)
		{
			m_ByteToToken[b] = encoding[c.ToString()];
		}

		foreach (var (enctext, token) in encoding)
		{
			var bytes = enctext.Select(a => byteDecoder[a]).ToArray();
			m_TokenToBytes[token] = bytes;
		}

		for (int i = 0, iMax = bytePairs.Count; i < iMax; i++)
		{
			var (str0, str1) = bytePairs[i];

			var b0 = encoding[str0];
			var b1 = encoding[str1];

			m_TokenPairs[(b0, b1)] = new RankedToken(encoding[$"{str0}{str1}"], i);
		}
	}

	public List<int> Encode(ReadOnlySpan<char> text)
	{
		var utf8Encoder = new Utf8Encoder();
		var mTokens = new List<int>();
		var tokens = new List<int>();

		foreach (var m in m_Pat.EnumerateMatches(text))
		{
			var mSpan = text.Slice(m.Index, m.Length);
			var mStr = new string(mSpan);

			if (m_Cache.TryGetValue(mStr, out var cached))
			{
				tokens.AddRange(cached);
				continue;
			}

			var bytes = utf8Encoder.GetBytes(mSpan);

			mTokens.Clear();
			foreach (var b in bytes)
			{
				mTokens.Add(m_ByteToToken[b]);
			}

			MergeTokenPairs(mTokens);

			m_Cache[mStr] = mTokens.ToArray();
			tokens.AddRange(mTokens);
		}

		return tokens;
	}

	public string Decode(IEnumerable<int> tokens)
	{
		var bytes = new List<byte>();

		foreach (var token in tokens)
		{
			foreach (var c in m_TokenToBytes[token])
			{
				bytes.Add(c);
			}
		}

		return Encoding.UTF8.GetString(CollectionsMarshal.AsSpan(bytes));
	}

	public void ClearCache()
	{
		m_Cache.Clear();
	}

	public void MergeTokenPairs(List<int> tokens)
	{
		while (tokens.Count > 1)
		{
			var bestPair = GetBestPair(tokens);
			if (!bestPair.HasValue)
				break;

			var (i, token) = bestPair.Value;
			tokens[i - 1] = token;
			tokens.RemoveAt(i);
		}
	}

	private (int Index, int Token)? GetBestPair(List<int> tokens)
	{
		if (tokens.Count <= 1)
			return null;

		int bestRank = int.MaxValue;
		(int Index, int Token)? bestPair = default;

		var p0 = tokens[0];

		for (int i = 1; i < tokens.Count; ++i)
		{
			var p1 = tokens[i];
			if (m_TokenPairs.TryGetValue((p0, p1), out var rt) && rt.Rank < bestRank)
			{
				bestRank = rt.Rank;
				bestPair = (i, rt.Token);
			}
			p0 = p1;
		}

		return bestPair;
	}

	private static Dictionary<byte, char> BytesToMinicode()
	{
		var b2u = new Dictionary<byte, char>();

		for (int c = '!'; c <= '~'; c++)
		{
			b2u[(byte)c] = (char)c;
		}
		for (int c = '¡'; c <= '¬'; c++)
		{
			b2u[(byte)c] = (char)c;
		}
		for (int c = '®'; c <= 'ÿ'; c++)
		{
			b2u[(byte)c] = (char)c;
		}
		var n = 0;
		for (int c = 0; c < 256; c++)
		{
			if (!b2u.ContainsKey((byte)c))
			{
				b2u[(byte)c] = (char)(256 + n);
				++n;
			}
		}

		return b2u;
	}

	[GeneratedRegex(@"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+", RegexOptions.Compiled)]
	private static partial Regex MakePat();

	private struct Utf8Encoder
	{
		private byte[] m_Bytes = new byte[8];

		public Utf8Encoder()
		{
		}

		public ReadOnlySpan<byte> GetBytes(ReadOnlySpan<char> text)
		{
			var n = Encoding.UTF8.GetByteCount(text);
			if (n > m_Bytes.Length)
			{
				m_Bytes = new byte[2 * m_Bytes.Length];
			}
			Encoding.UTF8.GetBytes(text, m_Bytes);
			return m_Bytes.AsSpan(0, n);
		}
	}

	private struct Utf8Decoder
	{
		private char[] m_Chars = new char[8];

		public Utf8Decoder()
		{
		}

		public ReadOnlySpan<char> GetChars(ReadOnlySpan<byte> bytes)
		{
			var n = Encoding.UTF8.GetCharCount(bytes);
			if (n > m_Chars.Length)
			{
				m_Chars = new char[2 * m_Chars.Length];
			}
			Encoding.UTF8.GetChars(bytes, m_Chars);
			return m_Chars.AsSpan(0, n);
		}
	}
}
