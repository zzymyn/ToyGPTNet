using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

namespace ToyGPT.NeuralNetwork.Encoders
{
	// Byte-Pair Encoding like GPT2
	public partial class BpeEncoder
		: IEncoder
	{
		private readonly Dictionary<string, int> m_Encoding;
		private readonly Dictionary<int, string> m_Decoding;
		private readonly Dictionary<byte, char> m_ByteEncoder;
		private readonly Dictionary<char, byte> m_ByteDecoder;
		private readonly Dictionary<(string, string), int> m_BpeRanks = new();
		private readonly Regex m_Pat = MakePat();
		private readonly Dictionary<string, string[]> m_Cache = new();

		public BpeEncoder(Dictionary<string, int> encoding, List<(string, string)> bpes)
		{
			m_Encoding = encoding;
			m_Decoding = encoding.ToDictionary(x => x.Value, x => x.Key);
			m_ByteEncoder = BytesToUnicode();
			m_ByteDecoder = m_ByteEncoder.ToDictionary(x => x.Value, x => x.Key);
			for (int i = 0, iMax = bpes.Count; i < iMax; i++)
			{
				m_BpeRanks[bpes[i]] = i;
			}
		}

		public List<int> Encode(ReadOnlySpan<char> text)
		{
			var bytes = new byte[8];
			var chars = new char[8];
			var tokens = new List<int>();

			foreach (var m in m_Pat.EnumerateMatches(text))
			{
				var byteCount = ConvertBytes(text.Slice(m.Index, m.Length), ref bytes);

				if (bytes.Length > chars.Length)
				{
					chars = new char[bytes.Length];
				}

				for (int i = 0; i < byteCount; i++)
				{
					chars[i] = m_ByteEncoder[bytes[i]];
				}

				var a = new string(chars, 0, byteCount);

				foreach (var b in DoBpe(a))
				{
					tokens.Add(m_Encoding[b]);
				}
			}

			return tokens;
		}

		private int ConvertBytes(ReadOnlySpan<char> text, ref byte[] bytes)
		{
			var n = Encoding.UTF8.GetByteCount(text);
			if (n > bytes.Length)
			{
				bytes = new byte[2 * bytes.Length];
			}
			Encoding.UTF8.GetBytes(text, bytes);
			return n;
		}

		public string[] DoBpe(string text)
		{
			if (m_Cache.TryGetValue(text, out var cachedValue))
				return cachedValue;

			var word = text.Select(c => new string(c, 1)).ToList();

			while (word.Count > 1)
			{
				var bestPair = GetBestPair(word);
				if (!bestPair.HasValue)
					break;
				var (p0, p1) = bestPair.Value;

				for (int i = 1; i < word.Count; ++i)
				{
					if (word[i - 1] == p0 && word[i] == p1)
					{
						word[i - 1] = p0 + p1;
						word.RemoveAt(i);
						--i;
					}
				}
			}

			var wordArray = word.ToArray();
			m_Cache[text] = wordArray;
			return wordArray;
		}

		private (string, string)? GetBestPair(List<string> word)
		{
			if (word.Count <= 1)
				return null;

			int bestRank = int.MaxValue;
			(string, string)? bestPair = default;

			var p0 = word[0];

			for (int i = 1; i < word.Count; ++i)
			{
				var p1 = word[i];
				if (m_BpeRanks.TryGetValue((p0, p1), out var rank) && rank < bestRank)
				{
					bestRank = rank;
					bestPair = (p0, p1);
				}
				p0 = p1;
			}

			return bestPair;
		}

		private static Dictionary<byte, char> BytesToUnicode()
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
	}
}
