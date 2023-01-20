using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ToyGPT.Lib
{
	internal class CharacterLevelTokenizer
		: ITokenizer
	{
		private readonly CancellationToken m_Ct;
		private readonly Dictionary<int, char> m_Token2Char = new();
		private readonly Dictionary<char, int> m_Char2Token = new();

		public CharacterLevelTokenizer(CancellationToken ct)
		{
			m_Ct = ct;
		}

		public async Task Load(IAsyncEnumerable<char> input)
		{
			var set = new HashSet<char>();
			
			await foreach (var c in input.WithCancellation(m_Ct))
			{
				set.Add(c);
			}

			var inOrder = set.ToList();
			inOrder.Sort();

			for (int i = 0; i < inOrder.Count; ++i)
			{
				var c = inOrder[i];
				m_Char2Token[c] = i;
				m_Token2Char[i] = c;
			}
		}

		public async IAsyncEnumerable<int> Encode(IAsyncEnumerable<char> input)
		{
			await foreach (var c in input.WithCancellation(m_Ct))
			{
				yield return m_Char2Token[c];
			}
		}

		public async IAsyncEnumerable<char> Decode(IAsyncEnumerable<int> input)
		{
			await foreach (var c in input.WithCancellation(m_Ct))
			{
				yield return m_Token2Char[c];
			}
		}
	}
}
