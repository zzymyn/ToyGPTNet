using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ToyGPT.NeuralNetwork.Encoders;

public class CharacterLevelEncoder
	: IEncoder
{
	private readonly Dictionary<int, char> m_Token2Char = new();
	private readonly Dictionary<char, int> m_Char2Token = new();

	public CharacterLevelEncoder()
	{
	}

	public void Load(ReadOnlySpan<char> input)
	{
		var set = new HashSet<char>();
		
		foreach (var c in input)
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

	public List<int> Encode(ReadOnlySpan<char> text)
	{
		var result = new List<int>(text.Length);

		foreach (var c in text)
		{
			result.Add(m_Char2Token[c]);
		}

		return result;
	}

	public string Decode(IEnumerable<int> tokens)
	{
		var sb = new StringBuilder();

		foreach (var token in tokens)
		{
			sb.Append(m_Token2Char[token]);
		}

		return sb.ToString();
	}
}
