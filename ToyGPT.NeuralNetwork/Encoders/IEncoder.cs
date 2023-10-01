using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ToyGPT.NeuralNetwork.Encoders
{
	public interface IEncoder
	{
		List<int> Encode(ReadOnlySpan<char> text);
		string Decode(IEnumerable<int> tokens);
		void ClearCache() { }
	}
}
