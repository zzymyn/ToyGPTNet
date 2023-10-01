using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ToyGPT.Lib;

internal interface ITokenizer
{
	IAsyncEnumerable<int> Encode(IAsyncEnumerable<char> input);
	IAsyncEnumerable<char> Decode(IAsyncEnumerable<int> input);
}
