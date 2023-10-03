using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.Serialization;
using System.Runtime.Serialization.Json;
using System.Text;
using System.Threading.Tasks;

namespace ToyGPT.Lib.Model;

[DataContract]
public class HParams
{
	[DataMember]
	public int n_vocab { get; set; }

	[DataMember]
	public int n_ctx { get; set; }

	[DataMember]
	public int n_embd { get; set; }

	[DataMember]
	public int n_head { get; set; }

	[DataMember]
	public int n_layer { get; set; }

	public static HParams ReadJson(string filePath)
	{
		var dcs = new DataContractJsonSerializer(typeof(HParams));
		using var fs = File.OpenRead(filePath);
		return (dcs.ReadObject(fs) as HParams) ?? throw new FileLoadException();
	}
}
