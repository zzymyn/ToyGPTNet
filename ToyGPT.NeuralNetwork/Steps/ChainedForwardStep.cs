using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CommunityToolkit.HighPerformance;

namespace ToyGPT.NeuralNetwork.Steps;

public static class ChainedForwardStep
{
	public static ChainedForwardStep<T0, T1> Create<T0, T1>(T0 v0, T1 v1)
		where T0 : INeuralNetworkForwardStep
		where T1 : INeuralNetworkForwardStep
	{
		return new(v0, v1);
	}
	public static ChainedForwardStep<T0, T1, T2> Create<T0, T1, T2>(T0 v0, T1 v1, T2 v2)
		where T0 : INeuralNetworkForwardStep
		where T1 : INeuralNetworkForwardStep
		where T2 : INeuralNetworkForwardStep
	{
		return new(v0, v1, v2);
	}
	public static ChainedForwardStep<T0, T1, T2, T3> Create<T0, T1, T2, T3>(T0 v0, T1 v1, T2 v2, T3 v3)
		where T0 : INeuralNetworkForwardStep
		where T1 : INeuralNetworkForwardStep
		where T2 : INeuralNetworkForwardStep
		where T3 : INeuralNetworkForwardStep
	{
		return new(v0, v1, v2, v3);
	}
	public static ChainedForwardStep<T0, T1, T2, T3, T4> Create<T0, T1, T2, T3, T4>(T0 v0, T1 v1, T2 v2, T3 v3, T4 v4)
		where T0 : INeuralNetworkForwardStep
		where T1 : INeuralNetworkForwardStep
		where T2 : INeuralNetworkForwardStep
		where T3 : INeuralNetworkForwardStep
		where T4 : INeuralNetworkForwardStep
	{
		return new(v0, v1, v2, v3, v4);
	}
	public static ChainedForwardStep<T0, T1, T2, T3, T4, T5> Create<T0, T1, T2, T3, T4, T5>(T0 v0, T1 v1, T2 v2, T3 v3, T4 v4, T5 v5)
		where T0 : INeuralNetworkForwardStep
		where T1 : INeuralNetworkForwardStep
		where T2 : INeuralNetworkForwardStep
		where T3 : INeuralNetworkForwardStep
		where T4 : INeuralNetworkForwardStep
		where T5 : INeuralNetworkForwardStep
	{
		return new(v0, v1, v2, v3, v4, v5);
	}
	public static ChainedForwardStep<T0, T1, T2, T3, T4, T5, T6> Create<T0, T1, T2, T3, T4, T5, T6>(T0 v0, T1 v1, T2 v2, T3 v3, T4 v4, T5 v5, T6 v6)
		where T0 : INeuralNetworkForwardStep
		where T1 : INeuralNetworkForwardStep
		where T2 : INeuralNetworkForwardStep
		where T3 : INeuralNetworkForwardStep
		where T4 : INeuralNetworkForwardStep
		where T5 : INeuralNetworkForwardStep
		where T6 : INeuralNetworkForwardStep
	{
		return new(v0, v1, v2, v3, v4, v5, v6);
	}
}


public class ChainedForwardStep<T0, T1>
	: INeuralNetworkForwardStep
	where T0 : INeuralNetworkForwardStep
	where T1 : INeuralNetworkForwardStep
{
	private readonly T0 m_V0;
	private readonly T1 m_V1;

	public ChainedForwardStep(T0 v0, T1 v1)
	{
		m_V0 = v0;
		m_V1 = v1;
	}

	public ReadOnlyMemory2D<float> Outputs => m_V1.Outputs;

	public ReadOnlyMemory2D<float> Forward(ReadOnlySpan2D<float> inputs)
	{
		var r0 = m_V0.Forward(inputs);
		var r1 = m_V1.Forward(r0.Span);
		return r1;
	}
}


public class ChainedForwardStep<T0, T1, T2>
	: INeuralNetworkForwardStep
	where T0 : INeuralNetworkForwardStep
	where T1 : INeuralNetworkForwardStep
	where T2 : INeuralNetworkForwardStep
{
	private readonly T0 m_V0;
	private readonly T1 m_V1;
	private readonly T2 m_V2;

	public ChainedForwardStep(T0 v0, T1 v1, T2 v2)
	{
		m_V0 = v0;
		m_V1 = v1;
		m_V2 = v2;
	}

	public ReadOnlyMemory2D<float> Outputs => m_V2.Outputs;

	public ReadOnlyMemory2D<float> Forward(ReadOnlySpan2D<float> inputs)
	{
		var r0 = m_V0.Forward(inputs);
		var r1 = m_V1.Forward(r0.Span);
		var r2 = m_V2.Forward(r1.Span);
		return r2;
	}
}


public class ChainedForwardStep<T0, T1, T2, T3>
	: INeuralNetworkForwardStep
	where T0 : INeuralNetworkForwardStep
	where T1 : INeuralNetworkForwardStep
	where T2 : INeuralNetworkForwardStep
	where T3 : INeuralNetworkForwardStep
{
	private readonly T0 m_V0;
	private readonly T1 m_V1;
	private readonly T2 m_V2;
	private readonly T3 m_V3;

	public ChainedForwardStep(T0 v0, T1 v1, T2 v2, T3 v3)
	{
		m_V0 = v0;
		m_V1 = v1;
		m_V2 = v2;
		m_V3 = v3;
	}

	public ReadOnlyMemory2D<float> Outputs => m_V3.Outputs;

	public ReadOnlyMemory2D<float> Forward(ReadOnlySpan2D<float> inputs)
	{
		var r0 = m_V0.Forward(inputs);
		var r1 = m_V1.Forward(r0.Span);
		var r2 = m_V2.Forward(r1.Span);
		var r3 = m_V3.Forward(r2.Span);
		return r3;
	}
}


public class ChainedForwardStep<T0, T1, T2, T3, T4>
	: INeuralNetworkForwardStep
	where T0 : INeuralNetworkForwardStep
	where T1 : INeuralNetworkForwardStep
	where T2 : INeuralNetworkForwardStep
	where T3 : INeuralNetworkForwardStep
	where T4 : INeuralNetworkForwardStep
{
	private readonly T0 m_V0;
	private readonly T1 m_V1;
	private readonly T2 m_V2;
	private readonly T3 m_V3;
	private readonly T4 m_V4;

	public ChainedForwardStep(T0 v0, T1 v1, T2 v2, T3 v3, T4 v4)
	{
		m_V0 = v0;
		m_V1 = v1;
		m_V2 = v2;
		m_V3 = v3;
		m_V4 = v4;
	}

	public ReadOnlyMemory2D<float> Outputs => m_V4.Outputs;

	public ReadOnlyMemory2D<float> Forward(ReadOnlySpan2D<float> inputs)
	{
		var r0 = m_V0.Forward(inputs);
		var r1 = m_V1.Forward(r0.Span);
		var r2 = m_V2.Forward(r1.Span);
		var r3 = m_V3.Forward(r2.Span);
		var r4 = m_V4.Forward(r3.Span);
		return r4;
	}
}


public class ChainedForwardStep<T0, T1, T2, T3, T4, T5>
	: INeuralNetworkForwardStep
	where T0 : INeuralNetworkForwardStep
	where T1 : INeuralNetworkForwardStep
	where T2 : INeuralNetworkForwardStep
	where T3 : INeuralNetworkForwardStep
	where T4 : INeuralNetworkForwardStep
	where T5 : INeuralNetworkForwardStep
{
	private readonly T0 m_V0;
	private readonly T1 m_V1;
	private readonly T2 m_V2;
	private readonly T3 m_V3;
	private readonly T4 m_V4;
	private readonly T5 m_V5;

	public ChainedForwardStep(T0 v0, T1 v1, T2 v2, T3 v3, T4 v4, T5 v5)
	{
		m_V0 = v0;
		m_V1 = v1;
		m_V2 = v2;
		m_V3 = v3;
		m_V4 = v4;
		m_V5 = v5;
	}

	public ReadOnlyMemory2D<float> Outputs => m_V5.Outputs;

	public ReadOnlyMemory2D<float> Forward(ReadOnlySpan2D<float> inputs)
	{
		var r0 = m_V0.Forward(inputs);
		var r1 = m_V1.Forward(r0.Span);
		var r2 = m_V2.Forward(r1.Span);
		var r3 = m_V3.Forward(r2.Span);
		var r4 = m_V4.Forward(r3.Span);
		var r5 = m_V5.Forward(r4.Span);
		return r5;
	}
}


public class ChainedForwardStep<T0, T1, T2, T3, T4, T5, T6>
	: INeuralNetworkForwardStep
	where T0 : INeuralNetworkForwardStep
	where T1 : INeuralNetworkForwardStep
	where T2 : INeuralNetworkForwardStep
	where T3 : INeuralNetworkForwardStep
	where T4 : INeuralNetworkForwardStep
	where T5 : INeuralNetworkForwardStep
	where T6 : INeuralNetworkForwardStep
{
	private readonly T0 m_V0;
	private readonly T1 m_V1;
	private readonly T2 m_V2;
	private readonly T3 m_V3;
	private readonly T4 m_V4;
	private readonly T5 m_V5;
	private readonly T6 m_V6;

	public ChainedForwardStep(T0 v0, T1 v1, T2 v2, T3 v3, T4 v4, T5 v5, T6 v6)
	{
		m_V0 = v0;
		m_V1 = v1;
		m_V2 = v2;
		m_V3 = v3;
		m_V4 = v4;
		m_V5 = v5;
		m_V6 = v6;
	}

	public ReadOnlyMemory2D<float> Outputs => m_V6.Outputs;

	public ReadOnlyMemory2D<float> Forward(ReadOnlySpan2D<float> inputs)
	{
		var r0 = m_V0.Forward(inputs);
		var r1 = m_V1.Forward(r0.Span);
		var r2 = m_V2.Forward(r1.Span);
		var r3 = m_V3.Forward(r2.Span);
		var r4 = m_V4.Forward(r3.Span);
		var r5 = m_V5.Forward(r4.Span);
		var r6 = m_V6.Forward(r5.Span);
		return r6;
	}
}

