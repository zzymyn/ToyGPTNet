using System;
using CommunityToolkit.HighPerformance;

namespace ToyGPT.NeuralNetwork.ActivationLoss;

public interface IActivationLossInstance
	: IForwardActivationLossInstance
	, IBackwardActivationLossInstance
{
}