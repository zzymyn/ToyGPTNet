using System.Collections.Generic;
using System.ComponentModel;
using System.Runtime.CompilerServices;

namespace ToyGPTNet.NeuralNetwork.UI;

internal abstract class NotifierBase
	: INotifyPropertyChanged
{
	public event PropertyChangedEventHandler? PropertyChanged;

	protected void OnPropertyChanged([CallerMemberName] string? prop = null)
	{
		PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(prop));
	}

	protected void OnPropertyChanged(PropertyChangedEventArgs args)
	{
		PropertyChanged?.Invoke(this, args);
	}

	protected bool SetField<T>(ref T? field, T value, [CallerMemberName] string? prop = null)
	{
		if (EqualityComparer<T>.Default.Equals(field, value))
			return false;
		field = value;
		OnPropertyChanged(prop);
		return true;
	}
}
