using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Input;

namespace ToyGPTNet.NeuralNetwork.UI.ViewModel;

public class ActionCommand : ICommand
{
	private readonly Action? m_Action;
	private readonly INotifyPropertyChanged? m_CanExecuteObject;
	private readonly PropertyInfo? m_CanExecuteProperty;

	public event EventHandler? CanExecuteChanged;

	public ActionCommand()
	{
		m_Action = null;
		m_CanExecuteObject = null;
		m_CanExecuteProperty = null;
	}

	public ActionCommand(Action action)
	{
		m_Action = action ?? throw new ArgumentNullException(nameof(action));
		m_CanExecuteObject = null;
		m_CanExecuteProperty = null;
	}

	public ActionCommand(Action action, INotifyPropertyChanged canExecuteObject, string canExecuteProperty)
		: this(canExecuteObject, canExecuteProperty)
	{
		m_Action = action ?? throw new ArgumentNullException(nameof(action));
	}

	public ActionCommand(INotifyPropertyChanged canExecuteObject, string canExecuteProperty)
	{
		if (string.IsNullOrEmpty(canExecuteProperty))
			throw new ArgumentNullException(nameof(canExecuteProperty));

		m_Action = null;
		m_CanExecuteObject = canExecuteObject ?? throw new ArgumentNullException(nameof(canExecuteObject));
		m_CanExecuteProperty = m_CanExecuteObject.GetType().GetProperty(canExecuteProperty) ?? throw new ArgumentNullException(nameof(canExecuteProperty));

		if (m_CanExecuteProperty.PropertyType != typeof(bool))
			throw new InvalidOperationException();

		PropertyChangedEventManager.AddHandler(m_CanExecuteObject, CanExecuteObject_PropertyChanged, m_CanExecuteProperty?.Name);
	}

	private void CanExecuteObject_PropertyChanged(object? sender, PropertyChangedEventArgs e)
	{
		CanExecuteChanged?.Invoke(this, EventArgs.Empty);
	}

	public bool CanExecute(object? parameter)
	{
		var result = m_CanExecuteProperty?.GetValue(m_CanExecuteObject);
		if (result != null)
			return (bool)result;
		return m_Action != null;
	}

	public void Execute(object? parameter)
	{
		m_Action?.Invoke();
	}
}
