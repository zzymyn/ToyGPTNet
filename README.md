# ToyGPTNEt

A neural networking library written in pure C# that doesn't use any Math or AI libraries, (ie no numpy, no tensorflow etc). At the moment it can run the GPT2 model on the CPU.

## Running

From the command-line you can run:

```
dotnet run --project ToyGPT -c Release
```

By default the 124M GPT2 model will be loaded from `Data/124M`, if the files don't exist, you will be asked if you want the files to be downloaded.