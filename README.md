# ToyGPTNet

A neural networking library written in pure C# that doesn't use any Math or AI libraries, (ie no numpy, no tensorflow etc). At the moment it can run the GPT2 model on the CPU.

## Running

From the command-line you can run:

```
dotnet run --project ToyGPT -c Release
```

By default the 124M GPT2 model will be loaded from `Data/124M`, if the files don't exist, you will be asked if you want the files to be downloaded.

Note that due to only running on the CPU, the performance can get quite slow as the input/output gets longer.

## References

* [Neural Networks from Scratch in Python](https://www.youtube.com/playlist?list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3)
* [GPT in 60 Lines of NumPy](https://jaykmody.com/blog/gpt-from-scratch/)
* [Speeding up the GPT - KV cache](https://www.dipkumar.dev/becoming-the-unbeatable/posts/gpt-kvcache/)