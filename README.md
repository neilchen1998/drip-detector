# Drip Detector

## Packages

### tqdm

[tqdm](https://github.com/tqdm/tqdm) is a fast progress bar tool in CLI written in Python.
It shows progress bars and the overhead is low.

## Apple Silicon Acceleration

In order to utilize the GPU on Mac, install the following packages with **conda**:

```zsh
conda install pytorch torchvision torchaudio -c pytorch-nightly
```

You can verify Metal Performance Shader (**mps**) is supported by running this script:

```python
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print (x)
else:
    print ("MPS device not found.")
```

The output should show something like this:

```zsh
tensor([1.], device='mps:0')
```

## Reference

*[Accelerated PyTorch training on Mac](https://developer.apple.com/metal/pytorch/)