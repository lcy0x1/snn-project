# Readme

Install PyTorch:

With GPU
```
conda install -y pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
conda install -y matplotlib
conda install -y -c conda-forge snntorch
conda install -y tqdm
```

Or CPU Only
```
conda install -y pytorch torchvision torchaudio cpuonly -c pytorch
conda install -y matplotlib
conda install -y -c conda-forge snntorch
conda install -y tqdm
```