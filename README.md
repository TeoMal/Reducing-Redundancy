# Sparse Training Experiments

<!-- TODO: name the paper this reproduces/extends and link it. The repo is
     currently called "Achlioptas-Paper", which means nothing outside the lab. -->

How much of a training batch — and how much of a gradient — do you actually
need? Two independent sparsification axes, measured on CIFAR.

## 1. Sparsifying the *examples* (`cifar10.py`)

Three loss functions over the same ResNet18 / CIFAR-10 setup:

| Loss | Rule |
|---|---|
| `CELoss` | Standard cross-entropy over the full batch |
| `TopKLoss` | Backprop through only the top `k`% highest-loss images (default `k = 0.5`) |
| `AdaptiveKLoss` | Backprop through the smallest set of top-loss samples whose cumulative loss reaches ⅔ of the batch total |

Each loss reports the **effective number of examples** it used, so training runs
are logged with an `effective_ratio` alongside accuracy. `adaptive_log.csv`
holds one such run: the adaptive rule settles around **45–54% of examples per
epoch** over the first epochs while train accuracy climbs from 0.40 to 0.66.

Per-epoch metrics (`examples_used`, `effective_examples_used`,
`effective_ratio`, `train_accuracy`, `test_accuracy`, `train_loss`) are appended
to CSV as they are produced, so an interrupted run keeps its history.

## 2. Sparsifying the *gradients* (`Gradient_Experiments.ipynb`)

`sparsify_gradients(model, top_percent=0.1)` keeps only the largest-magnitude
gradients per parameter tensor and zeroes the rest, applied through a
convolutional encoder–decoder on 32×32 inputs.

## 3. Baseline (`EffNetv2.ipynb`)

EfficientNetV2-S (ImageNet-21k pretrained) fine-tuned on CIFAR-10/100, as a
reference point for the above.

## Running it

```bash
python cifar10.py
```

Notebooks run standalone; both download CIFAR on first use.

## Results

<!-- TODO: the headline comparison — accuracy at equal epochs for CE vs TopK vs
     Adaptive, plotted against effective_ratio — is what makes this repo worth
     reading. adaptive_log.csv has the data for one arm; add the CE baseline. -->

Requires PyTorch, torchvision, opencv-python, matplotlib.
