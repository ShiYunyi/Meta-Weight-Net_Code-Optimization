# Meta-Weight-Net_Code-Optimization
A new code framework that uses pytorch to implement meta-learning, and takes Meta-Weight-Net as an example.

---

By using a trick, meta-learning and meta-networks have become plug-and-play. We can now apply the meta learning
algorithm directly to the existing pytorch model without rewriting it. 

This code takes Meta-Weight-Net ([Meta-Weight-Net: Learning an Explicit Mapping For Sample Weighting](https://arxiv.org/abs/1902.07379))
as an example to show how to use this trick. It rewrites an optimizer to assign non leaf node tensors to model parameters.
See `meta.py` and line 90-120 of `main.py` for details.

---
##Environment
- python 3.8
- pytorch 1.9.0
- torchvision 0.10.0

`noisy_long_tail_CIFAR.py` can generate noisy and long-tailed CIFAR datasets by calling `torchvision.datasets`. Because 
some class attributes have been changed, errors may occur in some earlier versions of torchvision. It can be solved by
changing the corresponding attribute name.
---
##Running this example
ResNet32 on CIFAR10-LT with imbalanced factor of 50:
```
python main.py --imbalanced_factor 50
```
ResNet32 on CIFAR10 with 40% uniform noise:
```
python main.py --meta_lr 1e-3 --meta_weight_decay 1e-4 --corruption_type uniform --corruption_ratio 0.4
```
---
##Acknowledgements
Thanks to the original code of Meta-Weight-Net (https://github.com/xjtushujun/meta-weight-net).

Contact: Shi Yunyi (2404208668@qq.com)
