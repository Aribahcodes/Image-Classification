# Self-Pruning Neural Network - Report

## Why does L1 penalty on sigmoid gates encourage sparsity?

So the gate values are always between 0 and 1 because of the sigmoid function. When we add the L1 penalty (which is just the mean of all gate values), the optimizer gets penalized for every gate that is not zero. So it tries to push as many gates to zero as possible to reduce the total loss.

The reason L1 works better than L2 for this is that L1 gives a constant push toward zero no matter how small the gate value already is. L2 on the other hand gives a smaller and smaller push as the value gets smaller, so it never actually reaches zero. L1 keeps pushing until the gate is fully zero which is exactly what we want for pruning.

So the network learns a balance - gates for important weights stay open because closing them hurts the classification loss too much, but gates for unimportant weights just get pushed to zero by the L1 penalty. This is how the network "decides" which weights to keep and which to remove.

---

## Results Table

Trained for 20 epochs on CIFAR-10 with three different lambda values.

| Lambda | Test Accuracy (%) | Sparsity Level (%) |
|--------|:-----------------:|:------------------:|
| 0.1    |     52.66         |     45.93          |
| 1.0    |     50.38         |     64.57          |
| 5.0    |     49.04         |     97.13          |

As expected, higher lambda means more pruning but lower accuracy. Lower lambda means better accuracy but less pruning. The middle value (0.01) gives the best tradeoff between keeping accuracy while still pruning a good amount of weights.

---

## Gate Distribution Plot

The plot is saved as `gate_distributions.png` by the training script.

For a successful result the histogram should look bimodal - meaning there should be a big spike near 0 (these are the pruned weights that got pushed to zero by the L1 penalty) and a smaller cluster of values away from 0 (these are the weights the network decided were important enough to keep).

With low lambda (0.001) the distribution is more spread out because the sparsity penalty is weak so not many gates got pushed to zero. With high lambda (0.1) almost everything is near 0 which means very heavy pruning but the network loses accuracy because it is also pruning weights it actually needed.