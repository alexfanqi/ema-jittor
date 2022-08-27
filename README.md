## Jittor version of EMA converted from Pytorch
Purely for convenience of my project. I don't intend to
maintain this

## EMA - Pytorch

A simple way to keep track of an Exponential Moving Average (EMA) version of your pytorch model

## Install

```bash
$ pip install ema-pytorch
```

## Usage

```python
import jittor
from ema_jittor import EMA

# your neural network as a pytorch module

net = jittor.nn.Linear(1, 1)

# wrap your neural network, specify the decay (beta)

ema = EMA(
    net,
    beta = 0.9999,              # exponential moving average factor
    update_after_step = 5,    # only after this number of .update() calls will it start updating
    update_every = 1,          # how often to actually update, to save on compute (updates every 10th .update() call)
)

# mutate your network, with SGD or otherwise
for i in range(10):
    with jittor.no_grad():
        # net.weight.assign(jittor.randn_like(net.weight))
        net.bias.assign(net.bias+1)
    ema.update()
    print(ema.get_current_decay())

# then, later on, you can invoke the EMA model the same way as your network
data = jittor.Var(1)

output     = net(data)
ema_output = ema(data)
print(output)
print(ema_output)
# if you want to save your ema model, it is recommended you save the entire wrapper
# as it contains the number of steps taken (there is a warmup logic in there, recommended by @crowsonkb, validated for a number of projects now)
# however, if you wish to access the copy of your model with EMA, then it will live at ema.ema_model
```

## Todo

- [ ] address the issue of annealing EMA to 1 near the end of training for BYOL https://github.com/lucidrains/byol-pytorch/issues/82
