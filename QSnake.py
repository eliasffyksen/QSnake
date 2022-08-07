from statistics import mean
from this import d
import torch as t
import torch.nn as nn
import sys
import timeit

class ParalellSum(nn.Module):
  def __init__(
    self,
    pipe_1,
    pipe_2,
    activation
  ):
    super().__init__()
    self.pipe_1 = pipe_1
    self.pipe_2 = pipe_2
    self.activation = activation

  def forward(self, x):
    out  = self.pipe_1(x)
    out += self.pipe_2(x)
    return self.activation(out)

class ResNet2D(nn.Module):
  def __init__(
    self,
    in_channels,
    reductions,
    extra_blocks_per_reducion,
    activation
  ):
    super().__init__()
    layers = []
    identity = nn.Identity()

    for _ in range(reductions):
      new_channels = in_channels * 2
      layers.append(ParalellSum(
        # Regular connection
        nn.Sequential(
          nn.Conv2d(
            in_channels,
            new_channels,
            kernel_size=(3,3),
            stride=(2,2),
            padding=(1,1)
          ),
          nn.BatchNorm2d(new_channels),
          activation,
          nn.Conv2d(new_channels, new_channels, (3,3), padding='same'),
          nn.BatchNorm2d(new_channels),
        ),
        # Skip connection
        nn.Sequential(
          nn.Conv2d(
            in_channels,
            new_channels,
            kernel_size=(1,1),
            stride=(2,2)
          ),
          nn.BatchNorm2d(new_channels)
        ),
        # Activation
        activation
      ))

      in_channels = new_channels
      for _ in range(extra_blocks_per_reducion):
        layers.append(ParalellSum(
          nn.Sequential(
            nn.Conv2d(in_channels, in_channels, (3,3), padding='same'),
            nn.BatchNorm2d(in_channels),
            activation,
            nn.Conv2d(in_channels, in_channels, (3,3), padding='same'),
            nn.BatchNorm2d(in_channels),
          ),
          identity,
          activation
        ))
    self.model = nn.Sequential(*layers)

  def forward(self, x):
    return self.model(x)

class QSnake(nn.Module):
  def __init__(self):
    super().__init__()
    self.board_size = 8
    channels = 4
    reductions = 2
    relu = nn.ReLU()
    linear_size = channels * 2 ** reductions
    out_size = 3

    self.model = nn.Sequential(
      nn.Conv2d(1, channels, (1,1)),
      nn.BatchNorm2d(channels),
      relu,
      ResNet2D(4, reductions, 1, relu),
      nn.AvgPool2d((2,2)),
      nn.Flatten(1),
      nn.Linear(linear_size, linear_size),
      relu,
      nn.Linear(linear_size, out_size)
    )

  def forward(self, x):
    x = x[:,None,::]
    return self.model(x)

if __name__ == '__main__':
  device = sys.argv[1]
  batch_size = int(sys.argv[2])
  qsnake = QSnake()
  qsnake.to(device=device)

  print(qsnake)

  parameter_sum = 0
  for p in qsnake.parameters():
    parameter_sum += p.numel()

  print('Parameters:', parameter_sum)

  x = None
  def setup():
    global x, batch_size
    x = t.randn((batch_size, 8, 8), device=device)

  number=100
  time = mean(timeit.repeat(
    lambda: qsnake(x),
    setup,
    number=1,
    repeat=number
  ))
  print('Forward pass time:', time)

