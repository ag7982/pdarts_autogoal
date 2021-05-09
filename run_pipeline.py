from torchvision.datasets import CIFAR10

from autogoal.grammar import generate_cfg

from pdarts_autogoal.pipeline import Pipeline


grammar = generate_cfg(Pipeline)

candidate = grammar.sample()

train_data = CIFAR10(
    root='./data', train=True, download=True
)

val_data = CIFAR10(
    root='./data', train=False, download=True
)

candidate.fit(train_data, val_data)