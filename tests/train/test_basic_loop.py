import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from mlexp.train.basic_loop import classification_loop


B_SIZE = 10


class SimpleDataset(Dataset):
    def __init__(self, x, y) -> None:
        super().__init__()
        self.x = x
        self.y = y

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        return x, y


class SimpleClassifier(torch.nn.Module):
    """(batchsize, 1) -> (batchsize, 2)"""

    def __init__(self):
        super(SimpleClassifier, self).__init__()
        self.internal_dim = 10
        self.output_dim = 2
        self.fc1 = torch.nn.Linear(1, self.internal_dim)
        self.relu = torch.nn.ReLU(inplace=True)
        self.out = torch.nn.Linear(self.internal_dim, self.output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.out(x)
        return x


def test_classification_loop():
    # tests that the loop finishes without any runtime error
    dev = torch.device("cpu")
    test_x = [[i] for i in range(100)]
    test_y = np.arange(100) % 2
    t_dataset = SimpleDataset(torch.Tensor(test_x[:80]), torch.Tensor(test_y[:80]).long())
    v_dataset = SimpleDataset(torch.Tensor(test_x[80:]), torch.Tensor(test_y[80:]).long())
    train_loader = DataLoader(t_dataset, batch_size=B_SIZE)
    test_loader = DataLoader(v_dataset, batch_size=B_SIZE)
    classifier = SimpleClassifier().to(dev)
    criterion = torch.nn.CrossEntropyLoss()
    opt = torch.optim.SGD(classifier.parameters(), lr=0.001)

    _, _ = classification_loop(
        device=dev,
        model=classifier,
        loss_fn=criterion,
        optimizer=opt,
        train_dataloader=train_loader,
        validation_dataloader=test_loader,
        enable_amp=False,
    )
