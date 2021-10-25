import pytest
from torch.utils.data import Dataset

from mlexp.utils.data import ReproducibleDataLoader


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


@pytest.mark.parametrize(
    "seed",
    [
        (2),
        (23),
        (329),
        (3047),
        (737903),
        (10000000),
    ],
)
def test_reproducible_dataloader(seed):
    test_x = [i for i in range(10)]
    test_y = [i + 1 for i in range(10)]
    test_dataset = SimpleDataset(test_x, test_y)

    data_loader = ReproducibleDataLoader(
        test_dataset,
        specified_seed=seed,
        batch_size=1,
        shuffle=True,
    )
    first_outputs = [p for p in data_loader]

    reproduced_data_loader = ReproducibleDataLoader(
        test_dataset,
        specified_seed=seed,
        batch_size=1,
        shuffle=True,
    )
    second_outputs = [p for p in reproduced_data_loader]

    for f, s in zip(first_outputs, second_outputs):
        assert f == s
