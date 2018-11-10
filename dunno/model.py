from dunno.network import SimpleModel, HeteroscedasticLoss
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import pdb


class Model:
    """Neural-network-based model with utility methods."""

    def __init__(self):
        self.net = SimpleModel()

    def fit(self, dataset: Dataset) -> None:
        """Fit model to dataset.

        Parameters
        ----------
        dataset
            Training dataset.

        """
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=32, shuffle=True
        )
        optim = torch.optim.Adam(self.net.parameters(), lr=0.01)
        criterion = HeteroscedasticLoss()
        self.net.train()
        for idx_e in range(256):
            loss_epoch = 0
            for idx_i, (x_batch, y_batch) in enumerate(dataloader):
                optim.zero_grad()
                y_hat_batch = self.net(x_batch)
                loss = criterion(y_hat_batch, y_batch)
                loss.backward()
                optim.step()
                loss_epoch += loss.item()
                # print(f'idx: {idx_i:3d} | loss: {loss.item():.2f}')
            loss_epoch /= len(dataloader)
            print(f'epoch; {idx_e:2d} | mean loss: {loss_epoch:.2f}')

    def predict_on_batch(self, x_batch) -> np.ndarray:
        """Predict with model."""
        x_batch = torch.tensor(x_batch)
        if len(x_batch.size()) == 1:
            pass
        pdb.set_trace()
        self.net.eval()
