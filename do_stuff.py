from dunno.network import SimpleModel, HeteroscedasticLoss
from dunno.dataset import get_dataset
from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch
import pdb


plt.switch_backend('TkAgg')
plt.style.use('seaborn')


def real(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    y = x**3
    hump = np.exp(-(x - 1)**2/(2*1))  # N(1, 1)
    var = np.ones(x.shape)*3**2 + 16**2*hump
    return y, var


def get_data():
    x_lim = (-4, 4)
    samples = 1024
    rng = np.random.RandomState(666)
    x_vec = rng.random_sample(samples)*(x_lim[1] - x_lim[0]) + x_lim[0]
    y_vec, y_noise = real(x_vec)
    y_vec += (y_noise**0.5)*rng.standard_normal(samples)
    return x_vec, y_vec


def train_model(model, ds):
    dataloader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=True)
    optim = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = HeteroscedasticLoss()
    for idx_e in range(256):
        loss_epoch = 0
        for idx_i, (x_batch, y_batch) in enumerate(dataloader):
            optim.zero_grad()
            y_hat_batch = model(x_batch)
            loss = criterion(y_hat_batch, y_batch)
            loss.backward()
            optim.step()
            loss_epoch += loss.item()
            # print(f'idx: {idx_i:3d} | loss: {loss.item():.2f}')
        loss_epoch /= len(dataloader)
        print(f'epoch; {idx_e:2d} | mean loss: {loss_epoch:.2f}')


if __name__ == '__main__':
    x_data, y_data = get_data()
    model = SimpleModel()
    ds = get_dataset()
    train_model(model, ds)
    x_vec = np.linspace(-6, 6, 101).astype(np.float32)
    x_batch = torch.tensor(x_vec).unsqueeze(1)
    with torch.no_grad():
        y_hat_batch = model(x_batch)
    y_hat_mean_vec = y_hat_batch[:, 0].numpy()
    y_hat_std_vec = (y_hat_batch[:, 1].exp().numpy())**0.5

    y_vec, y_var_vec = real(x_vec)
    y_std_vec = y_var_vec**0.5

    fig, ax = plt.subplots()
    tmp = ax.plot(x_vec, y_vec, label='truth')
    color = tmp[0].get_color()
    ax.plot(x_vec, y_vec + y_std_vec, color=color, linestyle='--')
    ax.plot(x_vec, y_vec - y_std_vec, color=color, linestyle='--')
    # ax.scatter(x_data, y_data, color='red', marker='.', label='training data')
    # ax.plot(x_vec, y_hat_mean_vec, color='green', label='prediction')
    # ax.plot(
    #     x_vec, y_hat_mean_vec + y_hat_std_vec, color='green', linestyle='--'
    # )
    # ax.plot(
    #     x_vec, y_hat_mean_vec - y_hat_std_vec, color='green', linestyle='--'
    # )
    ax.set_xlabel('Some x')
    ax.set_ylabel('Some y')
    ax.legend()
    plt.show()
