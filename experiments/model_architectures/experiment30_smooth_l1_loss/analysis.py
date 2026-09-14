import numpy as np
from matplotlib import pyplot as plt

def plot_smooth_l1(betas, x_range=(-0.1, 0.1)):

    def smooth_l1_loss(x, beta):
        # np.where wendet die Bedingung elementweise auf das Array an
        loss = np.where(np.abs(x) < beta,
                        0.5 * (x ** 2) / beta,  # MSE-Zone (quadratisch, gestaucht)
                        np.abs(x) - 0.5 * beta)  # MAE-Zone (linear, verschoben)
        return loss

    x = np.linspace(x_range[0], x_range[1], 1000)
    plt.figure(figsize=(10, 6))
    plt.plot(x, np.abs(x), 'k--', label='MAE (Referenz)', alpha=0.4)
    for beta in betas:
        y = smooth_l1_loss(x, beta)
        plt.plot(x, y, label=f'Smooth L1 (beta={beta})', linewidth=2.5)

        plt.scatter([-beta, beta], [smooth_l1_loss(-beta, beta), smooth_l1_loss(beta, beta)],
                    color='red', zorder=5, marker='x')

    plt.title('Smooth L1 Loss vs. MAE', fontsize=14)
    plt.xlabel('Fehler pro Sample $(y - \hat{y})$', fontsize=12)
    plt.ylabel('Loss-Wert', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(fontsize=11)

    plt.xlim(x_range[0], x_range[1])
    plt.ylim(0, max(x_range) * 1.1)

    plt.show()

# plot_smooth_l1(betas=[0.001, 0.005, 0.01])