import matplotlib.pyplot as plt
import numpy as np
from util import *
from rbm import RestrictedBoltzmannMachine 
from dbn import DeepBeliefNet

def plot_convergence(errors, title="Convergence Plot"):
    plt.figure(figsize=(8, 5))
    plt.plot(errors, label="Reconstruction Error")
    plt.xlabel('Iterations')
    plt.ylabel('Error')
    plt.title(title)
    plt.legend()
    plt.show()

def plot_performance(hidden_units, errors, title="Performance With Different Hidden Units"):
    plt.figure(figsize=(8, 5))
    plt.plot(hidden_units, errors, marker='o', linestyle='-')
    plt.xlabel('Number of Hidden Units')
    plt.ylabel('Final Reconstruction Error')
    plt.title(title)
    plt.show()

def visualize_receptive_fields(weights, image_shape, grid=(5, 5), title="Receptive Fields"):
    fig, axes = plt.subplots(nrows=grid[0], ncols=grid[1], figsize=(grid[1] * 3, grid[0] * 3))
    for i, ax in enumerate(axes.flatten()):
        if i < weights.shape[1]:
            img = weights[:, i].reshape(image_shape)
            ax.imshow(img, cmap='gray')
            ax.axis('off')
    plt.suptitle(title)
    plt.show()

if __name__ == "__main__":
    image_size = [28,28]
    train_imgs,train_lbls,test_imgs,test_lbls = read_mnist(dim=image_size, n_train=60000, n_test=10000)

    errors = []
    hidden_units = [50, 100, 200, 300, 400]
    final_errors = []

    for h in hidden_units:
        rbm = RestrictedBoltzmannMachine(ndim_visible=image_size[0] * image_size[1],
                                         ndim_hidden=h,
                                         is_bottom=True,
                                         image_size=image_size,
                                         is_top=False,
                                         n_labels=10,
                                         batch_size=10)
        rbm.cd1(visible_trainset=train_imgs, n_iterations=10000)  
        final_errors.append(rbm.errors[-1])  
        
        visualize_receptive_fields(rbm.weight_vh, image_size, title=f"Receptive Fields with {h} Hidden Units")
        plot_convergence(rbm.errors, f"Convergence with {h} Hidden Units")
    plot_performance(hidden_units, final_errors, "Performance with Different Numbers of Hidden Units")