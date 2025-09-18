import numpy as np
from PIL import Image
import gudhi as gd
from pathlib import Path
import matplotlib.pyplot as plt
import os
from time import time

def calculate_persistence(data):
    ##############
    # TDA part
    ##############
    # Build cubical complex from image intensity values
    cc = gd.CubicalComplex(dimensions=data.shape, top_dimensional_cells=data.flatten())
    return cc.persistence()

def plot_persistence_diagram(persistence):
    ##############
    # Persistence diagram
    ##############
    diag_points = np.array([[p[0] ,p[1][0], p[1][1]] for p in persistence if p[1][1] != float('inf')])
    plt.figure(figsize=(6,6))
    plt.scatter(diag_points[:,1], diag_points[:,2], c=diag_points[:,0], cmap='tab10', s=5)
    plt.plot([0,1],[0,1], 'k--')
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.xlabel("Birth")
    plt.ylabel("Death")
    plt.title("Persistence Diagram")
    plt.colorbar(label="Homology Dimension")
    plt.show()
    

def calculate_betti_numbers(persistence):
    ##############
    # Betti numbers
    ##############
    diag_points = np.array([[p[0] ,p[1][0], p[1][1]] for p in persistence if p[1][1] != float('inf')])

    H0_points = diag_points[diag_points[:,0] == 0][:,1:]
    H1_points = diag_points[diag_points[:,0] == 1][:,1:]
    # Betti numbers
    Betti_levels_H0 = []
    Betti_levels_H1 = []
    for level in np.linspace(0, 1, 20):
        temp0 = H0_points[(H0_points[:,0] <= level) & (H0_points[:,1] > level)]
        temp1 = H1_points[(H1_points[:,0] <= level) & (H1_points[:,1] > level)]
        Betti_levels_H0.append(len(temp0))
        Betti_levels_H1.append(len(temp1))

    # Combine both Betti levels in one array
    # H0 levels first index 0-19, then H1 levels from 20-39
    return np.array(Betti_levels_H0 + Betti_levels_H1)


if __name__ == "__main__":
    from skimage import data

    image = np.array(data.astronaut())
    start = time()
    persistence = calculate_persistence(image)
    betti_numbers = calculate_betti_numbers(persistence)
    end = time()
    print(f"Computation time: {end - start:.2f} seconds")
    print("Betti numbers:", betti_numbers)