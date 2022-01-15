# 3D Shape Modelling

> ### Practical Work

## 1. Voxel Carving

## 2. Neural Implicit Modelling

### 2.1 Programming Strategy

1. Draw the architecture of the MLP (input, output, layers, activation function).
   * Input: `X, Y, Z` (Voxel coordinates)
     * 3D mesh-grid using `np.mgrid`
   * Output: Occupancy (Voxel occupancy)
     * Defined using `np.ndarray`
     * Shape: `resolution * resolution * resolution/2`
   * Layers: 
        ```py
        self.layers = nn.Sequential(
            nn.Linear(3, 60),
            nn.Tanh(),
            nn.Linear(60, 120),
            nn.ReLU(),
            nn.Linear(120, 60),
            nn.ReLU(),
            nn.Linear(60, 30),
            nn.ReLU(), 
            nn.Linear(30, 1),
            #nn.Sigmoid()
        )
        ```
2. How are the training data (X, Y, Z, occupancy) formatted for the training ?
   * 3D mesh-grid using `np.mgrid`
        ```py
        X, Y, Z = np.mgrid[-1:1:step, -1:1:step, -0.5:0.5:step]  # Voxel coordinates
        ```
3. In the training function (`nif_train`), what is the loss function used ?
    ```py
    loss_function = nn.BCEWithLogitsLoss(pos_weight=p_weight)  # Sigmoid included in this loss function
    ```
4. Explain the normalization used to weight losses associated to inside and outside points in the training loss ?
   * We obtain how many voxels are occupied (`data_out=1`)
   * The weighing factor is calculated by a ratio of total size of `data_out` and the no. of occupied voxels
     * `data_out.size()[0] == (resolution^3)/2`
   * This factor is basically multiplied for the loss calculated for the +ve examples
   * See code below:
        ```py
        # Normalize cost between 0 and 1 in the grid
        n_one = (data_out == 1).sum()
        p_weight = (data_out.size()[0] - n_one)/n_one  # loss for positives will be multiplied by this factor in the loss function
        print("Pos. Weight: ", p_weight)
        ```
5. During the training how is the data organized into batches ?
   * For the `indices`, random permutations were made using `torch.randperm`
   * See code below:
        ```py
        # batch_size == 100
        # Iterate over batches
        for i in range(0, data_in.size()[0], batch_size):
            indices = permutation[i:i+batch_size]
            batch_x, batch_y = data_in[indices], data_out[indices]
            ...
        ```
6. What does the function `binary_acc` evaluate ? Is it used for the training ?
   * `binary_acc` does the Intersection Over Union (IOU) evaluation, which is fancy way of calculating the accuracy
   * Basically all the predicted values are compared with the test values, summed up (for those that are equal) and then divided by the total number of values of `y_test`
   * Before comparing, all the values are rounded to their nearest integer values.
7. How is the MLP used to generate a result to be visualized ?
   * MLP is used to obtain outputs from the given inputs `data_in `, which is essentially occupancy values only.
   * They're reshaped to dimension `(res * res * res/2)` to form a 3D grid-like which is finally used by the marching cube algorithm to transform the occupancy grid into a 3D mesh.
8. What is the memory size of the MLP ? how does it compare with: (i) A voxel occupancy grid; (ii) The original image set plus the calibration ?
   * If we choose a resolution of 100:
     * `data_in` size: `100*100*100/2, 3`
     * => Input to PyTorch MLP network: `(1, 500000, 3)`
   * From `torchsummary.summary()`:
     * Input size (MB): 5.72
     * Forward/backward pass size (MB): 2063.75
     * Params size (MB): 0.06
     * **Estimated Total Size (MB): 2069.54**
   * Voxel Occupancy grid:
     * TODO


### 2.2 Program Editing

#### 1.

In the main function, we make use of the function defined in `voxcarv3d.py`:

```py
# Generate X,Y,Z and occupancy
occupancy = compute_grid_vec(occupancy)
```

* `resolution` set to 100

Training Log:

```
Device:  cpu
Pos. Weight:  tensor(119.6855)
Starting epoch 1/10
Loss after mini-batch   500: 2.401
Loss after mini-batch  1000: 2.318
Loss after mini-batch  1500: 2.038
Loss after mini-batch  2000: 1.825
Loss after mini-batch  2500: 1.684
Loss after mini-batch  3000: 1.588
Loss after mini-batch  3500: 1.515
Loss after mini-batch  4000: 1.459
Loss after mini-batch  4500: 1.417
Loss after mini-batch  5000: 1.382
Binary accuracy:  tensor(98.)
Starting epoch 2/10
Loss after mini-batch   500: 1.048
Loss after mini-batch  1000: 1.060
Loss after mini-batch  1500: 1.063
Loss after mini-batch  2000: 1.059
Loss after mini-batch  2500: 1.057
Loss after mini-batch  3000: 1.063
Loss after mini-batch  3500: 1.063
Loss after mini-batch  4000: 1.064
Loss after mini-batch  4500: 1.063
Loss after mini-batch  5000: 1.064
Binary accuracy:  tensor(99.)
Starting epoch 3/10
Loss after mini-batch   500: 1.073
Loss after mini-batch  1000: 1.062
Loss after mini-batch  1500: 1.062
Loss after mini-batch  2000: 1.065
Loss after mini-batch  2500: 1.063
Loss after mini-batch  3000: 1.063
Loss after mini-batch  3500: 1.065
Loss after mini-batch  4000: 1.061
Loss after mini-batch  4500: 1.060
Loss after mini-batch  5000: 1.058
Binary accuracy:  tensor(97.)
Starting epoch 4/10
Loss after mini-batch   500: 1.054
Loss after mini-batch  1000: 1.052
Loss after mini-batch  1500: 1.060
Loss after mini-batch  2000: 1.057
Loss after mini-batch  2500: 1.056
Loss after mini-batch  3000: 1.058
Loss after mini-batch  3500: 1.058
Loss after mini-batch  4000: 1.057
Loss after mini-batch  4500: 1.056
Loss after mini-batch  5000: 1.055
Binary accuracy:  tensor(99.)
Starting epoch 5/10
Loss after mini-batch   500: 1.059
Loss after mini-batch  1000: 1.052
Loss after mini-batch  1500: 1.048
Loss after mini-batch  2000: 1.055
Loss after mini-batch  2500: 1.057
Loss after mini-batch  3000: 1.056
Loss after mini-batch  3500: 1.054
Loss after mini-batch  4000: 1.054
Loss after mini-batch  4500: 1.053
Loss after mini-batch  5000: 1.052
Binary accuracy:  tensor(99.)
Starting epoch 6/10
Loss after mini-batch   500: 1.039
Loss after mini-batch  1000: 1.037
Loss after mini-batch  1500: 1.037
Loss after mini-batch  2000: 1.044
Loss after mini-batch  2500: 1.053
Loss after mini-batch  3000: 1.052
Loss after mini-batch  3500: 1.051
Loss after mini-batch  4000: 1.050
Loss after mini-batch  4500: 1.050
Loss after mini-batch  5000: 1.049
Binary accuracy:  tensor(99.)
Starting epoch 7/10
Loss after mini-batch   500: 1.047
Loss after mini-batch  1000: 1.046
Loss after mini-batch  1500: 1.044
Loss after mini-batch  2000: 1.044
Loss after mini-batch  2500: 1.042
Loss after mini-batch  3000: 1.050
Loss after mini-batch  3500: 1.051
Loss after mini-batch  4000: 1.050
Loss after mini-batch  4500: 1.051
Loss after mini-batch  5000: 1.050
Binary accuracy:  tensor(99.)
Starting epoch 8/10
Loss after mini-batch   500: 1.050
Loss after mini-batch  1000: 1.052
Loss after mini-batch  1500: 1.049
Loss after mini-batch  2000: 1.045
Loss after mini-batch  2500: 1.050
Loss after mini-batch  3000: 1.048
Loss after mini-batch  3500: 1.047
Loss after mini-batch  4000: 1.047
Loss after mini-batch  4500: 1.047
Loss after mini-batch  5000: 1.046
Binary accuracy:  tensor(99.)
Starting epoch 9/10
Loss after mini-batch   500: 1.044
Loss after mini-batch  1000: 1.042
Loss after mini-batch  1500: 1.041
Loss after mini-batch  2000: 1.040
Loss after mini-batch  2500: 1.042
Loss after mini-batch  3000: 1.044
Loss after mini-batch  3500: 1.043
Loss after mini-batch  4000: 1.042
Loss after mini-batch  4500: 1.041
Loss after mini-batch  5000: 1.043
Binary accuracy:  tensor(99.)
Starting epoch 10/10
Loss after mini-batch   500: 1.042
Loss after mini-batch  1000: 1.045
Loss after mini-batch  1500: 1.044
Loss after mini-batch  2000: 1.042
Loss after mini-batch  2500: 1.041
Loss after mini-batch  3000: 1.041
Loss after mini-batch  3500: 1.040
Loss after mini-batch  4000: 1.042
Loss after mini-batch  4500: 1.043
Loss after mini-batch  5000: 1.043
Binary accuracy:  tensor(99.)
MLP trained.
Size of MLP:  56

Process finished with exit code 0
```

#### 2.
TODO