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
   * They're reshaped to dimension `res * res * res/2` to form a 3D grid-like which is finally used by the marching cube algorithm to transform the occupancy grid into a 3D mesh.
8. What is the memory size of the MLP ? how does it compare with: (i) A voxel occupancy grid; (ii) The original image set plus the calibration ?


### 2.2 Program Editing

TODO
