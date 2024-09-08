Beginner's Guide to Fourier Neural Operators (FNO)
==================================================

.. image:: /_static/images/cover.jpg
  :width: 800
    
The Challenge of Scientific Computing
-------------------------------------

Scientific computations, particularly those involving complex systems like fluid dynamics or many-body motions, are computationally expensive. Traditional numerical solvers, such as finite element methods (FEM) and finite difference methods (FDM), face several challenges:

1. High Computational Cost: To achieve good accuracy, these solvers need to discretize space and time into very fine grids and solve a great number of equations on these grids. This process can take days or even months for complex simulations.

2. Resolution Dependence: The error in these methods typically scales steeply with the resolution. Higher accuracy requires finer grids, which in turn increases computational cost.

3. Lack of Generalization: When an equation is solved on one discretization, it cannot be easily transferred to another resolution or discretization.

The Rise of Data-Driven Methods
-------------------------------

To address these challenges, researchers have been developing data-driven methods based on machine learning techniques, particularly deep learning. These methods learn from data of problems and solutions, allowing them to make predictions without solving all the underlying equations explicitly.

However, these data-driven methods also face limitations:

1. Data Quality Dependence: The quality of predictions depends heavily on the quality of training data.

2. Data Generation Cost: Generating high-quality training data often requires running traditional numerical solvers, which can still be time-consuming.

3. Resolution Specificity: Many neural network-based methods are trained on data of a specific resolution and may not generalize well to other resolutions.

Enter Neural Operators
----------------------

Neural operators, and specifically Fourier Neural Operators (FNOs), aim to address these limitations by learning mappings between function spaces. This approach offers several advantages:

1. Resolution Invariance: By encoding certain structures, neural operators can learn mappings that generalize across different resolutions. This means they can be trained on lower-resolution data but still make reasonable predictions at higher resolutions.

2. Efficiency: Once trained, neural operators can make predictions much faster than traditional numerical solvers, especially for complex problems.

3. Generalization: Instead of solving individual instances, neural operators learn to solve families of PDEs, making them more versatile.


.. image:: /_static/images/mesh_refinement.jpg
  :width: 800

.. image:: /_static/images/math_behind_no.jpg
  :width: 800

Fourier Neural Operators: A Powerful Neural Operator Model
----------------------------------------------------------

FNOs are a specific type of neural operator that leverages the power of the Fourier transform. They offer several key advantages:

1. Speed: FNOs use the Fast Fourier Transform (FFT) to perform convolutions in Fourier space, which is computationally efficient. This allows them to achieve speeds up to 1000x faster than traditional PDE solvers for some problems.

2. Accuracy: FNOs have demonstrated state-of-the-art performance in learning various PDEs, including the challenging Navier-Stokes equation for turbulent flows.

3. Resolution Invariance: FNOs can be trained on one resolution and evaluated on another, a property known as zero-shot super-resolution. This is particularly valuable when high-resolution data is scarce or expensive to obtain.

4. Global Context: Unlike convolutional neural networks (CNNs) which use local filters, FNOs use global spectral filters. This allows them to capture long-range dependencies more effectively, which is crucial for many physical systems.

Practical Applications
----------------------

The advantages of FNOs translate into significant benefits across various scientific and engineering domains:

1. Fluid Dynamics: FNOs have shown remarkable success in modeling turbulent flows, providing substantial speedup compared to traditional methods while maintaining high accuracy.

2. Climate Modeling: The ability of FNOs to capture complex, long-range interactions makes them promising for climate and weather forecasting applications.

3. Computational Mechanics: FNOs have been applied to problems in porous media, fluid mechanics, and solid mechanics, enhancing simulations in these areas.

4. Geosciences: Applications in seismic wave propagation, subsurface Earth discovery, and tomography have demonstrated the versatility of FNOs.


Implementing a Simple FNO Model
-------------------------------

For a more detailed guide, take a look at this link: [placeholder]. Let's walk through the process of implementing and training a simple FNO model using the `neuralop` library.

1. Setup and Imports
~~~~~~~~~~~~~~~~~~~~

First, import the necessary libraries:

.. code-block:: python

    import torch
    import matplotlib.pyplot as plt
    from neuralop.models import FNO
    from neuralop.data.datasets import load_darcy_flow_small
    from neuralop.utils import count_model_params
    from neuralop import LpLoss, H1Loss

2. Load the Dataset
~~~~~~~~~~~~~~~~~~~

We'll use the Darcy flow dataset as an example:

.. code-block:: python

    train_loader, test_loaders, output_encoder = load_darcy_flow_small(
        n_train=100,
        batch_size=16,
        test_resolutions=[16, 32],
        n_tests=[100, 50],
        test_batch_sizes=[32, 32],
    )


Detailed Structure of Fourier Neural Operators (FNO)
====================================================

Before we create an instance of the FNO model, it's crucial to understand its structure and the various components that make it up. The Fourier Neural Operator is designed to learn mappings between function spaces, making it particularly suited for solving partial differential equations (PDEs).

Overall Architecture
--------------------

The FNO architecture consists of several key components:

1. Data Projection Layer
2. FNO Blocks (which include Spectral Layers and MLP Layers)
3. Output Projection Layer


.. image:: /_static/images/fno_marked.jpg
  :width: 800

Let's dive into each of these components:

1. Data Projection Layer (Lifting)
----------------------------------

The data projection layer, also known as the lifting layer, is the first step in the FNO architecture. Its purpose is to lift the input data to a higher-dimensional space.

Function:

- Takes the input data and projects it to a higher-dimensional space.
- Increases the number of channels from the input dimension to the desired hidden dimension.

Implementation: This layer applies a linear transformation to increase the number of channels.


.. code-block:: python

    self.P = nn.Linear(self.in_channels, self.hidden_channels)

2. FNO Blocks
-------------

The core of the FNO architecture consists of multiple FNO blocks. Each block contains two main components:

a) Spectral Layer
~~~~~~~~~~~~~~~~~

The spectral layer is where the Fourier transform magic happens. It performs convolutions in the Fourier space, which is the key innovation of FNOs.

Function:

- Applies Fourier transform to the input
- Performs a linear transformation on the lower Fourier modes
- Applies inverse Fourier transform

Implementation: The spectral layer is implemented using the `SpectralConv2d` class (for 2D problems). Here's a simplified version:

.. code-block:: python

    class SpectralConv2d(nn.Module):
        def __init__(self, in_channels, out_channels, modes1, modes2):
            super(SpectralConv2d, self).__init__()
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
            self.modes2 = modes2
            self.scale = (1 / (in_channels * out_channels))
            self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
            self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

        def forward(self, x):
            batchsize = x.shape[0]
            # Compute Fourier coeffcients up to factor of e^(- something constant)
            x_ft = torch.fft.rfft2(x)

            # Multiply relevant Fourier modes
            out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
            out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
            out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

            # Return to physical space
            x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
            return x

b) MLP Layer
~~~~~~~~~~~~

After the spectral layer, an MLP (Multi-Layer Perceptron) layer is applied on the channels only. This introduces non-linearity and helps in learning complex patterns.

Function:

- Applies non-linear transformations to the output of the spectral layer
- Helps in learning more complex mappings

Implementation:

.. code-block:: python

    self.mlp = nn.Sequential(
        nn.Linear(self.hidden_channels, self.hidden_channels),
        nn.GELU(),
        nn.Linear(self.hidden_channels, self.hidden_channels),
    )

3. Output Projection Layer
--------------------------

The final layer projects the output back to the desired output dimension.

Function:

- Maps the high-dimensional representation back to the target output dimension

Implementation:

.. code-block:: python

    self.Q = nn.Linear(self.hidden_channels, self.out_channels)

Putting It All Together
-----------------------

The FNO model combines these components in a sequential manner:

1. The input data is first lifted to a higher dimension by the data projection layer.
2. It then passes through multiple FNO blocks, each consisting of a spectral layer and an MLP layer.
3. Finally, the output projection layer maps the result back to the desired output dimension.

Here's a simplified version of how these components come together in the forward pass:

.. code-block:: python

    def forward(self, x):
        x = self.fc0(x)  # Lifting layer
        for i in range(self.num_layers - 1):
            x1 = self.conv_layers[i](x)
            x2 = self.w_layers[i](x)
            x = x1 + x2
            x = self.mlp_layers[i](x)
        x = self.fc1(x)  # Output projection
        return x

This structure allows FNOs to efficiently learn mappings between function spaces, making them particularly effective for solving PDEs and other complex mathematical problems. Now finally let's see how to train an FNO model.

3. Set Up the FNO Model
~~~~~~~~~~~~~~~~~~~~~~~

Create an instance of the FNO model:

.. code-block:: python

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FNO(
        n_modes=(16, 16),
        hidden_channels=32,
        in_channels=1,
        out_channels=1,
    )
    model = model.to(device)

4. Define Optimizer and Scheduler
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Set up the optimizer and learning rate scheduler:

.. code-block:: python

    optimizer = torch.optim.Adam(model.parameters(), lr=8e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

5. Define Loss Functions
~~~~~~~~~~~~~~~~~~~~~~~~

Choose appropriate loss functions for training and evaluation:

.. code-block:: python

    l2loss = LpLoss(d=2, p=2)
    h1loss = H1Loss(d=2)
    train_loss = h1loss
    eval_losses = {"h1": h1loss, "l2": l2loss}

6. Train the Model
~~~~~~~~~~~~~~~~~~

Implement a training loop to train the FNO model:

.. code-block:: python

    n_epochs = 20
    for epoch in range(n_epochs):
        model.train()
        for batch in train_loader:
            x, y = batch['x'].to(device), batch['y'].to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = train_loss(out, y)
            loss.backward()
            optimizer.step()
        
        scheduler.step()
        
        # Evaluation code here (omitted for brevity)

7. Visualize Results
~~~~~~~~~~~~~~~~~~~~

After training, visualize the model's predictions:

.. code-block:: python

    test_samples = test_loaders[32].dataset

    fig = plt.figure(figsize=(7, 7))
    for index in range(3):
        data = test_samples[index]
        x = data["x"].to(device)
        y = data["y"].to(device)
        out = model(x.unsqueeze(0))
        
        plt.subplot(3, 1, index+1)
        plt.plot(x.squeeze().cpu().numpy(), label='Input')
        plt.plot(out.squeeze().detach().cpu().numpy(), label='Model')
        plt.plot(y.squeeze().cpu().numpy(), label='Ground Truth')
        plt.legend()
        plt.xlabel('Time')
        plt.ylabel('Value')
        

    plt.tight_layout()
    fig.show()

Conclusion
----------

This guide has introduced you to Fourier Neural Operators and provided a basic implementation using the `neuralop` library. FNOs offer a powerful approach to solving PDEs and learning complex mappings between function spaces. As you become more comfortable with the basics, you can explore more advanced features and applications of FNOs in various scientific and engineering domains.
