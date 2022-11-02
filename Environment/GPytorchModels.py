import gpytorch
import torch

class GPModel(gpytorch.models.ExactGP):
    """ Multitask Gaussian Process model. """
    def __init__(self, train_x, train_y, likelihood):
        super(GPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GaussianProcessRegressor:

    def __init__(self, 
                initial_lengthscale,
                noise,
                lengthscale_bounds, 
                lr, 
                n_iterations, 
                device = 'cpu') -> None:
        
        # Lengthscale bounds for optimiser 
        self.lengthscale_bounds = lengthscale_bounds
        # Learning rate
        self.lr = lr
        # Iterations
        self.n_iterations = n_iterations
        # Device (for gpu inference)
        self.device = device
        # Likelihood module 
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        # Create the model 
        self.model = GPModel(torch.DoubleTensor([]).to(self.device), 
                                torch.DoubleTensor([]).to(self.device), 
                                self.likelihood).to(self.device)
        self.model.covar_module.base_kernel.lengthscale = torch.DoubleTensor([initial_lengthscale], device=self.device)
        self.likelihood.initialize(noise=noise)
        # Optimizer #
        self.optimizer = torch.optim.Adam([{'params': self.model.parameters()}], lr=self.lr)
        self.lengthscale_history = []
        self.maximum_likelihood_loss = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

    def fit(self, train_X, train_Y, verbose=False, optimize=True):
        """ Train with the data """

        # Data to device (for GPU inference)
        train_X = torch.DoubleTensor(train_X, device=self.device)
        train_Y = torch.DoubleTensor(train_Y, device=self.device)

        # Add new data to the model #
        self.model.set_train_data(inputs=train_X, targets=train_Y, strict=False)

        if not optimize:
            return 

        # models to train mode 
        self.model.train()
        self.likelihood.train()
        
        # Optimization loop #

        for i in range(self.n_iterations):

            self.optimizer.zero_grad()
            output =self. model(train_X)
            loss = -self.maximum_likelihood_loss(output, train_Y).sum()
            loss.backward()

            # Clip the gradient to avoid instabilities
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 100.0)

            if verbose:
                print('Iter %d/%d - Loss: %.3f - L: %.3f' % (i + 1, self.n_iterations, loss.item(), self.model.covar_module.base_kernel.lengthscale))

            self.optimizer.step()

            # Clamp the hiperparameters #
            self.model.covar_module.base_kernel.lengthscale = self.model.covar_module.base_kernel.lengthscale.clamp(self.lengthscale_bounds[0], self.lengthscale_bounds[1])

        self.lengthscale_history.append(self.model.covar_module.base_kernel.lengthscale.item())

    def predict(self, target_X, return_std = False):
        """ Predict the values for target_X """

        self.model.eval()
        self.likelihood.eval()

        with torch.no_grad():

            predictions = self.likelihood(self.model(torch.DoubleTensor(target_X, device = self.device)))

            # Obtain the mean #
            mean = predictions.mean.cpu().numpy()

            if return_std:
                lower, upper = predictions.confidence_region()
                std = upper.cpu().numpy() - lower.cpu().numpy()
                return mean, std
            else: 
                return mean

        
        
if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    # Create the regressor #
    regressor = GaussianProcessRegressor(lengthscale_bounds = (0.0001, 0.5), initial_lengthscale=0.18, noise=1e-4, lr = 0.1, n_iterations = 20, device = 'cpu')
    # Create the data #
    real_x = np.linspace(0, 1, 100)
    train_x = real_x[np.random.randint(0, len(real_x), 5)]
    train_y = np.sin(2 * 2*np.pi * train_x)
    real_y = np.sin(2 * 2*np.pi * real_x)

    for i in range(10):

        train_x = np.concatenate((train_x, [real_x[np.random.randint(0, len(real_x))]]))
        train_y = np.concatenate((train_y, [np.sin(2 * 2*np.pi * train_x[-1])]))
        # Train the data #
        regressor.fit(train_x, train_y, verbose=True)
        # Predict the data #
        mu, std = regressor.predict(real_x, return_std=True)

        with plt.style.context("bmh"):

            plt.plot(real_x, real_y, "--r", label = "True function")
            plt.fill_between(real_x, mu-std, mu+std, color='blue', alpha = 0.2, label="Std.")
            plt.plot(real_x, mu, "b-", label="Predicted function")
            plt.plot(train_x, train_y, "kx", label="Train data")
            plt.legend()
            plt.show(block=True)

    with plt.style.context("bmh"):
        plt.plot(regressor.lengthscale_history,'r-')
        plt.show(block=True)
