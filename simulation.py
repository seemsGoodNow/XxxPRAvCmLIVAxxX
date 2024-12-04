import numpy as np
from typing import Union
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import matplotlib


class PathSimulationModel(ABC):
    @abstractmethod
    def get_simulated_paths(self):
        pass

    def plot_paths(self, paths, quantiles=[0.05, 0.95], ax=None) -> matplotlib.axes.Axes:
        if ax is None:
            fig, ax = plt.subplots(figsize=(18, 8), dpi=150)
        for i in range(paths.shape[1]):
            ax.plot(paths[:, i],  color='#A0A6A4', alpha=0.2)
        linecolor = '#E70E02'
        ax.plot(
            paths.mean(axis=1), color=linecolor,
            label='mean', lw=4, alpha=1
        )
        for q in quantiles:
            ax.plot(
                np.quantile(paths, axis=1, q=q), color=linecolor,
                label=q, lw=1, alpha=1, ls='--'
            )
        ax.legend()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.grid(linewidth=0.5)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(which='minor', length=2.5)
        ax.tick_params(which='major', length=5)
        ax.grid(which='minor', linewidth=0.15, color='tab:grey', alpha=0.25)
        ax.set_xlim(0, len(paths))



class DefaultCIR(PathSimulationModel):
    def __init__(self, alpha: float, theta: float, sigma: float):
        """class for generating default CIR model (using Euler–Maruyama method):
        r(t+1) = (
            r(t) 
            + alpha * (theta - r(t)) 
            + sigma * sqrt(r(t)) * (W(t) - W(t-1))
        )

        Parameters
        ----------
        alpha : float
            speed of returning to mean (theta)
        theta : float
            mean
        sigma : float
            deviation
        """
        self.alpha = alpha
        self.theta = theta
        self.sigma = sigma

    def get_simulated_paths(
            self, n_steps: int, n_paths: int, dt: float, r0: Union[int, float]
        ) -> np.ndarray:
        """

        Parameters
        ----------
        n_steps : int
            number of steps with width dt for every path
        n_paths : int
            number of path to simulate
        dt : float
            width of one step
        r0 : Union[int, float]
            initial value

        Returns
        -------
        np.ndarray
            array with shape (n_steps+1, n_paths), first row is initial r0
        """
        # initialization of first step
        r = np.zeros((n_steps+1, n_paths))
        r[0, :] = r0
        for t in range(n_steps):
            r[t+1, :] = (
                r[t, :] 
                # alpha * (theta - rt) - determined
                + self.alpha * (self.theta - r[t, :])
                # sigma * sqrt(rt) * dW - stochastic
                + self.sigma * r[t, :] * np.random.normal(loc=0, scale=np.sqrt(dt), size=n_paths)
            )
            r[t+1, :] = np.maximum(r[t+1, :], 0)
        return r


class ConstPossionJumpsCIR(PathSimulationModel):

    def __init__(
        self, alpha: float, theta: float, sigma: float, lambda_jump: float,
        jump_size: float
    ):
        """class for generating default CIR model (using Euler–Maruyama method)
        with Poisson jumps:
        r(t+1) = (
            r(t) 
            + alpha * (theta - r(t)) 
            + sigma * sqrt(r(t)) * (W(t) - W(t-1))
            + (N(t) - N(t-1)) * J
        )

        N(t) - N(t-1) - Poisson process with intensity lambda_, 
            determine number of jumps 
        J(t) - strength of jump, const

        Parameters
        ----------
        alpha : float
            speed of returning to mean (theta)
        theta : float
            mean
        sigma : float
            deviation
        lambda_jump : float
            measure for N(t): N(t) - N(t-1) ~ Poisson(lambda_jump * 1)
        jump_size : float
            constant size of jump
        """
        self.alpha = alpha
        self.theta = theta
        self.sigma = sigma
        self.lambda_jump = lambda_jump
        self.jump_size = jump_size

    def get_simulated_paths(
            self, n_steps: int, n_paths: int, dt: float, r0: Union[int, float]
        ) -> np.ndarray:
        """

        Parameters
        ----------
        n_steps : int
            number of steps with width dt for every path
        n_paths : int
            number of path to simulate
        dt : float
            width of one step
        r0 : Union[int, float]
            initial value

        Returns
        -------
        np.ndarray
            array with shape (n_steps+1, n_paths), first row is initial r0
        """
        # initialization of first step
        r = np.zeros((n_steps+1, n_paths))
        r[0, :] = r0
        for t in range(n_steps):
            r[t+1, :] = (
                r[t, :] 
                # alpha * (theta - rt) - determined
                + self.alpha * (self.theta - r[t, :])
                # sigma * sqrt(rt) * dW - stochastic
                + self.sigma * r[t, :] * np.random.normal(loc=0, scale=np.sqrt(dt), size=n_paths)
                # Poisson jumps
                + (
                    # number of jumps 
                    np.random.poisson(lam=self.lambda_jump * dt, size=n_paths)
                    # Constant jump size
                    * self.jump_size
                )
            )
            r[t+1, :] = np.maximum(r[t+1, :], 0)
        return r 


class NormalPossionJumpsCIR(PathSimulationModel):

    def __init__(
        self, alpha: float, theta: float, sigma: float, lambda_jump: float,
        jump_mean: float, jump_std: float
    ):
        """class for generating default CIR model (using Euler–Maruyama method)
        with Poisson jumps:
        r(t+1) = (
            r(t) 
            + alpha * (theta - r(t)) 
            + sigma * sqrt(r(t)) * (W(t) - W(t-1))
            + (N(t) - N(t-1)) * J(t)
        )

        N(t) - N(t-1) - Poisson process with intensity lambda_jump which 
            determines number of jumps 
        J(t) - strength of jump (normal(jump_mean, jump_std))

        Parameters
        ----------
        alpha : float
            speed of returning to mean (theta)
        theta : float
            mean
        sigma : float
            deviation
        """
        self.alpha = alpha
        self.theta = theta
        self.sigma = sigma
        self.lambda_jump = lambda_jump
        self.jump_mean = jump_mean
        self.jump_std = jump_std

    def get_simulated_paths(
            self, n_steps: int, n_paths: int, dt: float, r0: Union[int, float]
        ) -> np.ndarray:
        """

        Parameters
        ----------
        n_steps : int
            number of steps with width dt for every path
        n_paths : int
            number of path to simulate
        dt : float
            width of one step
        r0 : Union[int, float]
            initial value

        Returns
        -------
        np.ndarray
            array with shape (n_steps+1, n_paths), first row is initial r0
        """
        # initialization of first step
        r = np.zeros((n_steps+1, n_paths))
        r[0, :] = r0
        for t in range(n_steps):
            r[t+1, :] = (
                r[t, :] 
                # alpha * (theta - rt) - determined
                + self.alpha * (self.theta - r[t, :])
                # sigma * sqrt(rt) * dW - stochastic
                + self.sigma * r[t, :] * np.random.normal(loc=0, scale=np.sqrt(dt), size=n_paths)
                # Poisson jumps
                + (
                    # number of jumps 
                    np.random.poisson(lam=self.lambda_jump * dt, size=n_paths)
                    # jump size
                    * np.random.normal(
                        loc=self.jump_mean, scale=self.jump_std, size=n_paths
                    )
                )
            )
            r[t+1, :] = np.maximum(r[t+1, :], 0)
        return r 
    


