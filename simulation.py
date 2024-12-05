import numpy as np
import pandas as pd
from typing import Union, Dict, NoReturn, Type
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import matplotlib
from sklearn.linear_model import LinearRegression


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
            self, n_steps: int, n_paths: int, dt: float, r0: Union[int, float],
            dW=None,
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
        if dW is None:
            dW = np.random.normal(loc=0, scale=np.sqrt(dt), size=(n_steps, n_paths))
        # initialization of first step
        r = np.zeros((n_steps+1, n_paths))
        r[0, :] = r0
        for t in range(n_steps):
            r[t+1, :] = (
                r[t, :] 
                # alpha * (theta - rt) - determined
                + self.alpha * (self.theta - r[t, :])
                # sigma * sqrt(rt) * dW - stochastic
                + self.sigma * r[t, :] * dW[t, :]
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
            self, n_steps: int, n_paths: int, dt: float, r0: Union[int, float],
            dW=None
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
        if dW is None:
            dW = np.random.normal(loc=0, scale=np.sqrt(dt), size=(n_steps, n_paths))
        # initialization of first step
        r = np.zeros((n_steps+1, n_paths))
        r[0, :] = r0
        for t in range(n_steps):
            r[t+1, :] = (
                r[t, :] 
                # alpha * (theta - rt) - determined
                + self.alpha * (self.theta - r[t, :])
                # sigma * sqrt(rt) * dW - stochastic
                + self.sigma * r[t, :] * dW[t, :]
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
        if dW is None:
            dW = np.random.normal(loc=0, scale=np.sqrt(dt), size=(n_steps, n_paths))
        # initialization of first step
        r = np.zeros((n_steps+1, n_paths))
        r[0, :] = r0
        for t in range(n_steps):
            r[t+1, :] = (
                r[t, :] 
                # alpha * (theta - rt) - determined
                + self.alpha * (self.theta - r[t, :])
                # sigma * sqrt(rt) * dW - stochastic
                + self.sigma * r[t, :] * dW[t, :]
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


class FXSimulationModel(PathSimulationModel):
    def __init__(
        self, sigma: float, 
    ):
        self.sigma = sigma

    def get_simulated_paths(
            self, 
            n_steps: int, n_paths: int, dt: float,
            r_dom: np.array,
            r_for: np.array, 
            dW_fx: np.array,
            fx_0: float
        ) -> np.ndarray:
        fx = np.zeros((n_steps+1, n_paths))
        fx[0, :] = fx_0
        for t in range(n_steps):
            fx[t+1, :] = (
                fx[t, :] 
                * (
                    (r_for[t, :] - r_dom[t, :]) * dt 
                    + self.sigma * dW_fx[t, :]
                    + 1
                )
            )
            fx[t+1, :] = np.maximum(fx[t+1, :], 0)

        return fx

class ParamEstimator(ABC):

    @abstractmethod
    def get_estimation(self):
        pass


class OLSCIRParamsEstimator(ParamEstimator):

    @staticmethod
    def get_estimation(r: np.ndarray, dt=1) -> Dict[str, float]:
        r_diff = np.diff(r)
        r_sqrt = np.sqrt(r[:-1])
        z1 = dt / r_sqrt
        z2 = dt * r_sqrt
        X = np.stack([z1, z2], axis=1)
        y = r_diff / r_sqrt
        ols_model = LinearRegression(fit_intercept=False).fit(X, y)
        beta1, beta2 = ols_model.coef_

        alpha_hat = -1 * beta2
        theta_hat = beta1 / alpha_hat
        sigma_hat = np.std(y - ols_model.predict(X)) / dt
        return {
            'alpha': alpha_hat,
            'theta': theta_hat,
            'sigma': sigma_hat,
        }

class ConstPoissonParamsEstimator(ParamEstimator):

    @staticmethod
    def get_estimation(r: np.ndarray, quantiles=[0.025, 0.975], dt=1) -> Dict[str, float]:
        r_diff = np.diff(r)
        jumps = r_diff[
            (r_diff < np.quantile(r_diff, quantiles[0]))
            | (r_diff > np.quantile(r_diff, quantiles[1]))
        ]
        jump_size = jumps.mean()
        lambda_jump = jumps.shape[0] / (r_diff.shape[0] * dt)
        return {
            'jump_size': jump_size,
            'lambda_jump': lambda_jump,
        }


class FXsigmaParamsEstimator(ParamEstimator):

    @staticmethod
    def get_estimation(
        r_dom_history: np.array,
        r_for_history: np.array, 
        fx_history: np.array, dt=1
    ) -> Dict[str, float]:
        fx_diff = np.diff(fx_history)
        fx = fx_history[:-1]
        r_dom = r_dom_history[:-1]
        r_for = r_for_history[:-1]
        y = fx_diff/fx - (r_dom - r_for)/100
        error = np.std(y - y.mean())
        sigma = error / dt
        return {
            'sigma': sigma,
        }


class RangeAccrualPricingStrategy:
    def __init__(self, ra_config: Dict, simulation_config: Dict) -> NoReturn:
        self.n_paths = simulation_config.get('n_paths', 1000)
        self.dt = simulation_config.get('df', 1)
        date_range = pd.date_range(
            ra_config['contract_start'], 
            ra_config['contract_end'],
            freq='1D'
        )
        # except weekends
        date_range = date_range[~date_range.weekday.isin([0, 6])]
        self.n_steps = int(len(date_range) / self.dt)
        self.r_dom_history = simulation_config['r_dom_history']
        self.r_for_history = simulation_config['r_for_history']
        self.fx_history = simulation_config['fx_history']
        self.r_dom_0 = self.r_dom_history[-1]
        self.r_for_0 = self.r_for_history[-1]
        self.fx_0 = self.fx_history[-1]

        self.upper_bound = ra_config.get('upper_bound', np.inf)
        self.lower_bound = ra_config.get('lower_bound', -np.inf)
        self.notional = ra_config['notional']
        self.contract_start = ra_config['contract_start']
        self.contract_end = ra_config['contract_end']
        self.notional_payoff = ra_config.get(
            'notional_payoff', 
            self.notional / self.n_steps
        )

    def get_simulation_results(self):
        pass

    def plot_range_w_paths(self, ax=None) -> matplotlib.axes.Axes:
        if ax is None:
            fig, ax = plt.subplots(figsize=(18, 8), dpi=150)
        for i in range(self.fx.shape[1]):
            ax.plot(self.fx[:, i],  color='#A0A6A4', alpha=0.2)
        linecolor = '#E70E02'
        ax.plot(
            self.fx.mean(axis=1), color=linecolor,
            label='mean', lw=4, alpha=1
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
        ax.set_xlim(0, len(self.fx[:, i]))

        ax.axvline(self.lower_bound, color='black', lw=5, ls='--')
        ax.axvline(self.upper_bound, color='black', lw=5, ls='--')
        return ax

class BaselineRangeAccrualPricingStrategy(RangeAccrualPricingStrategy):
    def __init__(self, ra_config: Dict, simulation_config: Dict) -> NoReturn:
        super().__init__(ra_config=ra_config, simulation_config=simulation_config)
        self.poisson_jump_quantiles = [0.025, 0.975]

    def estimate_model_params(self) -> NoReturn:
        self.model_params = {
            'poisson_const_params': ConstPoissonParamsEstimator.get_estimation(
                r=self.r_dom_history, 
                dt=self.dt, quantiles=self.poisson_jump_quantiles
            ),
            'cir_dom_params': OLSCIRParamsEstimator.get_estimation(self.r_dom_history, dt=self.dt),
            'cir_for_params': OLSCIRParamsEstimator.get_estimation(self.r_for_history, dt=self.dt),
            'fx_params': FXsigmaParamsEstimator.get_estimation(
                r_dom_history=self.r_dom_history,
                r_for_history=self.r_for_history,
                fx_history=self.fx_history, 
                dt=self.dt
            )
        }
        self.r_dom_model = ConstPossionJumpsCIR(
            **self.model_params['poisson_const_params'], 
            **self.model_params['cir_dom_params']
        )
        self.r_for_model = DefaultCIR(
            **self.model_params['cir_for_params']
        )
        self.fx_model = FXSimulationModel(
            **self.model_params['fx_params']
        )

    def get_simulation_results(self) -> float:
        self.estimate_model_params()
        # Simulating correlated BM
        dW_dom, dW_for, dW_fx = self.get_correlated_bm()
        common_kws = {
            'n_steps': self.n_steps, 'n_paths': self.n_paths, 'dt': self.dt
        }
        self.r_dom = self.r_dom_model.get_simulated_paths(
            **common_kws, r0=self.r_dom_0, dW=dW_dom
        )
        self.r_for = self.r_for_model.get_simulated_paths(
            **common_kws, r0=self.r_for_0, dW=dW_for
        )
        self.fx = self.fx_model.get_simulated_paths(
            **common_kws, r_dom=self.r_dom, r_for=self.r_for,
             dW_fx=dW_fx, fx_0=self.fx_0
        )
        # first value is given by history, so ignore it
        future = self.fx[1:]
        # calc expected payoff as payoff per day * probability that
        # price will be between lower_bound and upper_bound
        bank_expected_payoff = (
            (
                # condition that price between bounds
                (future >= self.lower_bound) 
                & (future <= self.upper_bound)
            ).mean(axis=1) # mean => probability of price being between bounds by steps
            * self.notional_payoff # payoff per one step
        ).sum() # sum expectations from all steps
        self.fair_price = self.notional - bank_expected_payoff

        return self.fair_price

    
    def get_correlated_bm(self):
        A = np.stack([
            self.r_dom_history, self.r_for_history, self.fx_history
        ], axis=1)
        corr_matrix = np.corrcoef(np.diff(A, axis=0).T)
        L = np.linalg.cholesky(corr_matrix)
        random_bm = np.random.normal(
            loc=0,
            scale=np.sqrt(self.dt),
            size=(self.n_steps, self.n_paths, len(corr_matrix))
        )
        correlated_bm = random_bm @ L
        dW_dom = correlated_bm[:, :, 0]
        dW_for = correlated_bm[:, :, 1]
        dW_fx = correlated_bm[:, :, 2]
        return dW_dom, dW_for, dW_fx
