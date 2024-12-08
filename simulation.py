import numpy as np
import pandas as pd
from typing import Union, Dict, NoReturn, Optional, Tuple
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import matplotlib
from sklearn.linear_model import LinearRegression


class PathSimulationModel(ABC):
    """Base class for simulation paths by different models
    """
    @abstractmethod
    def get_simulated_paths(self):
        pass

    def plot_paths(
        self,
        paths: np.ndarray,
        quantiles=[0.05, 0.95],
        ax=None
    ) -> matplotlib.axes.Axes:
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
        return ax


class DefaultCIR(PathSimulationModel):
    """Class with default CIR model
    """
    def __init__(self, alpha: float, theta: float, sigma: float):
        """CIR model (using Euler–Maruyama method):
        r(t+dt) = (  
            r(t) 
            + alpha * (theta - r(t)) * dt 
            + sigma * sqrt(r(t)) * (W(t+dt) - W(t))
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
            self, 
            n_steps: int, 
            n_paths: int,
            dt: float, 
            r0: Union[int, float],
            dW: Optional[np.ndarray] = None,
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
        dW : Optional[np.ndarray]
            stochastic noize with scale dt. If nothing passed than this noize
            will be generated automaticly

        Returns
        -------
        np.ndarray
            array with shape (n_steps+1, n_paths), first row is initial r0
        """
        if dW is None:
            dW = np.random.normal(
                loc=0, scale=np.sqrt(dt), size=(n_steps, n_paths)
            )
        # initialization of first step
        r = np.zeros((n_steps+1, n_paths))
        r[0, :] = r0
        for t in range(n_steps):
            r[t+1, :] = (
                r[t, :] 
                # alpha * (theta - rt) * dt - determined
                + self.alpha * (self.theta - r[t, :]) * dt
                # sigma * sqrt(rt) * dW - stochastic
                + self.sigma * r[t, :] * dW[t, :]
            )
            # cases when rate becomes less than zero replace to zero
            r[t+1, :] = np.maximum(r[t+1, :], 0)
        return r


class ConstPossionJumpsCIR(PathSimulationModel):
    """Class with default CIR model with Poisson jumps of constant size
    """
    def __init__(
        self, 
        alpha: float, 
        theta: float,
        sigma: float, 
        lambda_jump: float,
        jump_size: float
    ):
        """Default CIR model with Poisson jumps (using Euler–Maruyama method):
        r(t+dt) = (
            r(t) 
            + alpha * (theta - r(t)) * dt 
            + sigma * sqrt(r(t)) * (W(t+dt) - W(t))
            + J * (N(t+dt) - N(t))
        )

        N(t+dt) - N(t) - Poisson process with intensity lambda_jump, determines
            number of jumps per period dt.
            J (jump_size) - constant strength of jump

        Parameters
        ----------
        alpha : float
            speed of returning to mean (theta)
        theta : float
            mean
        sigma : float
            deviation
        lambda_jump : float
            measure for N(t): N(t+dt) - N(t) ~ Poisson(lambda_jump * dt)
        jump_size : float
            constant size of jump
        """
        self.alpha = alpha
        self.theta = theta
        self.sigma = sigma
        self.lambda_jump = lambda_jump
        self.jump_size = jump_size

    def get_simulated_paths(
            self, 
            n_steps: int,
            n_paths: int,
            dt: float, 
            r0: Union[int, float],
            dW: Optional[np.ndarray] = None
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
        dW : Optional[np.ndarray]
            stochastic noize with scale dt. If nothing passed than this noize
            will be generated automaticly

        Returns
        -------
        np.ndarray
            array with shape (n_steps+1, n_paths), first row is initial r0
        """
        if dW is None:
            dW = np.random.normal(
                loc=0, scale=np.sqrt(dt), size=(n_steps, n_paths)
            )
        # initialization of first step
        r = np.zeros((n_steps+1, n_paths))
        r[0, :] = r0
        for t in range(n_steps):
            r[t+1, :] = (
                r[t, :] 
                # alpha * (theta - rt) * dt - determined
                + self.alpha * (self.theta - r[t, :]) * dt
                # sigma * sqrt(rt) * dW - stochastic
                + self.sigma * r[t, :] * dW[t, :]
                # Poisson jumps
                + (
                    # Constant jump size
                    self.jump_size
                    # number of jumps 
                    * np.random.poisson(lam=self.lambda_jump * dt, size=n_paths)
                )
            )
            r[t+1, :] = np.maximum(r[t+1, :], 0)
        return r 


class NormalPossionJumpsCIR(PathSimulationModel):
    """Class with default CIR model with Poisson jumps of normal distributed
    size
    """
    def __init__(
        self, 
        alpha: float, 
        theta: float, 
        sigma: float, 
        lambda_jump: float,
        jump_mean: float, 
        jump_std: float
    ):
        """Default CIR model with Poisson jumps of normal distributed size
        (using Euler–Maruyama method):
        r(t+dt) = (
            r(t) 
            + alpha * (theta - r(t)) * dt 
            + sigma * sqrt(r(t)) * (W(t+dt) - W(t))
            + J * (N(t+dt) - N(t))
        )

        N(t+dt) - N(t) - Poisson process with intensity lambda_jump, determines
            number of jumps per period dt.
            J (jump_size) ~ N(jump_mean, jump_std) strength of jump

        Parameters
        ----------
        alpha : float
            speed of returning to mean (theta)
        theta : float
            mean
        sigma : float
            deviation
        lambda_jump : float
            measure for N(t): N(t+dt) - N(t) ~ Poisson(lambda_jump * dt)
        jump_mean : float
            mean size of jumps for N(jump_mean, jump_std)
        jump_std : float
            std of jumps for N(jump_mean, jump_std)
        """
        self.alpha = alpha
        self.theta = theta
        self.sigma = sigma
        self.lambda_jump = lambda_jump
        self.jump_mean = jump_mean
        self.jump_std = jump_std

    def get_simulated_paths(
            self, 
            n_steps: int, 
            n_paths: int,
            dt: float,
            r0: Union[int, float],
            dW: Optional[np.ndarray] = None
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
        dW : Optional[np.ndarray]
            stochastic noize with scale dt. If nothing passed than this noize
            will be generated automaticly

        Returns
        -------
        np.ndarray
            array with shape (n_steps+1, n_paths), first row is initial r0
        """
        if dW is None:
            dW = np.random.normal(
                loc=0, scale=np.sqrt(dt), size=(n_steps, n_paths)
            )
        # initialization of first step
        r = np.zeros((n_steps+1, n_paths))
        r[0, :] = r0
        for t in range(n_steps):
            r[t+1, :] = (
                r[t, :] 
                # alpha * (theta - rt) * dt - determined
                + self.alpha * (self.theta - r[t, :]) * dt
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
    """Class for log model of currency rate
    """
    def __init__(
        self, 
        sigma: float, 
    ):
        """Log model of currency rate (using Euler–Maruyama method):
        fx(t+dt) = fx(t) * (
            (r_for(t) - r_dom(t)) * dt
            + sigma * (W(t+dt) - W(t))
            + 1
        )

        Parameters
        ----------
        sigma : float
            deviation
        """
        self.sigma = sigma

    def get_simulated_paths(
            self, 
            n_steps: int, 
            n_paths: int,
            dt: float,
            fx0: float,
            r_dom: np.array,
            r_for: np.array, 
            dW_fx: np.array,
        ) -> np.ndarray:
        """Log model of currency rate

        Parameters
        ----------
        n_steps : int
            number of steps with width dt for every path
        n_paths : int
            number of path to simulate
        dt : float
            width of one step
        r_dom : np.array
            domestic rate, array with shape (n_steps+1, n_paths), 
            first row is initial r0 
        r_for : np.array
            foreign rate, array with shape (n_steps+1, n_paths), 
            first row is initial r0 
        dW_fx : np.array
            pre-generated stochastic noize with scale dt correlated with 
            noize of r_dom ant r_for
        fx0 : float
            initial value

        Returns
        -------
        np.ndarray
            array with shape (n_steps+1, n_paths), first row is initial fx0
        """
        fx = np.zeros((n_steps+1, n_paths))
        fx[0, :] = fx0
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
    """Base class for different types of parameters estimators
    """

    @abstractmethod
    def get_estimation(self):
        pass


class OLSCIRParamsEstimator(ParamEstimator):
    """Estimator for Default CIR model, based on OLS method:
        y(t) = beta1 * z1(t) + beta2 * z2(t) + eps(t),
        where 
            y(t) = (r(t+dt) - r(t)) / sqrt(r(t))
            beta1 = alpha * theta
            z1(t) = dt / sqrt(r(t))
            beta2 = -alpha
            z2(t) = dt * sqrt(r(t))
            eps(t) = sigma * sqrt(dt) * N(0, 1)
        and estimations are:
            alpha_hat = -beta2_hat
            theta_hat = beta1_hat / alpha_hat
            sigma_hat = std(err) / dt, where err = std(y_true - y_pred)
    """

    @staticmethod
    def get_estimation(r: np.ndarray, dt: float) -> Dict[str, float]:
        """Get OLS estimations for CIR

        Parameters
        ----------
        r : np.ndarray
            historical values of rate, used to fit OLS
        dt : int
            time step

        Returns
        -------
        Dict[str, float]
            estimation dict with keys 'alpha', 'theta', 'sigma'
        """
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
        sigma_hat = np.std(y - ols_model.predict(X)) / np.sqrt(dt)
        return {
            'alpha': alpha_hat,
            'theta': theta_hat,
            'sigma': sigma_hat,
        }

class ConstPoissonParamsEstimator(ParamEstimator):
    """Estimator of constant jump size for CIR with Poisson jumps
    """

    @staticmethod
    def get_estimation(
        r: np.ndarray, dt: float, quantile: float = 0.95
    ) -> Dict[str, float]:
        """Get estimation of jump size and intensivity based on passed quatniles.
        quantiles define the intensivity of jumps (how much observation will
        be marked as jumps), jump_mean is mean of modules all determined jumps

        Parameters
        ----------
        r : np.ndarray
            historical values of rate, used to fit params
        dt : float
            time step
        quantile : float
            if abs(change) is higher than quantile(changes), then it is jump

        Returns
        -------
        Dict[str, float]
            dict with keys 'jump_size' and 'lambda_jump'
        """
        r_diff = np.abs(np.diff(r))
        jumps = r_diff[
            (r_diff > np.quantile(r_diff, quantile))
        ]
        jump_size = jumps.mean() # take modules
        lambda_jump = jumps.shape[0] / (r_diff.shape[0] * dt)
        return {
            'jump_size': jump_size,
            'lambda_jump': lambda_jump,
        }


class FXsigmaParamsEstimator(ParamEstimator):
    """Class for estimating sigma parameter of currency rate based on OLS strategy:
    fx(t+dt) = fx(t) * (
            (r_for(t) - r_dom(t)) * dt
            + sigma * (W(t+dt) - W(t))
            + 1
        )
    so
    (fx(t+dt) - fx(t)) / fx(t)  - (r_for(t) + r_dom(t)) * dt = sigma * sqrt(dt) * N(0, 1)
    so if left part is y(t), than right part is absolutely stochastic and
    std(y_true - y_pred) = sigma_hat * sqrt(dt).
    """

    @staticmethod
    def get_estimation(
        r_dom_history: np.array,
        r_for_history: np.array, 
        fx_history: np.array,
        dt: float
    ) -> Dict[str, float]:
        """Get OLS estimation of sigma for currency rate

        Parameters
        ----------
        r_dom_history : np.array
            history of domestic rate
        r_for_history : np.array
            history of foreign rate
        fx_history : np.array
            history currency rate
        dt : int, optional
            time step

        Returns
        -------
        Dict[str, float]
            dict with key 'sigma'
        """
        fx_diff = np.diff(fx_history)
        fx = fx_history[:-1]
        r_dom = r_dom_history[:-1]
        r_for = r_for_history[:-1]
        y = fx_diff/fx - (r_dom - r_for) * dt / 100
        # mean = best model for predicting noize
        error = np.std(y - y.mean())
        sigma = error / np.sqrt(dt)
        return {
            'sigma': sigma,
        }


class RangeAccrualPricingStrategy:
    """Base class for strategies of predicting range accrual fair value
    """
    def __init__(self, ra_config: Dict, simulation_config: Dict) -> NoReturn:
        """
        Parameters
        ----------
        ra_config : Dict
            dict with keys:
                'contract_start': str, start of contract
                'contract_end': str, end of contract
                'upper_bound': float,
                'lower_bound': float,
                'notional': float,
                'notional_payoff': payoff per day or array of payoffs
        simulation_config : Dict
            dict with keys:
                'r_dom_history': np.array, history of domestic rate for 
                    estimating parameters of using models.
                'r_for_history': np.array, history of foreign rate for 
                    estimating parameters of using models.
                'fx_history': np.array, history of currency rate for 
                    estimating parameters of using models.
                'n_paths': int, number of paths to simulate
                'n_steps': int, number of steps per path to simulate
                'dt': float, size of one time step
        """
        self.n_paths = simulation_config.get('n_paths', 1000)
        self.dt = simulation_config['dt']
        if 'n_steps' in simulation_config:
            self.n_steps = simulation_config['n_steps']
            self.trade_date_range = pd.date_range(
                ra_config['contract_start'], 
                periods=self.n_steps,
                freq='1D'
            )
        else:
            # calculation of dates between start and end
            date_range = pd.date_range(
                ra_config['contract_start'], 
                ra_config['contract_end'],
                freq='1D'
            )
            # except weekends
            date_range = date_range[~date_range.weekday.isin([0, 6])]
            self.trade_date_range = date_range
            self.n_steps = int(len(date_range))
        self.r_dom_history = simulation_config['r_dom_history']
        self.r_for_history = simulation_config['r_for_history']
        self.fx_history = simulation_config['fx_history']
        # initial value is last passed history value
        self.r_dom_0 = self.r_dom_history[-1]
        self.r_for_0 = self.r_for_history[-1]
        self.fx0 = self.fx_history[-1]
        # parameters of range accrual
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
        """Function for plotting range acrual trajectories with boundaries
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(18, 8), dpi=150)
        for i in range(self.fx.shape[1]):
            ax.plot(self.fx[:, i],  color='#99A6AD', alpha=0.1)
        linecolor = '#DE3C4B'
        ax.plot(
            self.fx.mean(axis=1), color=linecolor,
            label='Mean FX value', lw=5, alpha=1,
        )
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.grid(linewidth=0.5)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(which='minor', length=2.5)
        ax.tick_params(which='major', length=5)
        ax.grid(which='minor', linewidth=0.15, color='tab:grey', alpha=0.25)
        ax.set_xlim(0, self.n_steps+1)
        if not self.lower_bound in [np.inf, -np.inf]:
            ax.axhline(
                self.lower_bound, color='#101D42',
                lw=3, ls='--', label='Lower bound'
            )
        if not self.upper_bound in [np.inf, -np.inf]:
            ax.axhline(
                self.upper_bound, color='#0D1317', lw=3, ls='--',
                label='Upper bound'
            )
            
        ax.axvline(
            1, label=f'Start of contract: {self.contract_start}',
            lw=2, ls='--', color='tab:grey'
        )
        ax.legend()
        ax.set_xlabel('Day from the start of contract (date of making en agrement is 1)')
        return ax

class BaselineRangeAccrualPricingStrategy(RangeAccrualPricingStrategy):
    def __init__(self, ra_config: Dict, simulation_config: Dict) -> NoReturn:
        """
        Parameters
        ----------
        ra_config : Dict
            dict with keys:
                'contract_start': str, start of contract
                'contract_end': str, end of contract
                'upper_bound': float,
                'lower_bound': float,
                'notional': float,
                'notional_payoff': payoff per day or array of payoffs
        simulation_config : Dict
            dict with keys:
                'r_dom_history': np.array, history of domestic rate for 
                    estimating parameters of using models.
                'r_dom_crysis_history': np.array, history of crysis cycles of
                    changing rates for estimating poisson jumps params
                'r_for_history': np.array, history of foreign rate for 
                    estimating parameters of using models.
                'fx_history': np.array, history of currency rate for 
                    estimating parameters of using models.
                'n_paths': int, number of paths to simulate
                'n_steps': int, number of steps per path to simulate
                'dt': float, size of one time step
                'poisson_jump_quantile': quantile for determine which change
                    is a poisson jump, by default  0.9
        """
        super().__init__(ra_config=ra_config, simulation_config=simulation_config)
        self.poisson_jump_quantile = simulation_config.get(
            'poisson_jump_quantile', 0.9
        )
        self.r_dom_crysis_history = simulation_config['r_dom_crysis_history']

    def estimate_model_params(self) -> NoReturn:
        """Estimation of model params based on input and history data
        """
        self.model_params = {
            # parameters of poisson jumps
            'poisson_const_params': ConstPoissonParamsEstimator.get_estimation(
                r=self.r_dom_crysis_history, # crysis data
                dt=self.dt, quantile=self.poisson_jump_quantile
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
        # building models
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
        # Simulating correlated BM based on history data
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
             dW_fx=dW_fx, fx0=self.fx0
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

    
    def get_correlated_bm(self) -> Tuple[np.array]:
        """Function for building correlated noize between r_dom, r_for and fx
        based on history correlations
        """
        A = np.stack([
            self.r_dom_history, self.r_for_history, self.fx_history
        ], axis=1)
        # correlation between changes
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
