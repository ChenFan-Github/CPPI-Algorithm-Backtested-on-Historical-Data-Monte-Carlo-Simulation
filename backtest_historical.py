import pandas as pd
import numpy as np
from scipy.optimize import minimize


class portfolio_risky_account():
    '''
    Constructing risky account of a portfolio based on risk parity strategy, which executes portfolio rebalance on a daily basis with stocks data from past 60 trading days.
    '''
    def __init__(self, estimation_window):
        self.estimation_window = estimation_window
        
        
    def safe_account_portfolio(self, r, **kwargs):
        if isinstance(r, pd.Series):
            r = pd.DataFrame(r, columns=['Safe Account'])
        else:
            pass
        return r
    
    
    def risk_parity_portfolio(self, r, **kwargs):
        """
        Backtests a given weighting scheme.
        """
        n_periods = r.shape[0]
        # return windows
        windows = [(start, start+self.estimation_window) for start in range(n_periods-self.estimation_window)]
        weights = [self.weighting_erc(r.iloc[win[0]:win[1]], **kwargs) for win in windows]
        # convert List of weights to DataFrame
        weights = pd.DataFrame(weights, index=r.iloc[self.estimation_window:].index, columns=r.columns)
        returns = pd.DataFrame((weights * r).sum(axis="columns",  min_count=1), columns = ['Risky Account']) #mincount is to generate NAs if all inputs are NAs
        return returns, weights


    # backtest risk_contribution() function to give back the weighter
    def weighting_erc(self, r, **kwargs):
        """
        Produces the weights of the Equally-weighted Risk contributions given a covariance matrix of the returns.
        """
        est_cov = self.sample_cov(r, **kwargs)
        # estimates the cov matrix
        return self.equal_risk_contributions(est_cov)
    
    
    def sample_cov(self, r, **kwargs):
        """
        Returns the sample covariance of the supplied returns.
        """
        return r.cov()
    
    
    def equal_risk_contributions(self, cov):
        """
        Returns the weights of the portfolio that equalizes the contributions of the constituents based on the given covariance matrix.
        """
        n = cov.shape[0]
        return self.target_risk_contributions(target_risk=np.repeat(1/n,n), cov=cov)
    
    
    def target_risk_contributions(self, target_risk, cov):
        """
        Returns the weights of the portfolio that gives you the weights such that the contributions to portfolio risk are as close as possible to the target_risk, given the covariance matrix.
        """
        n = cov.shape[0]
        init_guess = np.repeat(1/n, n)
        bounds = ((0.0, 1.0),) * n # an N-tuple of 2-tuples!
        # construct the constraints
        weights_sum_to_1 = {'type': 'eq',
                            'fun': lambda weights: np.sum(weights) - 1
        }
        def msd_risk(weights, target_risk, cov):
            """
            Returns the Mean Squared Difference in risk contributions between weights and target_risk.
            """
            w_contribs = self.risk_contribution(weights, cov)
            return ((w_contribs-target_risk)**2).sum()

        weights = minimize(msd_risk, 
                           init_guess,
                           args=(target_risk, cov), 
                           method='SLSQP',
                           options={'disp': False},
                           constraints=(weights_sum_to_1,),
                           bounds=bounds)
        return weights.x
    
    
    def risk_contribution(self, w, cov):
        """
        Compute the contributions to risk of the constituents of a portfolio, given a set of portfolio weights and a covariance matrix.
        """
        total_portfolio_var = self.portfolio_vol(w,cov)**2
        marginal_contrib = cov@w
        risk_contrib = np.multiply(marginal_contrib,w.T)/total_portfolio_var
        return risk_contrib
    
    
    def portfolio_vol(self, weights, covmat):
        """
        Computes the vol of a portfolio from a covariance matrix and constituent weights.
        weights: a numpy array or N x 1 maxtrix and covmat is an N x N matrix.
        """
        vol = (weights.T @ covmat @ weights)**0.5
        return vol 

