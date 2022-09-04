import pandas as pd
import numpy as np
from scipy.stats import norm


class summary_stats():
    def __init__(self, periods_per_year, riskfree_rate):
        self.periods_per_year = periods_per_year
        self.riskfree_rate = riskfree_rate
        
        
    def summary(self, r):
        """
        Return a DataFrame that contains aggregated summary stats for the returns in the columns of r.
        """
        ann_r = r.aggregate(self.annualised_rets)
        ann_vol = r.aggregate(self.annualised_vol)
        ann_sr = r.aggregate(self.sharpe_ratio)
        dd = r.aggregate(lambda r: self.drawdown(r).Drawdown.min())
        skew = r.aggregate(self.skewness)
        kurt = r.aggregate(self.kurtosis)
        cf_var5 = r.aggregate(self.var_gaussian, modified=True)
        #hist_cvar5 = r.aggregate(self.cvar_historic)
        return pd.DataFrame({
            "Annualized Return": ann_r,
            "Annualized Vol": ann_vol,
            "Skewness": skew,
            "Kurtosis": kurt,
            "Cornish-Fisher VaR (5%)": cf_var5,
            #"Historic CVaR (5%)": hist_cvar5,
            "Sharpe Ratio": ann_sr,
            "Max Drawdown": dd
        })

    
    def annualised_rets(self, r):
        """
        To annualise return given a set of returns, and input the number of periods each year of the data.
        """
        obs = r.shape[0]
        return ((r+1).prod())**(self.periods_per_year/obs)-1

    
    def annualised_vol(self, r):
        """
        Annualise Volotility of a set of returns.
        Take a set of returns, and input the number of periods each year of the date.
        """
        vol = r.std()
        return vol*np.sqrt(self.periods_per_year)

    
    def sharpe_ratio(self, r):
        """
        Compute annualised Sharpe Ratio of a set of returns.
        Takes returns, risk free rate, and input periods per year.
        """
        rfr_periods = (self.riskfree_rate+1)**(1/self.periods_per_year)-1
        excess_rets = r-rfr_periods
        rets = self.annualised_rets(excess_rets)
        vol = self.annualised_vol(r)
        return rets/vol

    
    def drawdown(self, rets_series: pd.Series):
        """
        Take a time series of assets returns.
        Return DataFrame of: Wealth, Peaks, and Drawdown.
        """
        wealth_index = (rets_series+1).cumprod()
        previous_peaks = wealth_index.cummax()
        drawdown = (wealth_index-previous_peaks)/previous_peaks
        return pd.DataFrame({'Wealth':wealth_index,
                         'Peaks': previous_peaks,
                         'Drawdown': drawdown})

    
    def skewness(self, r):
        """
        Computes the skewness of the supplied Series or DataFrame.
        Returns a float or a Series.
        """
        demeaned_r = r-r.mean()
        # use the population standard deviation, so set dof=0 --> degree of freedom is zero
        sigma_r = r.std(ddof=0)
        exp = (demeaned_r**3).mean()
        return exp/sigma_r**3

    
    def kurtosis(self, r):
        """
        Computes the kurtosis of the supplied Series or DataFrame.
        Returns a float or a Series.
        """
        demeaned_r = r-r.mean()
        # use the population standard deviation, so set dof=0 --> degree of freedom is zero
        sigma_r = r.std(ddof=0)
        exp = (demeaned_r**4).mean()
        return exp/sigma_r**4


    def var_gaussian(self, r, level=5, modified=False): 
        """
        Return the Parametric Gaussian VaR of a Series or DataFrame.
        VaR: a positive number to measure how worse of reuturns a portfolio has (excepting the 5% worst returns) .
        If "modified" = True, then the modified VaR is returned, using the Cornish Fisher modification, otherwise Gasussian.
        """
        # Compute the Z score assuming it was Gaussian
        z = norm.ppf(level/100)
        if modified:
            # modified the Z score based on observerd skewness and kurtosis
            s = self.skewness(r)
            k = self.kurtosis(r)
            z = (z +
                     (z**2 - 1)*s/6 +
                     (z**3 - 3*z)*(k-3)/24 - 
                     (2*z**3 - 5*z)*(s**2)/36
                )
        return -(r.mean() + z*r.std(ddof=0))