import math
import pandas as pd
import numpy as np
from bond_pricer import *


class monte_carlo(bond):
    def __init__(self, n_scenarios = 1000, n_years = 2, steps_per_year = 252):
        """
        n_scenarios: The number of scenarios/trajectories.
        n_years:  The number of years to generate data for.
        steps_per_year: data frequency, ie.granularity of the simulation.
        """
        self.n_scenarios = n_scenarios
        self.n_years = n_years
        self.steps_per_year = steps_per_year


    def equity_return(self,  mu=0.07, sigma=0.15, s_0=100.0):
        """
        Monte Carlo Method for Stock Prices simulation based on Geometric Brownian Motion model. The function returns a numpy array of n_paths columns and n_years*steps_per_year rows.
        mu: Annualized Drift, e.g. Market Return.
        sigma: Annualized Volatility.
        s_0: initial wealth.
        """
        dt = 1/self.steps_per_year
        n_steps = int(self.n_years * self.steps_per_year) + 1
        rets_plus_1 = np.random.normal(loc=(1+mu)**dt, scale=(sigma*np.sqrt(dt)), size=(n_steps, self.n_scenarios))
        rets_plus_1[0] = 1
        prices = s_0*pd.DataFrame(rets_plus_1).cumprod()
        rets = pd.DataFrame(rets_plus_1-1)
        return prices, rets
    

    def cir(self, a=0.05, b=0.03, sigma=0.05, r_0=None):
        """
        Returns:
            1. random annualised interest rate evolution over time using CIR MOdel.
            2. Prices of Zero Coupon Bonds.
        r_0 initial int rate as annualised rates.
        b: long term rate as annualised rates.
        a: a parameter, how fast convert to long term rate.
        """
        if r_0 is None: r_0 =b    
        r_0 = self.ann_to_inst(r_0)
        dt = 1/self.steps_per_year
        num_steps = int(self.n_years*self.steps_per_year)+1 
        shock = np.random.normal(0,np.sqrt(dt),(num_steps, self.n_scenarios))
        rates = np.empty_like(shock) 
        rates[0] = r_0
        prices = np.empty_like(shock)

        def price(ttm, r):
            """
            ttm = T-t --> \tau 
            """
            h = math.sqrt(a**2 + 2*sigma**2)
            _A = ((2*h*math.exp((h + a)*ttm/2))/(2*h + (h+a)*(math.exp(h*ttm)-1)))**(2*a*b/sigma**2)
            _B = (2*(math.exp(h*ttm)-1))/(2*h + (h+a)*(math.exp(h*ttm)-1))
            _P = _A*np.exp(-_B*r)
            return _P

        prices[0] = price(self.n_years, r_0)
 
        for step in range(1, num_steps): 
            r_t = rates[step-1]
            d_r_t = a*(b-r_t)*dt + sigma*np.sqrt(r_t)*shock[step]
            rates[step] = abs(r_t + d_r_t)
            prices[step] = price(self.n_years-step*dt, rates[step])

        rates = pd.DataFrame(data=self.inst_to_ann(rates), index=range(num_steps))
        prices = pd.DataFrame(data=prices, index=range(num_steps))
        return rates, prices
    
    
    def ann_to_inst(self, r):
        """
        Convert annualised to short term.
        """
        return np.log1p(r)
    
    
    def inst_to_ann(self,r):
        """
        Convet short rate to annualised rate.
        """
        return np.expm1(r)
    
    
    
    