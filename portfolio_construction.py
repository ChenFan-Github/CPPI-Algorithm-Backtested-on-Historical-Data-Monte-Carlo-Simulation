import pandas as pd
import numpy as np
from scipy.optimize import minimize
from backtest_simulation import *

   
class cppi(monte_carlo):
    def cppi_construction(self, risky_r, safe_r, m=3, cash=1000, floor=0.8, drawdown=None, riskfree_rate=0.03):
        """
        Run a backtest of the CPPI strategy, given a set of returns for the risky asset
        Returns a dictionary.
        """
        dates = risky_r.index
        n_steps = len(dates)
        account_value = cash
        floor_value = cash*floor
        previous_peak = cash

        if safe_r is None:  
            safe_r = pd.DataFrame().reindex_like(risky_r)
            safe_r.values[:] = riskfree_rate/self.steps_per_year 
    
        if isinstance(risky_r, pd.Series): 
            risky_r = pd.DataFrame(risky_r, columns=['Risky Account'])
        else:
            col = risky_r.columns[0]
            risky_r.rename(columns = {col:'Risky Account'}, inplace=True)
    
        if isinstance(safe_r, pd.Series): 
            safe_r = pd.DataFrame(safe_r, columns=['Safe Account'])
        else:
            col = safe_r.columns[0]
            safe_r.rename(columns = {col:'Safe Account'}, inplace=True)

        if safe_r is None:  
            safe_r = pd.DataFrame().reindex_like(risky_r)
            safe_r.values[:] = riskfree_rate/12         
        
        account_history = pd.DataFrame().reindex_like(risky_r).rename(columns={'Risky Account':'Wealth History'})
        risky_w_history = pd.DataFrame().reindex_like(risky_r).rename(columns={'Risky Account':'Risky Assets Weights History'})
        cushion_history = pd.DataFrame().reindex_like(risky_r).rename(columns={'Risky Account':'Cushion History'})

        for step in range(n_steps): 
            if drawdown!=None:
                previous_peak = np.maximum(previous_peak, account_value)
                floor_value = np.maximum((1-drawdown)*previous_peak, cash*floor) 

            if np.isnan(risky_r.iloc[step]).values[0]:
                cushion_history.iloc[step] = np.NaN
                risky_w_history.iloc[step] = np.NaN
                account_history.iloc[step] = np.NaN
                
            else: 
                cushion = (account_value - floor_value)/account_value 
                risky_w = m*cushion
                risky_w = np.minimum(risky_w, 1)
                risky_w = np.maximum(risky_w, 0)
                safe_w = 1-risky_w
                risky_alloc = account_value*risky_w
                safe_alloc = account_value*safe_w
                account_value = risky_alloc*(1+risky_r.iloc[step]).values + safe_alloc*(1+safe_r.iloc[step]).values
                cushion_history.iloc[step] = cushion
                risky_w_history.iloc[step] = risky_w
                account_history.iloc[step] = account_value

        risky_wealth = cash*(1+risky_r).cumprod()

        backtest_result = {
            "Wealth": account_history,   # wealth accumulation with CPPI
            "Risky Wealth": risky_wealth,  # wealth accumulation if not running CPPI
            "Risk Budget": cushion_history, # the percentage of money as cushion for lossing
            "Risky Allocation": risky_w_history, # percentage of risky asset in portfolio
            "m": m, # aggresive factor
            "cash": cash, #initial money
            "floor": floor, # protection
            "risky_r":risky_r, # risky asset return 
            "safe_r": safe_r, # safe asset return 
                        } 
        return backtest_result