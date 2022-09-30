import ipywidgets as widgets
from IPython.display import display
import matplotlib.pyplot as plt
from portfolio_construction import *
from backtest_simulation import *
from metrics import *


class visualisation():
    def __init__(self, n_scenarios, steps_per_year, n_years, mu, sigma_stocks, a, b, sigma_bonds, r_0, m, cash, floor, drawdown):
        self.n_scenarios=n_scenarios
        self.steps_per_year=steps_per_year
        self.n_years = n_years
        
        self.mu=mu
        self.sigma_stocks=sigma_stocks
        
        self.a=a
        self.b=b
        self.sigma_bonds=sigma_bonds
        self.r_0=r_0
        
        self.m=m
        self.cash=cash
        self.floor=floor
        self.drawdown = drawdown
        
        
    def charting_display(self):
        simulation = monte_carlo(self.n_scenarios, self.n_years, self.steps_per_year)
        sim_stocks = simulation.equity_return(self.mu, self.sigma_stocks, s_0=100)[1][1:]
        
        int_rate_simu, zcb = simulation.cir(self.a, self.b, self.sigma_bonds, self.r_0)
        bond_price = simulation.bond_price(maturity=10, principal=1000, coupon_rate=0.0303, coupons_per_year=2, discount_rate=int_rate_simu)
        sim_bonds = pd.DataFrame(bond_price.pct_change()[1:])        
        
        btr = cppi().cppi_construction(risky_r = sim_stocks, 
                                       safe_r = sim_bonds, 
                                       m=self.m, 
                                       cash=self.cash,
                                       floor=self.floor,
                                       drawdown=self.drawdown, 
                                       riskfree_rate=0.03)
        
        wealth = btr['Wealth']
        fig, (wealth_ax, hist_ax) = plt.subplots(nrows=1, ncols=2, sharey=True, gridspec_kw={'width_ratios':[3,2]}, figsize=(24,10))
        fig.suptitle("Backtest CPPI with Simulated Data",fontsize=20)

        plt.subplots_adjust(wspace=0.0)
        amplify = wealth.values.max()

        wealth.plot(ax=wealth_ax, legend=False, figsize=(12,6), color='lightgreen')
        wealth_ax.axhline(y=self.cash, ls=':',color='k')
        wealth_ax.axhline(y=self.cash*self.floor, ls='--',color='r')
        wealth_ax.set_ylim(top=amplify)
        wealth_ax.plot([0],[self.cash],color='indianred',marker='o',markersize=6)

        terminal_prices = wealth.iloc[-1]
        terminal_prices.plot.hist(ax=hist_ax, bins=30, ec='w', fc='goldenrod',orientation='horizontal')
        hist_ax.axhline(y=self.cash, ls=':',color='k')
        hist_ax.axhline(y=self.cash*self.floor, ls='--',color='r')
        
        stats = summary_stats(self.steps_per_year, riskfree_rate=.03)
        result = stats.summary(wealth.pct_change().dropna()).mean()
        
        return pd.DataFrame(result, columns=['CPPI Performance Average: Simulated Data'])
    
    
