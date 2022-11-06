#!pip install yahoo_fin
#!pip install yahoo_fin --upgrade
#!pip install requests_html
#!pip install investpy

import yahoo_fin.stock_info as si
import investpy
import numpy as np
import pandas as pd
from bond_pricer import *
from pandas.tseries.offsets import BDay
import datetime as dt
from pandas.tseries.holiday import AbstractHolidayCalendar, Holiday, nearest_workday, \
    USMartinLutherKingJr, USPresidentsDay, GoodFriday, USMemorialDay, \
    USLaborDay, USThanksgivingDay


class USTradingCalendar(AbstractHolidayCalendar):
    """
    To identify holidays.
    """
    rules = [
        #Holiday('NewYearsDay', month=1, day=1, observance=nearest_workday),
        USMartinLutherKingJr,
        USPresidentsDay,
        #GoodFriday,
        USMemorialDay,
        Holiday('USIndependenceDay', month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        #Holiday('Christmas', month=12, day=25, observance=nearest_workday)
    ]


class historical_data(USTradingCalendar, bond):
    """
    Get historical data through APIs.
    """
    def __init__(self):
        self.start_date = '2021-01-01'
        self.end_date = '2022-11-6'
        self.ticker_list = ['TSLA', 'AAPL', 'JPM'] # Randomly selected three stocks as an example
        self.bond = 'U.S. 10Y'
    
    
    def get_trading_close_holidays(self):
        inst = USTradingCalendar()

        start = list(map(int, self.start_date.split('-')))
        end = list(map(int, self.end_date.split('-')))

        s_yr = start[0]
        s_mon = start[1]
        s_day = start[2]

        e_yr = end[0]
        e_mon = end[1]
        e_day = end[2]
        return inst.holidays(dt.datetime(s_yr, s_mon, s_day), dt.datetime(e_yr, e_mon, e_day))
    
    
    def stock_price(self):
        """
        Get prices for risky assets.
        """
        trading_data = pd.DataFrame()
        
        for ticker in self.ticker_list:
            ind = si.get_data(ticker, start_date=self.start_date, end_date = self.end_date).close
            trading_data[ticker] = ind
        return trading_data
    
    
    def stock_return(self):
        """
        Get returns for risky assets.
        """
        return self.stock_price().pct_change().dropna()
        
    
    def bond_ytm(self):
        """
        Get yeild for safe assets.
        """
        dates = [self.start_date, self.end_date]
        j=0
        start_end = ['','']
        for i in dates:
            convert = i.split('-')
            convert.reverse()
            start_end[j] = '/'.join(convert)
            j += 1
        bonds_10yrs = investpy.bonds.get_bond_historical_data(bond=self.bond, from_date=start_end[0], to_date=start_end[1])
        ytm = bonds_10yrs.Close
        idx = self.risky_price().index
        b_ytm = ytm.loc[idx]
        isBusinessDay = BDay().onOffset
        weekdays = ytm.index.map(isBusinessDay)
        s = ytm[weekdays]
        holidays = self.get_trading_close_holidays()
        s= s[1:-1].drop(holidays)
        s = pd.DataFrame(s.drop(['2021-04-02', '2022-06-20']))
        s.rename(columns = {'Close':'10yrs Note'}, inplace = True)
        return pd.DataFrame(b_ytm)
    
    
    def bond_price(self, coupon_rate=0.03):
        """
        Get price for safe assets.
        """
        ytm = self.bond_ytm()
        price = self.bond_price(maturity=10, principal=1000, coupon_rate=coupon_rate, coupons_per_year=2, discount_rate=ytm)
        return price
    
    
    def bond_return(self, coupon_rate=0.03):
        """
        Get returns for safe assets.
        """
        ret = self.bond_price().pct.change().dropna()
        return ret
    
    
    def spy_index(self):
        """
        Get SPY index.
        """
        spy_idx = si.get_data('spy', start_date=self.start_date, end_date = self.end_date).close
        return spy_idx
    
    
    def spy_rets(self):
        """
        Get SPY returns.
        """
        idx = self.spy_index()
        rets = idx.pct_change().dropna()
        return pd.DataFrame(rets).rename(columns = {'close':'SP500'})



    
    
    
    
    
    
    
        
        
        
