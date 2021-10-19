# Standard import statements
import numpy as np
import pandas as pd
import datetime as dt
from datetime import date as dte
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
import seaborn as sns

import mead_general as mead
import mead_pandas as mpd

# England region populations
# https://www.ons.gov.uk/peoplepopulationandcommunity/populationandmigration/populationestimates
# /bulletins/annualmidyearpopulationestimates/mid2019
population_regions = {
    'North East':2.67e6,
    'North West':7.34e6,
    'Yorkshire and The Humber':5.50e6,
    'East Midlands':4.84e6,
    'West Midlands':5.93e6,
    'East of England':6.24e6,
    'London':8.96e6,
    'South East':9.18e6, 
    'South West':5.62e6,
}

# UK nation populations (from Google)
population_nations = {
    'England':55.98e6,
    'Scotland':5.45e6,
    'Wales':3.14e6,  
    'Northern Ireland':1.89e6,
}

# Country populations as of 2020
# https://www.worldometers.info/world-population/population-by-country/
population_countries = {
    'United Kingdom': 67.886e6,
    'Spain': 46.755e6,
    'Germany': 83.784e6,
    'Austria': 9.006e6,
    'United States': 331.002e6,
    'France': 65.274e6,
    'India': 1380.004e6,
    'Brazil': 212.559e6,
    'Italy': 60.461e6,
    'Russia': 145.934e6,
}

# Lockdown dates for Nations and Regions
lockdowns = {
    'North East': {(dte(2020, 3, 23), dte(2020, 5, 13)), (dte(2020, 11, 5), dte(2020, 12, 2)), (dte(2021, 1, 5), dte(2021, 3, 28))},
    'North West': {(dte(2020, 3, 23), dte(2020, 5, 13)), (dte(2020, 11, 5), dte(2020, 12, 2)), (dte(2021, 1, 5), dte(2021, 3, 28))},
    'Yorkshire and The Humber': {(dte(2020, 3, 23), dte(2020, 5, 13)), (dte(2020, 11, 5), dte(2020, 12, 2)), (dte(2021, 1, 5), dte(2021, 3, 28))},
    'East Midlands': {(dte(2020, 3, 23), dte(2020, 5, 13)), (dte(2020, 11, 5), dte(2020, 12, 2)), (dte(2021, 1, 5), dte(2021, 3, 28))},
    'West Midlands': {(dte(2020, 3, 23), dte(2020, 5, 13)), (dte(2020, 11, 5), dte(2020, 12, 2)), (dte(2021, 1, 5), dte(2021, 3, 28))},
    'East of England': {(dte(2020, 3, 23), dte(2020, 5, 13)), (dte(2020, 11, 5), dte(2020, 12, 2)), (dte(2020, 12, 26), dte(2021, 3, 28))},
    'London': {(dte(2020, 3, 23), dte(2020, 5, 13)), (dte(2020, 11, 5), dte(2020, 12, 2)), (dte(2020, 12, 20), dte(2021, 3, 28))},
    'South East': {(dte(2020, 3, 23), dte(2020, 5, 13)), (dte(2020, 11, 5), dte(2020, 12, 2)), (dte(2020, 12, 20), dte(2021, 3, 28))},
    'South West': {(dte(2020, 3, 23), dte(2020, 5, 13)), (dte(2020, 11, 5), dte(2020, 12, 2)), (dte(2021, 1, 5), dte(2021, 3, 28))},
    'England': {(dte(2020, 3, 23), dte(2020, 5, 13)), (dte(2020, 11, 5), dte(2020, 12, 2)), (dte(2021, 1, 5), dte(2021, 3, 28))},
    'Wales': {(dte(2020, 3, 23), dte(2020, 5, 13)), (dte(2020, 10, 23), dte(2020, 11, 9)), (dte(2020, 12, 20), dte(2021, 3, 26))},
    'Scotland': {(dte(2020, 3, 23), dte(2020, 5, 13)), (dte(2020, 10, 9), dte(2020, 10, 25)), (dte(2021, 1, 5), dte(2021, 4, 1))},
    'Northern Ireland': {(dte(2020, 3, 23), dte(2020, 5, 13)), (dte(2020, 10, 16), dte(2020, 11, 20)), (dte(2020, 11, 27), dte(2020, 12, 11)), (dte(2020, 12, 26), dte(2021, 3, 31))},
    'United Kingdom': {(dte(2020, 3, 23), dte(2020, 5, 13)), (dte(2020, 11, 5), dte(2020, 12, 2)), (dte(2021, 1, 5), dte(2021, 3, 28))},
}

### Parameters ###

# General
days_in_roll = 7     # Number of days that contribute to a 'roll' (one week; seven days)
pop_norm_num = 1e5   # Normalisation for y axes (per population; usually 100,000; sometimes 1,000,000)
infect_duration = 5. # Number of days an average person is infectious for (used in calculations of R number; lower values squash R about 1)
rolling_offset = 3   # Number of days to offset rolling data

# Cases plot
case_bar_color = 'cornflowerblue'
case_line_color = 'b'
case_line_label = 'Positive tests'

# Hospitalisations plot
hosp_fac = 10
hosp_bar_color = 'g'
hosp_line_color = 'forestgreen'
hosp_line_label = r'Hospital admissions $[\times %d]$'%(int(hosp_fac))

# Deaths plot
death_fac = 20
death_bar_color = 'indianred'
death_line_color = 'r'
death_line_label = r'Deaths $[\times %d]$'%(int(death_fac))

### ###

def read_JHU_data(file):
    '''
    Read and organise JHU data downloaded from https://github.com/CSSEGISandData/COVID-19
    '''
    df = pd.read_csv(file)
    df.drop(['Lat', 'Long'], axis='columns', inplace=True)
    df['Region'] = df['Country/Region']+df['Province/State'].fillna('') # Merge country and region data
    df.drop(['Country/Region', 'Province/State'], axis='columns', inplace=True)
    col = df.pop('Region')
    df.insert(0, col.name, col)
    df.set_index('Region', inplace=True)
    df.columns = pd.to_datetime(df.columns)
    df.rename(index={'US': 'United States'}, inplace=True)
    df.rename(index={'UK': 'United Kingdom'}, inplace=True)
    return df

def calculate_JHU_data(df):
    '''
    Calculate daily data from cumulative
    '''
    center = False
    df_new = df.diff(axis='columns').copy() # Convert from cumulative to daily numbers
    df_new = df_new.rolling(7, min_periods=1, axis='columns', center=center).sum()/7. # Average over week
    if center: # Remove final three entries that will be anomalously low
        for _ in range(3):
            df_new.drop(df_new.columns[len(df_new.columns)-1], axis='columns', inplace=True)
    return df_new

def plot_UK_death_rate(df):
    '''
    Plot the UK death rate from COVID-19 over the course of the pandemic.
    Also add information about flu deaths and total deaths.
    '''
    import matplotlib.patches as mpatches
    import matplotlib.ticker as ticker

    # Parameters
    deaths = { # Comparable deaths
        'All-cause deaths in a typical year': 530841./365., # 2019 total averaged over 365 days
        'Flu deaths in a typical flu year': 15000./365., # Total averaged over year
        'Flu deaths at height of a bad flu year': 4.*25000./365., # Compress total over 3 months (shoddy)
    }

    # Plot
    _, ax = plt.subplots(figsize=(18.,4.))
    sort_month_axis(plt)
    plot_month_spans(plt, alpha=0.025)
    plot_lockdown_spans(plt, df, region='United Kingdom', label='Lockdowns')
    sns.lineplot(data=df.loc['United Kingdom'], label='UK COVID-19 deaths')
    plt.ylabel('Deaths per day during pandemic')
    plt.xlabel('')
    plt.ylim(bottom=0., top=1500)
    plt.xlim([dte(2020, 1, 1), max(df.columns)+relativedelta(months=6)])
    for i, death in enumerate(deaths.keys()):
        if i == 0:
            delta = -85.
        else:
            delta = 40.
        plt.axhline(deaths[death], color='black', alpha=0.8, ls=':')
        plt.text(dte(2021, 11, 1), deaths[death]+delta, death)
    handles, _ = ax.get_legend_handles_labels()
    patch = mpatches.Patch(alpha=0., label=dt.date.today().strftime("%Y-%m-%d"))
    handles.insert(0, patch) 
    plt.legend(handles=handles, loc='upper left', framealpha=1., edgecolor='k')
    ax.get_yaxis().set_major_formatter(ticker.FuncFormatter(lambda x, _: format(int(x), ',')))
    #plt.title(dt.date.today().strftime("%Y-%m-%d"), x=0.19, y=0.88, loc='Center', bbox=dict(facecolor='w', edgecolor='k'))
    plt.tight_layout()

def plot_world_death_rate(df, countries):
    '''
    World death rate in different countries.
    '''
    plt.figure(figsize=(18.,4.))
    sort_month_axis(plt)
    plot_month_spans(plt, alpha=0.025)
    for country in countries:
        sns.lineplot(data=df.loc[country]*1e6/population_countries[country], label=country)
    plt.ylabel('Deaths per day per million population')
    plt.xlabel('')
    plt.ylim(bottom=0.)
    plt.xlim([dte(2020, 1, 1), max(df.columns)+relativedelta(months=3)])
    plt.legend(loc='upper right', edgecolor='k')
    plt.title(dt.date.today().strftime("%Y-%m-%d"), x=0.015, y=0.88, loc='Left', bbox=dict(facecolor='w', edgecolor='k'))
    plt.tight_layout()

def plot_world_deaths(df, countries):
    ''' 
    World total deaths in different countries.
    '''
    import matplotlib.ticker as ticker
    _, ax = plt.subplots(figsize=(18.,4.))
    sort_month_axis(plt)
    plot_month_spans(plt, alpha=0.025)
    for country in countries:
        sns.lineplot(data=df.loc[country]*1e6/population_countries[country], label=country)
    plt.ylabel('Total deaths per million population')
    plt.xlabel('')
    plt.ylim(bottom=0.)
    plt.xlim([dte(2020, 1, 1), max(df.columns)+relativedelta(months=3)])
    plt.legend(loc='upper right', edgecolor='k')
    plt.title(dt.date.today().strftime("%Y-%m-%d"), x=0.015, y=0.88, loc='Left', bbox=dict(facecolor='w', edgecolor='k'))
    ax.get_yaxis().set_major_formatter(ticker.FuncFormatter(lambda x, _: format(int(x), ',')))
    plt.tight_layout()

def download_data(area, metrics, verbose=True):
    '''
    Download latest UK government data
    '''
    import requests
    
    # Parameters
    today = dt.date.today() # Today's date
    
    if verbose: 
        print('Downloading data')
        print('')

    # Form the link
    link = 'data?areaType='+area
    for metric in metrics:
        link = link+'&metric='+metric
    link = link+'&format=csv'
    
    # Full URL and destination file path
    url = 'https://api.coronavirus.data.gov.uk/v2/'+link
    file = '/Users/Mead/COVID-19/govUK/data/'+area+'_'+today.strftime("%Y-%m-%d")+'.csv'
    
    if verbose: 
        print('URL: %s'%(url))
        print('File: %s'%(file))
        print('')

    req = requests.get(url, allow_redirects=True)
    open(file, 'wb').write(req.content) # Write to disk

    if verbose:
        print('Metrics downloaded:')
        for metric in metrics:
            print(metric)
        print('')
        print('Download complete')
        print('')

def read_data(infile, metrics, verbose=False):
    '''
    Read in data, no manipulations apart from renaming and deleting columns
    '''

    # Read data into a pandas data frame
    df = pd.read_csv(infile)

    # Print data to screen
    if verbose:
        print(type(df))
        print(df)
        print('')

    # Convert date column to actual datetime type
    df['date'] = pd.to_datetime(df['date'])

    # Remove unnecessary columns
    for col in ['areaType', 'areaCode']:
        df.drop(col, inplace=True, axis=1)
    
    # Rename columns
    df.rename(columns={'areaName':'Region'}, inplace=True, errors='raise')
    df.rename(columns=metrics, inplace=True, errors='raise')

    # Print data to screen again
    if verbose:
        print(type(df))
        print(df)
        print('')

    # Sort
    if verbose: mpd.data_head(df, 'Original data')
    df.sort_values(['Region', 'date'], ascending=[True, True], inplace=True)
    if verbose: mpd.data_head(df, 'Sorted data')
    
    # Return the massaged pandas data frame
    return df

def data_calculations(df, verbose):
    '''
    Perform calculations on data (assumes data organised in date from high to low for each region)
    '''
    # Parameters
    days_roll = days_in_roll

    # Calculate rolling cases and deaths (sum over previous week)
    for col in ['Cases', 'Deaths', 'Hosp']:
        if col in df:
            print('Calculating:', col+'_roll_Mead')
            df[col+'_roll_Mead'] = df.groupby('Region')[col].rolling(days_roll).sum(skipna=False).reset_index(0,drop=True)
    mpd.data_head(df, 'Weekly rolling numbers calculated', verbose)

    def doubling_time(ratio):
        '''
        ratio must be the ratio of number of cases between days_roll 
        (e.g., 7 days apart)
        '''
        return days_roll*np.log(2.)/np.log(ratio)

    # Calculate doubling times and R estimates
    # TODO: Surely can avoid creating col_roll_past and col_ratio using offsetting somehow?
    for col in ['Cases', 'Deaths']:
        if col+'_roll_Mead' in df:
            print('Calculating:', col+'_double')
            df[col+'_roll_past'] = df.groupby('Region')[col+'_roll_Mead'].shift(periods=days_roll)
            df[col+'_ratio'] = df[col+'_roll_Mead']/df[col+'_roll_past']
            df[col+'_double'] = df[col+'_ratio'].apply(doubling_time)
            df[col+'_R'] = 1.+infect_duration/df[col+'_double']
            df.drop([col+'_roll_past', col+'_ratio'], inplace=True, axis='columns')
    mpd.data_head(df, 'Doubling times and R values calculated', verbose)
 
def useful_info(regions, data):
    '''
    Print useful information
    '''

    # Parameters
    norm_pop = pop_norm_num

    # File info
    print('Today\'s date:', dt.date.today())
    latest = data['date'].max().strftime("%Y-%m-%d")
    print('Latest date in file:', latest)
    print()

    # Loop over regions
    for region in regions:
        
        # Calculations for per 100,000 population
        pop = regions[region]
        fac = norm_pop/pop
        
        # Region
        print('Region:', region)

        # Isolate regional data
        df = data.loc[data['Region']==region].copy()
        df.sort_values(['date'], ascending=[False], inplace=True)

        # Date
        print('Date:', df['date'].iloc[0].strftime("%Y-%m-%d"))

        # Daily tally
        for col in ['Cases', 'Deaths']:
            if col in df:
                daily = df[col].iloc[0]
                norm_daily = daily*fac
                print('Daily new '+col.lower()+': %d, or per 100,000 population: %.1f.'%(daily, norm_daily))

        # Weekly tally
        for col in ['Cases', 'Deaths']:
            if col+'_roll' in df:
                weekly = df[col+'_roll'].iloc[0]
                if col+'_roll_Mead' in df:
                    my_weekly = df[col+'_roll_Mead'].iloc[0]
                    if weekly != my_weekly:
                        raise ValueError('My calculation of weekly roll disagrees with official')
                norm_weekly = weekly*fac
                print('Weekly '+col.lower()+': %d, or per 100,000 population: %.1f.'%(weekly, norm_weekly))
        
        # Doubling times
        for col in ['Cases']:
            if col+'_double' in df:
                cases_double = df[col+'_double'].iloc[0]
                if cases_double > 0:
                    print(col+' doubling time [days]: %.1f'%(cases_double))
                else:
                    print(col+' halving time [days]: %.1f'%(-cases_double))

        # R values
        if 'Cases_R' in df:
            R = df['Cases_R'].iloc[0]
            print('Estimated R value: %.2f'%(R))

        # White space
        print()

def sort_month_axis(plt):
    '''
    Get the months axis of a plot looking nice
    '''
    import matplotlib.dates as mdates
    import matplotlib.ticker as ticker
    locator_monthstart = mdates.MonthLocator() # Start of every month
    locator_monthmid = mdates.MonthLocator(bymonthday=15) # Middle of every month
    fmt = mdates.DateFormatter('%b') # Specify the format - %b gives us Jan, Feb...
    X = plt.gca().xaxis
    X.set_major_locator(locator_monthstart)
    X.set_minor_locator(locator_monthmid)
    X.set_major_formatter(ticker.NullFormatter())
    X.set_minor_formatter(fmt)
    plt.tick_params(axis='x', which='minor', bottom=False, labelbottom=True)
    plt.xlabel(None)

def plot_month_spans(plt, color='black', alpha=0.05):
    '''
    Plot the spans between months
    '''
    for year in [2020, 2021, 2022, 2023]:
        for month in [2, 4, 6, 8, 10, 12]:
            plt.axvspan(dt.date(year, month, 1), dt.date(year, month, 1)+relativedelta(months=+1), 
                        alpha=alpha, 
                        color=color,
                        lw=0.,
                        )

def plot_lockdown_spans(plt, data, region, color='red', alpha=0.25, label='Lockdown'):
    '''
    Plot the spans of lockdown
    '''
    for id, dates in enumerate(lockdowns.get(region)):
        lockdown_start_date = dates[0]
        if len(dates) == 1:
            lockdown_end_date = max(data.date)
        elif len(dates) == 2:
            lockdown_end_date = dates[1]
        else:
            raise TypeError('Lockdown dates should have either one or two entries')
        if id == 0:
            llabel = label
        else:
            llabel = None
        plt.axvspan(lockdown_start_date, lockdown_end_date, 
                    alpha=alpha, 
                    color=color, 
                    label=llabel,
                    lw=0.,
                   )

def plot_bar_data(df, start_date, end_date, regions, outfile=None, pop_norm=True, Nmax=None, plot_type='Square'):
    '''
    Plot daily data and bar/line charts
    params
        df - pandas data frame
        date - used for plot title, usually today's date
        start_date - start date for x axis
        end_date - end date for x axis
        regions - list of regions to plot (should be length 1, 4 or 9)
        outfile - output file path
        pop_norm - Should plotted quantites be normalised per head of population
        Nmax - Maximum value for y axis
        plot_type - Either 'Square', 'Long', or 'Elongated'
    '''
    # Imports
    import matplotlib
    import matplotlib.ticker as ticker
    
    # Parameters
    days_roll = days_in_roll
    pop_num = pop_norm_num
    bar_width = 1.
    use_seaborn = True
    dpi = 200 # 100 is default
    date = max(df['date'])

    ### Figure options ###

    # Seaborn
    if use_seaborn:
        sns.set_theme(style='ticks')
    else:
        sns.reset_orig
        matplotlib.rc_file_defaults()

    # Number of plots
    n = len(regions)

    # Size
    if n == 9:
        if plot_type == 'Square':
            figx = 18.; figy = 11.
        elif plot_type == 'Long':
            figx = 18.; figy = 50.
        else:
            raise ValueError('plot_type must be either Square or Long')
    elif n == 4:
        if plot_type == 'Square':
            figx = 18.; figy = 11.
        elif plot_type == 'Long':
            figx = 18.; figy = 24.
        else:
            raise ValueError('plot_type must be either Square or Long')
    elif n == 1:
        if plot_type == 'Elongated':
            figx = 18; figy = 4.
        else:
            figx = 6.; figy = 4.
    else:
        print('Number of regions:', n)
        raise ValueError('Only one, four or nine regions supported')

    ### ###

    # Lockdowns
    plot_lockdowns = True
    plot_months = True

    # Initialise plot
    _, axs = plt.subplots(figsize=(figx,figy), sharex=True, sharey=True, dpi=dpi)
    
    # Loop over regions
    for i, region in enumerate(regions):
        
        if pop_norm:
            pop_fac = pop_num/regions[region]
        else:
            pop_fac = 1.
        
        # Sort layout
        if plot_type in ['Long', 'Elongated']:
            plt.subplot(n, 1, i+1)
        elif plot_type == 'Square':
            if not mead.is_perfect_square(n):
                raise ValueError('Need a square number of plots for square plot')
            r = int(np.sqrt(n))
            plt.subplot(r, r, i+1)
        else:
            raise ValueError('plot_type not understood')
          
        # Spans
        if plot_months: plot_month_spans(plt)
        if plot_lockdowns: plot_lockdown_spans(plt, df, region)

        # Plot bars for numbers per day
        for col in ['Cases', 'Hosp', 'Deaths']:
            if col in df:
                if col == 'Cases':
                    fac = pop_fac
                    bar_color = case_bar_color
                elif col == 'Hosp':
                    fac = pop_fac*hosp_fac
                    bar_color = hosp_bar_color
                elif col == 'Deaths':
                    fac = pop_fac*death_fac
                    bar_color = death_bar_color
                else:
                    raise ValueError('Something went wrong')
                plt.bar(df[df['Region']==region]['date'],
                        df[df['Region']==region][col]*fac,
                        width=bar_width,
                        color=bar_color,
                        linewidth=0.
                       )

        # Lines for weekly average
        for col in ['Cases_roll', 'Hosp_roll', 'Deaths_roll']:
            if col in df:
                if col == 'Cases_roll':
                    fac = pop_fac/days_roll
                    line_color = case_line_color
                    line_label = case_line_label
                elif col == 'Hosp_roll':
                    fac = pop_fac*hosp_fac/days_roll
                    line_color = hosp_line_color
                    line_label = hosp_line_label
                elif col == 'Deaths_roll':
                    fac = pop_fac*death_fac/days_roll
                    line_color = death_line_color
                    line_label = death_line_label
                else:
                    raise ValueError('Something went wrong')
                plt.plot(pd.to_datetime(df[df['Region']==region]['date'])-dt.timedelta(rolling_offset),
                         df[df['Region']==region][col]*fac,
                         color=line_color, 
                         label=line_label
                        )

        # Ticks and month arragement on x axis
        sort_month_axis(plt)

        # Axes limits
        plt.xlim(left=start_date, right=end_date)
        plt.ylim(bottom=0)
        if Nmax != None: 
            plt.ylim(top=Nmax)
        if pop_norm:
            plt.ylabel('Number per day per 100,000 population')
        else:           
            plt.ylabel('Number per day')

        # Title of region
        if n == 1:
            title = region+'\n%s'%(date.strftime("%Y-%m-%d"))
            ytit = 0.83
        else:
            title = region
            ytit = 0.90
        if plot_type == 'Elongated':
            xtit = 0.01
        else:
            xtit = 0.03
        plt.title(title, x=xtit, y=ytit, loc='Left', bbox=dict(facecolor='w', edgecolor='k'))

        # Title of date
        if (n == 4 or n == 9) and i == 0:
            title = date.strftime("%Y-%m-%d")
            plt.title(title, x=0.97, y=0.89, loc='Right', bbox=dict(facecolor='w', edgecolor='k'))

        # Legend
        if (n == 9 and i == 2) or (n == 4 and i == 1) or n==1: 
            legend = plt.legend(loc='upper right', framealpha=1.)
            legend.get_frame().set_edgecolor('k')

        # Commas on y axis for long numbers
        if not pop_norm: 
            axs.get_yaxis().set_major_formatter(ticker.FuncFormatter(lambda x, _: format(int(x), ',')))

    #if n != 1: plt.tight_layout()
    plt.tight_layout()
    if(outfile != None): plt.savefig(outfile)
    plt.show(block = False)

def plot_rolling_data(df, start_date, end_date, regions, pop_norm=True, plot_type='Cases', log=True):

    # Plot rolling daily data to directly compare region-to-region

    import matplotlib
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Parameters
    days_roll = days_in_roll
    pop_num = pop_norm_num
    use_seaborn = True
    lockdown_region = 'London'
    plot_lockdowns = True
    plot_months = True
    figx = 17.; figy = 6.
    date = max(df['date'])

    ### Figure options ###

    # Seaborn
    if use_seaborn:
        sns.set_theme(style='ticks')
    else:
        sns.reset_orig
        matplotlib.rc_file_defaults()

    if log:
        if plot_type == 'Cases':
            ymin = 1.
        elif plot_type == 'Deaths':
            ymin = 1e-2
        else:
            raise ValueError('Plot_type not recognised')
    else:
        ymin = 0.

    # Plot
    plt.subplots(figsize=(figx, figy))
    if plot_months: plot_month_spans(plt)
    if plot_lockdowns: plot_lockdown_spans(plt, df, lockdown_region)

    # Loop over regions
    for i, region in enumerate(regions):
        if pop_norm:
            pop_fac = pop_num/regions[region]
        else:
            pop_fac = 1.
        plt.plot(df[df['Region']==region]['date'],
                 df[df['Region']==region][plot_type+'_roll']*pop_fac/days_roll,
                 color='C{}'.format(i), 
                 label=region
                )
        plt.title('Daily new '+plot_type.lower()+': %s' % (date.strftime("%Y-%m-%d")))

    # Ticks and month arragement on x axis
    sort_month_axis(plt)
    plt.xlim(left=start_date, right=end_date)
    if pop_norm:
        plt.ylabel('Number per day per 100,000 population')       
    else:
        plt.ylabel('Total number per day')
    if log: plt.yscale('log')
    plt.ylim(bottom=ymin)
    plt.legend()#loc='upper left')
    plt.show()

def plot_doubling_times(df, start_date, end_date, regions, plot_type='Cases'):

    # Plot doubling-time (or halving time) data to directly compare region-to-region

    import matplotlib
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Parameters
    use_seaborn = True
    lockdown_region = 'London'
    plot_lockdowns = True
    plot_months = True
    figx = 17.; figy = 6.
    ymin = 0.; ymax = 30.
    date = max(df['date'])

    ### Figure options ###

    # Seaborn
    if use_seaborn:
        sns.set_theme(style='ticks')
    else:
        sns.reset_orig
        matplotlib.rc_file_defaults()

    # Plot
    plt.subplots(figsize=(figx, figy))
    if plot_months: plot_month_spans(plt)
    if plot_lockdowns: plot_lockdown_spans(plt, df, lockdown_region)

    # Loop over regions
    for i, region in enumerate(regions):

        # Plot data
        for f in [1, -1]:
            if f == 1:
                ls = '-'
                label=region
            else:
                ls = '--'
                label=None
            plt.plot(df[df['Region']==region]['date'],
                     f*df[df['Region']==region][plot_type+'_double'],
                     color='C{}'.format(i),
                     ls=ls,
                     label=label,
                    )
        plt.title(plot_type+' doubling or halving time: %s'%(date.strftime("%Y-%m-%d")))

    # Axes limits
    sort_month_axis(plt)
    plt.xlim(left=start_date, right=end_date)
    plt.ylabel('Doubling or halving time in days')
    plt.ylim(bottom=ymin, top=ymax)
    plt.legend()#loc='upper left')
    plt.show()

def plot_R_estimates(df, start_date, end_date, regions, plot_type='Cases'):

    # Plot rolling daily data to directly compare region-to-region

    import matplotlib
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Parameters
    use_seaborn = True
    lockdown_region = 'London'
    figx = 17.; figy = 6.
    plot_lockdowns = True
    plot_months = True
    Rmin = 0.; Rmax = 2.
    date = max(df['date'])

    ### Figure options ###

    # Seaborn
    if use_seaborn:
        sns.set_theme(style='ticks')
    else:
        sns.reset_orig
        matplotlib.rc_file_defaults()

    # Plot
    plt.subplots(figsize=(figx, figy))
    if plot_months: plot_month_spans(plt)
    if plot_lockdowns: plot_lockdown_spans(plt, df, lockdown_region)
    plt.axhline(1., color='black')

    # Loop over regions
    for i, region in enumerate(regions):
        plt.plot(df[df['Region']==region]['date'],
                 df[df['Region']==region][plot_type+'_R'],
                 color='C{}'.format(i), 
                 label=region,
                )
        plt.title('Estimated R values: %s'%(date.strftime("%Y-%m-%d")))

    # Axes limits
    sort_month_axis(plt)
    plt.xlim(left=start_date, right=end_date)
    plt.ylabel(r'$R$')
    plt.ylim(bottom=Rmin, top=Rmax)
    plt.legend()
    plt.show()