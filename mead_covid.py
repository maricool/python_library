# Standard import statements
import numpy as np
import pandas as pd
import datetime as dt
from datetime import date as dte
from dateutil.relativedelta import relativedelta

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

# Lockdown dates for Nations and Regions
lockdowns = {
    'North East':{(dte(2020, 3, 23), dte(2020, 3, 23)), (dte(2020, 11, 5), dte(2020, 12, 2)), (dte(2021, 1, 5), dte(2021, 3, 28))},
    'North West':{(dte(2020, 3, 23), dte(2020, 3, 23)), (dte(2020, 11, 5), dte(2020, 12, 2)), (dte(2021, 1, 5), dte(2021, 3, 28))},
    'Yorkshire and The Humber':{(dte(2020, 3, 23), dte(2020, 3, 23)), (dte(2020, 11, 5), dte(2020, 12, 2)), (dte(2021, 1, 5), dte(2021, 3, 28))},
    'East Midlands':{(dte(2020, 3, 23), dte(2020, 3, 23)), (dte(2020, 11, 5), dte(2020, 12, 2)), (dte(2021, 1, 5), dte(2021, 3, 28))},
    'West Midlands':{(dte(2020, 3, 23), dte(2020, 3, 23)), (dte(2020, 11, 5), dte(2020, 12, 2)), (dte(2021, 1, 5), dte(2021, 3, 28))},
    'East of England':{(dte(2020, 3, 23), dte(2020, 3, 23)), (dte(2020, 11, 5), dte(2020, 12, 2)), (dte(2020, 12, 26), dte(2021, 3, 28))},
    'London':{(dte(2020, 3, 23), dte(2020, 3, 23)), (dte(2020, 11, 5), dte(2020, 12, 2)), (dte(2020, 12, 20), dte(2021, 3, 28))},
    'South East':{(dte(2020, 3, 23), dte(2020, 3, 23)), (dte(2020, 11, 5), dte(2020, 12, 2)), (dte(2020, 12, 20), dte(2021, 3, 28))},
    'South West':{(dte(2020, 3, 23), dte(2020, 3, 23)), (dte(2020, 11, 5), dte(2020, 12, 2)), (dte(2021, 1, 5), dte(2021, 3, 28))},
    'England':{(dte(2020, 3, 23), dte(2020, 3, 23)), (dte(2020, 11, 5), dte(2020, 12, 2)), (dte(2021, 1, 5), dte(2021, 3, 28))},
    'Wales':{(dte(2020, 3, 23), dte(2020, 3, 23)), (dte(2020, 10, 23), dte(2020, 11, 9)), (dte(2020, 12, 20), dte(2021, 3, 26))},
    'Scotland':{(dte(2020, 3, 23), dte(2020, 3, 23)), (dte(2020, 10, 9), dte(2020, 10, 25)), (dte(2021, 1, 5), dte(2021, 4, 1))},
    'Northern Ireland':{(dte(2020, 3, 23), dte(2020, 3, 23)), (dte(2020, 10, 16), dte(2020, 11, 20)), (dte(2020, 11, 27), dte(2020, 12, 11)), (dte(2020, 12, 26), dte(2021, 3, 31))},
}

# Parameters
days_in_roll = 7   # Number of days that contribute to a 'roll' (one week; seven days)
pop_norm_num = 1e5 # Normalisation for y axes (per population; usually 100,000; sometimes 1,000,000)

def download_data(area, metrics):

    # Download latest data
    
    import requests
    
    verbose = True
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
    file = 'data/'+area+'_'+today.strftime("%Y-%m-%d")+'.csv'
    
    if verbose: 
        print('URL: %s' % (url))
        print('')
        print('File: %s' % (file))
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

def read_data(infile, metrics):

    # Read in data, no manipulations apart from renaming and deleting columns
    
    # Parameters
    verbose = False
    
    # Read data into a pandas data frame
    data = pd.read_csv(infile)

    # Print data to screen
    if verbose:
        print(type(data))
        print(data)
        print('')

    # Convert date column to actual date data type
    data.date = pd.to_datetime(data['date'])

    # Remove unnecessary columns
    for col in ['areaType', 'areaCode']:
        data.drop(col, inplace=True, axis=1)
    
    # Rename columns
    data.rename(columns={'areaName': 'Region'}, inplace=True, errors="raise")
    data.rename(columns=metrics, inplace=True, errors="raise")

    # Print data to screen again
    if verbose:
        print(type(data))
        print(data)
        print('')

    # Sort
    if verbose: data_head(data, 'Original data')
    sort_data(data)
    if verbose: data_head(data, 'Sorted data')

    # Print specific columns to screen
    # TODO: This is probably wrong
    if verbose:      
        print(data['date'])
        print('')
        for metric in metrics:
            print(data[metrics[metric]])
            print('')
    
    # Return the massaged pandas data frame
    return data

def sort_data(df):
    # Sort
    df.sort_values(['Region', 'date'], ascending=[True, False], inplace=True)

def data_head(df, comment, verbose=False):
    # Utility function for writing out subset of dataframe with comment
    if verbose:
        print(comment)
        print(df.head(15))
        print()

def data_calculations(df, verbose):

    # Perform calculations on data (assumes organised in date from high to low for each region)
    
    # Parameters
    days_roll = days_in_roll

    # Calculate rolling cases and deaths (sum over previous week)
    for col in ['Cases', 'Deaths', 'Hosp']:
        if col in df:
            print('Calculating:', col+'_roll_Mead')
            df[col+'_roll_Mead'] = df.apply(lambda x: df.loc[(df.Region == x.Region) & (df.date <= x.date) & (df.date > x.date+relativedelta(days=-days_roll)), col].sum(skipna=False), axis=1)
    data_head(df, 'Rolling numbers calculated', verbose)

    # Calculate doubling times
    # TODO: Surely can avoid creating col_roll_past using offsetting somehow?
    for col in ['Cases', 'Deaths']:
        if col+'_roll_Mead' in df:
            print('Calculating:', col+'_double')
            df[col+'_roll_past'] = df.apply(lambda x: df.loc[(df.Region == x.Region) & (df.date == x.date+relativedelta(days=-days_roll)), col+'_roll_Mead'].sum(), axis=1)
            df[col+'_double'] = days_roll*np.log(2.)/np.log(df[col+'_roll_Mead']/df[col+'_roll_past'])
            df.drop(col+'_roll_past', inplace=True, axis=1)
    data_head(df, 'Doubling times calculated', verbose)
 
def useful_info(regions, data):

    # Print useful information

    # Parameters
    norm_pop = 100000

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
        df = data.loc[data['Region'] == region].copy()
        df.sort_values(['date'], ascending=[False], inplace=True)

        # Date
        print('Date:', df['date'].iloc[0].strftime("%Y-%m-%d"))

        # Daily tally
        for col in ['Cases', 'Deaths']:
            if col in df:
                daily = df[col].iloc[0]
                norm_daily = daily*fac
                print('Daily new '+col.lower()+': %d, or per 100,000 population: %.1f.' % (daily, norm_daily))

        # Weekly tally
        for col in ['Cases', 'Deaths']:
            if col+'_roll' in df:
                weekly = df[col+'_roll'].iloc[0]
                if col+'_roll_Mead' in df:
                    my_weekly = df[col+'_roll_Mead'].iloc[0]    
                    if weekly != my_weekly:
                        raise ValueError('My calculation of weekly roll disagrees with official')
                norm_weekly = weekly*fac
                print('Weekly '+col.lower()+': %d, or per 100,000 population: %.1f.' % (weekly, norm_weekly))
        
        # Doubling times
        for col in ['Cases']:
            if col+'_double' in df:
                cases_double = df[col+'_double'].iloc[0]
                if cases_double > 0:
                    print(col+' doubling time [days]: %.1f' % (cases_double))
                else:
                    print(col+' halving time [days]: %.1f' % (-cases_double))
            print()

def sort_month_axis(plt):

    # Get the months axis of a plot looking nice

    import matplotlib.dates as mdates
    import matplotlib.ticker as mticker

    locator_monthstart = mdates.MonthLocator() # Start of every month
    locator_monthmid = mdates.MonthLocator(bymonthday=15) # Middle of every month
    fmt = mdates.DateFormatter('%b') # Specify the format - %b gives us Jan, Feb...

    X = plt.gca().xaxis
    X.set_major_locator(locator_monthstart)
    X.set_minor_locator(locator_monthmid)
    X.set_major_formatter(mticker.NullFormatter())
    X.set_minor_formatter(fmt)
    plt.tick_params(axis='x', which='minor', bottom=False, labelbottom=True)
    plt.xlabel('')

def plot_month_spans(plt):

    # Plot the spans between months

    month_color = 'black'
    month_alpha = 0.05

    for year in [2020, 2021]:
        for month in [2, 4, 6, 8, 10, 12]:
            plt.axvspan(dt.date(year, month, 1), dt.date(year, month, 1)+relativedelta(months=+1), 
                        alpha=month_alpha, 
                        color=month_color,
                        lw=0.,
                        )

def plot_lockdown_spans(plt, data, region):

    # Plot the spans of lockdown

    lockdown_color = 'red'
    lockdown_alpha = 0.25
    lockdown_lab = 'Lockdown'

    for id, dates in enumerate(lockdowns.get(region)):
        lockdown_start_date = dates[0]
        if len(dates) == 1:
            lockdown_end_date = max(data.date)
        elif len(dates) == 2:
            lockdown_end_date = dates[1]
        else:
            raise TypeError('Lockdown dates should have either one or two entries')
        if id == 0:
            label = lockdown_lab
        else:
            label = None
        plt.axvspan(lockdown_start_date, lockdown_end_date, 
                alpha=lockdown_alpha, 
                color=lockdown_color, 
                label=label,
                lw=0.,
                )

def plot_bar_data(data, date, start_date, end_date, regions, outfile=None, pop_norm=True, Nmax=None, plot_type='Square'):

    # Plot daily data and bar/line charts

    # Imports
    import matplotlib
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Parameters
    days_roll = days_in_roll
    pop_num = pop_norm_num
    bar_width = 0.9
    use_seaborn = True
    dpi = 200 # 100 is default

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
            figx = 17.; figy = 13.
        elif plot_type == 'Long':
            figx = 17.; figy = 50
        else:
            raise ValueError('plot_type must be either Square or Long')
    elif n == 4:
        if plot_type == 'Square':
            figx = 12.; figy = 9.
        elif plot_type == 'Long':
            figx = 12.; figy = 24.
        else:
            raise ValueError('plot_type must be either Square or Long')
    elif n == 1:
        figx = 6.; figy = 4.
    else:
        raise ValueError('Only one or nine regions supported')

    # Cases
    case_bar_color = 'cornflowerblue'
    case_line_color = 'b'
    case_line_label = 'Positive tests'

    # Hospitalisations 
    hosp_fac = 5.
    hosp_bar_color = 'g'
    hosp_line_color = 'forestgreen'
    hosp_line_label = r'Hospital admissions $[\times %d]$'%(int(hosp_fac))

    # Deaths
    death_fac = 10.
    death_bar_color = 'indianred'
    death_line_color = 'r'
    death_line_label = r'Deaths $[\times %d]$'%(int(death_fac))

    # Special dates
    #relax_color = 'green'
    #relax_alpha = 0.25
    #relax_lab = 'Relaxation'

    ### ###

    # Relaxations
    #Christmas_date = dt.date(2020, 12, 25)

    # Lockdowns
    plot_lockdowns = True
    plot_months = True
    #plot_relax = False

    # Initialise plot
    plt.subplots(figsize=(figx, figy), sharex=True, sharey=True, dpi=dpi)
    
    # Loop over regions
    for i, region in enumerate(regions):
        
        if pop_norm:
            pop_fac = pop_num/regions[region]
        else:
            pop_fac = 1.
        
        if n == 9:
            if plot_type == 'Square':
                plt.subplot(3, 3, i+1)
            elif plot_type == 'Long':
                plt.subplot(9, 1, i+1)
            else:
                raise ValueError('Something went wrong with plot_type')
        elif n == 4:
            if plot_type == 'Square':
                plt.subplot(2, 2, i+1)
            elif plot_type == 'Long':
                plt.subplot(4, 1, i+1)
            else:
                raise ValueError('Something went wrong with plot_type')
        elif n == 1:
            plt.subplot(1, 1, 1)
        else:
            raise ValueError('Only supports either one or nine regions')
          
        # Months shading
        if plot_months:
            plot_month_spans(plt)

        # Lockdowns
        if plot_lockdowns:
            plot_lockdown_spans(plt, data, region)

        # Important individual dates
        #if plot_relax:
        #    plt.axvspan(Christmas_date, Christmas_date+relativedelta(days=+1), 
        #                color=relax_color, 
        #                alpha=relax_alpha, 
        #                label=relax_lab)

        # Plot data
        q = "Region == '%s'" % (region) # Query to isolate regions

        # Bar chart for numbers per day
        for col in ['Cases', 'Hosp', 'Deaths']:
            if col in data:
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
                plt.bar(data.query(q)['date'],
                        data.query(q)[col]*fac,
                        width=bar_width,
                        color=bar_color,
                        linewidth=0.)

        # Lines for weekly average
        for col in ['Cases_roll', 'Hosp_roll', 'Deaths_roll']:
            if col in data:
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
                plt.plot(data.query(q)['date'],
                        data.query(q)[col]*fac,
                        color=line_color, 
                        label=line_label)

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
            plt.ylabel('Total number per day')

        # Finalise
        if n == 1 and region == 'North East':
            plt.title(region+'\n%s' %(date.strftime("%Y-%m-%d")), x=0.03, y=0.88, loc='Left', bbox=dict(facecolor='w', edgecolor='k'))
        else:
            plt.title(region, x=0.03, y=0.88, loc='Left', bbox=dict(facecolor='w', edgecolor='k'))
        if (n == 4 or n == 9) and i == 0:
            plt.title(date.strftime("%Y-%m-%d"), x=0.97, y=0.88, loc='Right', bbox=dict(facecolor='w', edgecolor='k'))
        if (n == 9 and i == 2) or (n == 4 and i == 1) or (n==1 and region=='North East'): 
            legend = plt.legend(loc='upper right', framealpha=1.)
            legend.get_frame().set_edgecolor('k')

    if(outfile != None):
        plt.savefig(outfile)
    plt.show(block = False)

def plot_rolling_data(data, date, start_date, end_date, regions, pop_norm=True, plot_type='Cases', log=True):

    # Plot rolling daily data to directly compare region-to-region

    import matplotlib
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Parameters
    days_roll = days_in_roll
    pop_num = pop_norm_num
    use_seaborn = True
    lockdown_region = 'London'

    ### Figure options ###

    # Seaborn
    if use_seaborn:
        sns.set_theme(style='ticks')
    else:
        sns.reset_orig
        matplotlib.rc_file_defaults()

    if log:
        plot = plt.semilogy
        if plot_type == 'Cases':
            ymin = 1.
        elif plot_type == 'Deaths':
            ymin = 1e-2
        else:
            raise ValueError('Plot_type not recognised')
    else:
        plot = plt.plot
        ymin = 0.

    # Size
    figx = 17.; figy = 6.

    ### ###

    # Lockdowns
    plot_lockdowns = True

    # Months
    plot_months = True

    # Plot
    plt.subplots(figsize=(figx, figy))

    # Months shading
    if plot_months:
        plot_month_spans(plt)

    # Lockdowns
    if plot_lockdowns:
        plot_lockdown_spans(plt, data, lockdown_region)

    # Loop over regions
    for i, region in enumerate(regions):
        
        if pop_norm:
            pop_fac = pop_num/regions[region]
        else:
            pop_fac = 1.

        # Plot data
        q = "Region == '%s'"%(region) # Query to isolate regions      

        plot(data.query(q)['date'],
            data.query(q)[plot_type+'_roll']*pop_fac/days_roll,
            color='C{}'.format(i), 
            label=region)
        plt.title('Daily new '+plot_type.lower()+': %s' % (date.strftime("%Y-%m-%d")))

        # Cases doubling
        #elif plot_type == 'Cases_double':
        #    plot.plot(data.query(q)['date'],
        #             data.query(q)['Cases_double'],
        #             color='C{}'.format(i),
        #             ls='-',
        #             label=region)
        #    plt.plot(data.query(q)['date'],
        #             -data.query(q)['Cases_double'],
        #             color='C{}'.format(i),
        #             ls='--')
        #    plt.title('Cases doubling or halving time: %s' % (date.strftime("%Y-%m-%d")))

        # Deaths doubling
        #elif plot_type == 'Deaths_double':
        #    plt.plot(data.query(q)['date'],
        #             data.query(q)['Deaths_double'],
        #             color='C{}'.format(i), 
        #             label=region)
        #    plt.title('Deaths doubling or halving time: %s' % (date.strftime("%Y-%m-%d")))

        #else:
        #    raise ValueError('plot_type specified incorrectly')

    # Ticks and month arragement on x axis
    sort_month_axis(plt)

    # Axes limits
    plt.xlim(left=start_date, right=end_date)
    if (plot_type == 'Cases') or (plot_type == 'Deaths') or (plot_type == 'Cases_log') or (plot_type == 'Deaths_log'):
        if pop_norm:
            plt.ylabel('Number per day per 100,000 population')       
        else:
            plt.ylabel('Total number per day')
        plt.ylim(bottom=ymin)
    elif (plot_type == 'Cases_double') or (plot_type == 'Deaths_double'):
        plt.ylabel('Doubling or halving time in days')
        plt.ylim(bottom=0., top=30.)
    else:
        raise ValueError('plot_type specified incorrectly') 

    # Finalise
    plt.legend()#loc='upper left')
    plt.show()