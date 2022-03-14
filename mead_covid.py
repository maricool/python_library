# Standard imports
import numpy as np
import pandas as pd
import datetime as dt
from datetime import date
from dateutil.relativedelta import relativedelta
import matplotlib
import matplotlib.dates as dates
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import requests
import seaborn as sns

# My imports
import mead_general as mead
import mead_pandas as mpd

# England region populations
# https://www.ons.gov.uk/peoplepopulationandcommunity/populationandmigration/populationestimates
# /bulletins/annualmidyearpopulationestimates/mid2019
population_regions = {
    'North East': 2.67e6,
    'North West': 7.34e6,
    'Yorkshire and The Humber': 5.50e6,
    'East Midlands': 4.84e6,
    'West Midlands': 5.93e6,
    'East of England': 6.24e6,
    'London': 8.96e6,
    'South East': 9.18e6, 
    'South West': 5.62e6,
}

# Canadian provincial populations
# https://worldpopulationreview.com/canadian-provinces
population_Canadian_provinces = {
    'Canada': 38.01e6,
    'Ontario': 14.733e6,
    'Quebec': 8.576e6,
    'British Columbia': 5.146e6,
    'Alberta': 4.428e6,
    'Manitoba': 1.380e6,
    'Saskatchewan': 1.178e6,
    'Nova Scotia': 0.979e6,
    'New Brunswick': 0.781e6,
    'Newfoundland and Labrador': 0.521e6,
    'Prince Edward Island': 0.160e6,
    'Northwest Territory': 0.045e6,
    'Yukon': 0.042e6,
    'Nanavut': 0.039e6,
}

# UK nation populations (from Google)
population_nations = {
    'England': 55.98e6,
    'Scotland': 5.45e6,
    'Wales': 3.14e6,  
    'Northern Ireland': 1.89e6,
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
    'Australia': 25.500e6,
    'Canada': 37.742e6,
    'China': 1439.323e6,
    'Netherlands': 17.135e6,
    'South Africa': 59.309e6,
    'Belgium': 11.590e6,
}

# Lockdown dates for Nations and Regions
lockdowns = {
    'North East':{
        (date(2020, 3, 23), date(2020, 5, 13)), 
        (date(2020, 11, 5), date(2020, 12, 2)), 
        (date(2021, 1, 5), date(2021, 3, 28)),
        },
    'North West':{
        (date(2020, 3, 23), date(2020, 5, 13)), 
        (date(2020, 11, 5), date(2020, 12, 2)), 
        (date(2021, 1, 5), date(2021, 3, 28)),
        },
    'Yorkshire and The Humber':{
        (date(2020, 3, 23), date(2020, 5, 13)), 
        (date(2020, 11, 5), date(2020, 12, 2)), 
        (date(2021, 1, 5), date(2021, 3, 28)),
        },
    'East Midlands':{
        (date(2020, 3, 23), date(2020, 5, 13)), 
        (date(2020, 11, 5), date(2020, 12, 2)), 
        (date(2021, 1, 5), date(2021, 3, 28)),
        },
    'West Midlands':{
        (date(2020, 3, 23), date(2020, 5, 13)), 
        (date(2020, 11, 5), date(2020, 12, 2)), 
        (date(2021, 1, 5), date(2021, 3, 28)),
        },
    'East of England':{
        (date(2020, 3, 23), date(2020, 5, 13)), 
        (date(2020, 11, 5), date(2020, 12, 2)), 
        (date(2020, 12, 26), date(2021, 3, 28)),
        },
    'London':{
        (date(2020, 3, 23), date(2020, 5, 13)), 
        (date(2020, 11, 5), date(2020, 12, 2)), 
        (date(2020, 12, 20), date(2021, 3, 28)),
        },
    'South East':{
        (date(2020, 3, 23), date(2020, 5, 13)), 
        (date(2020, 11, 5), date(2020, 12, 2)), 
        (date(2020, 12, 20), date(2021, 3, 28)),
        },
    'South West':{
        (date(2020, 3, 23), date(2020, 5, 13)), 
        (date(2020, 11, 5), date(2020, 12, 2)), 
        (date(2021, 1, 5), date(2021, 3, 28)),
        },
    'England':{
        (date(2020, 3, 23), date(2020, 5, 13)), 
        (date(2020, 11, 5), date(2020, 12, 2)), 
        (date(2021, 1, 5), date(2021, 3, 28)),
        },
    'Wales':{
        (date(2020, 3, 23), date(2020, 5, 13)), 
        (date(2020, 10, 23), date(2020, 11, 9)), 
        (date(2020, 12, 20), date(2021, 3, 26)),
        },
    'Scotland':{
        (date(2020, 3, 23), date(2020, 5, 13)), 
        (date(2020, 10, 9), date(2020, 10, 25)), 
        (date(2021, 1, 5), date(2021, 4, 1)),
        },
    'Northern Ireland':{
        (date(2020, 3, 23), date(2020, 5, 13)), 
        (date(2020, 10, 16), date(2020, 11, 20)), 
        (date(2020, 11, 27), date(2020, 12, 11)), 
        (date(2020, 12, 26), date(2021, 3, 31)),
        },
    'United Kingdom':{
        (date(2020, 3, 23), date(2020, 5, 13)), 
        (date(2020, 11, 5), date(2020, 12, 2)), 
        (date(2021, 1, 5), date(2021, 3, 28)),
        },
    'Canada':{},
    'Alberta':{},
    #'British Columbia':{
    #    (date(2020, 3, 18), date(2020, 5, 19)),
    #    (date(2020, 11, 7), date(2021, 6, 15)),
    #},
    'British Columbia':{},
    'Saskatchewan':{},
    'Manitoba':{},
    'Ontario':{},
    'Quebec':{},
    'New Brunswick':{},
    'Prince Edward Island':{},
    'Nova Scotia':{},
    'Newfoundland and Labrador':{},
    'Yukon':{},
    'Northwest Territory':{},
    'Nunavut':{},
}

### Parameters ###

# General
days_in_roll = 7     # Number of days that contribute to a 'roll' (one week; seven days)
pop_norm_num = 1e5   # Normalisation for y axes (per population; usually 100,000; sometimes 1,000,000)
infect_duration = 5. # Number of days an average person is infectious for (used in calculations of R number; lower values squash R about 1)
rolling_offset = 3   # Number of days to offset rolling data

# Cases
case_bar_color = 'cornflowerblue'
case_line_color = 'b'
case_label = 'Cases'

# Hospitalisations
hosp_fac = 10
hosp_bar_color = 'g'
hosp_line_color = 'forestgreen'
hosp_label = r'Hospitalisations $[\times %d]$'%(int(hosp_fac))

# Hospital beds
bed_bar_color = 'm'
bed_line_color = 'violet'
bed_label = r'Hospitalised'

# ICU
icu_bar_color = 'm'
icu_line_color = 'violet'
icu_label = r'Intensive care'

# Deaths
death_fac = 20
death_bar_color = 'indianred'
death_line_color = 'r'
death_label = r'Deaths $[\times %d]$'%(int(death_fac))

### Canada data ###

def read_Canada_data(infile):
    df = pd.read_csv(infile)
    df.sort_values(by=['prname', 'date'], ascending=[True, False], inplace=True)
    df['date'] = pd.to_datetime(df['date'])
    for col in ['numtotal_last7', 'numdeaths_last7']:
        df[col] = df[col]/7.
    return df

def read_Canada_hosp_data(infile):
    df = pd.read_csv(infile)
    df.rename(columns={'Date': 'date'}, inplace=True, errors='raise')
    df.sort_values(by=['date'], ascending=[True], inplace=True)
    df['date'] = pd.to_datetime(df['date'])
    return df

def read_Canada_vaccine_data(infile):
    df = pd.read_csv(infile)
    df.rename(columns={'prename': 'prname', 'week_end': 'date'}, inplace=True, errors='raise')
    df.sort_values(by=['prname', 'date'], ascending=[True, True], inplace=True)
    df['date'] = pd.to_datetime(df['date'])
    df['proptotal_additional'] = df['proptotal_additional'].fillna(0)
    return df

def plot_Canada_data(df, df_hosp, death_fac=20, Nmax=None):

    # Parameters
    dpi = 200
    bar_width = 1.
    bar_alpha = 1.
    nrow = 1
    start_date = date(2020, 8, 1)
    end_date = max(df['date'])+relativedelta(months=3)
    ylabel = 'Cases/hospital occupation each day'
    y2label = 'Deaths per day'
    y2fac = death_fac
    titx = 0.015; tity = 0.86
    figx = 10.; figy = 4.
    use_seaborn = True
    province = 'Canada'

    # Seaborn
    if use_seaborn:
        sns.set_theme(style='ticks')
    else:
        sns.reset_orig
        matplotlib.rc_file_defaults()

    # Make plot
    plt.subplots(nrow, 1, figsize=(figx,figy*nrow), dpi=dpi)
    ax = plt.subplot(nrow, 1, 1)
    plot_month_spans(plt)
    plot_lockdown_spans(plt, df, province)

    # Bars
    dfs = [df[df['prname']==province], df_hosp, df_hosp, df[df['prname']==province]]
    things = ['numtoday', 'COVID_HOSP', 'COVID_OTHER', 'numdeathstoday']
    facs = [1., 1., 1., death_fac]
    colors = [case_bar_color, icu_bar_color, hosp_bar_color, death_bar_color]
    labels = [None, 'ICU', 'Hospitalised', None]
    for dg, thing, fac, color, label in zip(dfs, things, facs, colors, labels):
        plt.bar(
            x=dg['date'], 
            height=dg[thing]*fac, 
            width=bar_width, 
            color=color, 
            linewidth=0., 
            alpha=bar_alpha,
            label=label
            )

    # Lines
    things = ['numtotal_last7', 'numdeaths_last7']
    facs = [1., death_fac]
    colors = [case_line_color, death_line_color]
    labels = ['Cases', 'Deaths']
    for thing, fac, color, label in zip(things, facs, colors, labels):
        plt.plot(
            df[df['prname']==province]['date']-dt.timedelta(rolling_offset), 
            df[df['prname']==province][thing]*fac, 
            color=color, 
            label=label,
            )

    # Axis stuff
    plt.xlabel(None)
    plt.ylabel(ylabel)
    ax.get_yaxis().set_major_formatter(ticker.FuncFormatter(lambda x, _: format(int(x), ',')))
    ax2 = ax.secondary_yaxis('right', functions=(lambda y: y/y2fac, lambda y: y*y2fac))
    ax2.set_ylabel(y2label)
    ax2.get_yaxis().set_major_formatter(ticker.FuncFormatter(lambda x, _: format(int(x), ',')))
    plt.ylim((0., Nmax))
    plt.xlim((start_date, end_date))
    date_label = max(df['date'])
    title = province+'\n%s'%(date_label.strftime("%Y-%m-%d"))
    plt.title(title, x=titx, y=tity, loc='left', fontsize='small', bbox=dict(facecolor='w', edgecolor='k'))
    plt.legend(loc='upper right')

    # Sort x-axis with month data
    X = plt.gca().xaxis
    X.set_major_locator(dates.MonthLocator())
    X.set_minor_locator(dates.MonthLocator(bymonthday=15))
    X.set_major_formatter(ticker.NullFormatter())
    fmt = dates.DateFormatter('%b') # Specify the format - %b gives us Jan, Feb...
    X.set_minor_formatter(fmt)
    plt.tick_params(axis='x', which='minor', bottom=False, labelbottom=True)
    plt.xlabel(None)

    # Finalize
    plt.tight_layout()

def plot_Canadian_provinces_data(df, provinces, Nmax=100):

    # Parameters
    dpi = 200
    bar_width = 1.
    bar_alpha = 1.
    nrow = len(provinces)
    pop = population_Canadian_provinces
    start_date = date(2020, 3, 1)
    end_date = max(df['date'])+relativedelta(months=2)
    ylabel = 'Total number per day per 100,000 population'
    titx = 0.01
    figx = 18.; figy = 2.
    use_seaborn = True

    # Seaborn
    if use_seaborn:
        sns.set_theme(style='ticks')
    else:
        sns.reset_orig
        matplotlib.rc_file_defaults()

    # Make plot
    fig, _ = plt.subplots(nrow, 1, figsize=(figx,figy*nrow), dpi=dpi)

    for iplot, province in enumerate(provinces):

        # Initialise each individual subplot
        ax = plt.subplot(nrow, 1, iplot+1)
        plot_month_spans(plt)
        plot_lockdown_spans(plt, df, province)

        # Bar plots
        things = ['numtoday', 'numdeathstoday']
        facs = [pop_norm_num/pop[province], death_fac*pop_norm_num/pop[province]]
        colors = [case_bar_color, death_bar_color]
        for thing, fac, color in zip(things, facs, colors):
            plt.bar(
                x=df[df['prname']==province]['date'], 
                height=df[df['prname']==province][thing]*fac, 
                width=bar_width, 
                color=color, 
                linewidth=0., 
                alpha=bar_alpha,
                )

        # Line plots
        things = ['numtotal_last7', 'numdeaths_last7']
        facs = [pop_norm_num/pop[province], death_fac*pop_norm_num/pop[province]]
        colors = [case_line_color, death_line_color]
        labels = [case_label, death_label]
        for thing, fac, color, label in zip(things, facs, colors, labels):
            plt.plot(
                df[df['prname']==province]['date']-dt.timedelta(rolling_offset), 
                df[df['prname']==province][thing]*fac, 
                color=color, 
                label=label,
                )

        # Axis stuff
        plt.xlabel(None)
        plt.ylim((0., Nmax))
        plt.xlim((start_date, end_date))
        if iplot == 0:
            date_label = max(df['date'])
            title = province+'\n%s'%(date_label.strftime("%Y-%m-%d"))
            tity = 0.70
        else:
            title = province
            tity = 0.80
        plt.title(title, x=titx, y=tity, loc='left', fontsize='small', bbox=dict(facecolor='w', edgecolor='k'))
        if iplot == 0: plt.legend(loc='upper right')

        # Sort x-axis with month data
        X = plt.gca().xaxis
        X.set_major_locator(dates.MonthLocator())
        X.set_minor_locator(dates.MonthLocator(bymonthday=15))
        X.set_major_formatter(ticker.NullFormatter())
        if iplot == nrow-1:
            fmt = dates.DateFormatter('%b') # Specify the format - %b gives us Jan, Feb...
        else:
            fmt = ticker.NullFormatter()
        X.set_minor_formatter(fmt)
        plt.tick_params(axis='x', which='minor', bottom=False, labelbottom=True)
        plt.xlabel(None)

    # One big y-axis label
    ax = fig.add_subplot(111, frameon=False)
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    ax.set_ylabel(ylabel)

    # Finalize
    plt.tight_layout()

def plot_Canadian_province_data(df, df_vax=None, province='British Columbia', Nmax=5750, 
    fake_hosp=False, fake_hosp_fac=0.01):

    # Parameters
    dpi = 200
    bar_width = 1.
    bar_alpha = 1.
    #start_date = date(2020, 3, 1)
    start_date = date(2020, 8, 1)
    end_date = max(df['date'])+relativedelta(months=1)
    y1label = 'New cases per day'
    #y2label = 'New deaths/hospitalisations per day'
    y2label = 'New deaths per day'
    y2fac = 50
    titx = 0.015; tity = 0.63
    figx = 10.; figy = 4.
    use_seaborn = True
    rolling_offset = 3 # Number of days to offset rolling data
    restriction_alpha = 0.08
    restriction_max = 5000./Nmax
    restriction_color = 'black'
    plot_waves = True
    plot_restrictions = True
    plot_vax = True

    # Cases plot
    case_bar_color = 'cornflowerblue'
    case_line_color = 'b'
    case_label = 'Cases'

    # Hospitalisations plot
    hosp_fac = y2fac
    hosp_bar_color = 'g'
    hosp_line_color = 'forestgreen'
    hosp_label = 'Hospitalizations'

    # Deaths plot
    death_fac = y2fac
    death_bar_color = 'indianred'
    death_line_color = 'r'
    death_label = 'Deaths'

    # Vaccines
    one_dose_color = 'limegreen'
    two_dose_color = one_dose_color
    thr_dose_color = two_dose_color
    one_dose_max = 1.00
    one_dose_min = restriction_max+(1.-restriction_max)*(2./3.)
    two_dose_max = one_dose_min
    two_dose_min = restriction_max+(1.-restriction_max)*(1./3.)
    thr_dose_max = two_dose_min
    thr_dose_min = restriction_max
    alpha_vax = 0.018

    # Seaborn
    if use_seaborn:
        sns.set_theme(style='ticks')
    else:
        sns.reset_orig
        matplotlib.rc_file_defaults()

    # Make plot
    plt.subplots(1, 1, figsize=(figx,figy), dpi=dpi, sharex=True)

    ax = plt.subplot(1, 1, 1)
    if not plot_restrictions:
        plot_month_spans(plt, alpha=0.025)
    plot_lockdown_spans(plt, df, province)

    # Bars
    things = ['numtoday', 'numtoday', 'numdeathstoday']
    facs = [1., hosp_fac*fake_hosp_fac, death_fac]
    colors = [case_bar_color, hosp_bar_color, death_bar_color]
    plots = [True, fake_hosp, True]
    for thing, fac, color, plot in zip(things, facs, colors, plots):
        if plot:
            plt.bar(
                x=df[df['prname']==province]['date'], 
                height=df[df['prname']==province][thing]*fac, 
                width=bar_width, 
                color=color, 
                linewidth=0., 
                alpha=bar_alpha,
                )

    # Lines
    things = ['numtotal_last7', 'numtotal_last7', 'numdeaths_last7']
    facs = [1., hosp_fac*fake_hosp_fac, death_fac]
    colors = [case_line_color, hosp_line_color, death_line_color]
    labels = [case_label, hosp_label, death_label]
    plots = [True, fake_hosp, True]
    for thing, fac, color, label, plot in zip(things, facs, colors, labels, plots):
        if plot:
            plt.plot(
                df[df['prname']==province]['date']-dt.timedelta(rolling_offset), 
                df[df['prname']==province][thing]*fac, 
                color=color, 
                label=label,
                )

    if province == 'British Columbia':

        # Special dates
        special_dates = {
            'masks mandated': date(2020, 11, 19),
            'masks mandated again': date(2021, 8, 25),
            'testing capacity breached': date(2021, 12, 24),
        }
        dirs = ['right', 'left', 'left']
        yposes = [0.6, 0.5, 0.7]
        for label, special_date, dir, ypos in zip(special_dates.keys(), special_dates.values(), dirs, yposes):
            plt.axvline(special_date, color='k', ls='--', lw=0.5)
            if dir == 'left':
                plt.text(special_date+relativedelta(days=-2), ypos*Nmax, label+' →', size='small', ha='right')#, transform=ax.transAxes)
            else:
                plt.text(special_date+relativedelta(days=+2), ypos*Nmax, '← '+label, size='small', ha='left')#, transform=ax.transAxes)

        # Waves
        if plot_waves:
            waves = ['alpha', 'delta', 'omicron']
            days = [date(2021, 4, 10), date(2021, 9, 9), date(2022, 1, 16)]
            yposs = [0.29, 0.2, 0.80]
            for wave, day, ypos in zip(waves, days, yposs):
                plt.text(day, ypos*Nmax, '|← '+wave+' →|', ha='center', va='center')

        # Restrictions
        if plot_restrictions:
            restrictions = {
                'school closures':{
                    (date(2020, 3, 15), date(2020, 6, 1)),  # Initial March 2020 school closure
                },
                'mask mandate':{
                    (date(2020, 11, 19), date(2021, 7, 1)), # Masks first madated indoors in November 2020
                    (date(2021, 8, 25), date(2022, 3, 11)), # Brief period in July/August 2021 where mask mandate was dropped
                },
                'gym closures':{
                    (date(2020, 3, 21), date(2020, 5, 19)),  # Initial March 2020 gym closures
                    (date(2020, 10, 19), date(2021, 5, 25)), # Step 1 of restart BC (TODO: Start date)
                    (date(2021, 12, 22), date(2022, 1, 18)), # Omicron gym closures
                },
                'indoor gatherings limited':{ 
                    (date(2020, 10, 19), date(2021, 6, 15)), # Step 2 of restart BC
                    (date(2021, 12, 22), date(2022, 2, 16)), # Gatherings limited  because of Omicron
                },
                'indoor gatherings banned':{
                    (date(2020, 11, 8), date(2020, 11, 23)), # Winter 2020 firebreak lockdown
                    (date(2020, 10, 19), date(2021, 5, 25)), # Step 1 of restart BC (TODO: Start date)
                },
                'vaccine passport':{
                    (date(2021, 9, 13), date(2022, 4, 8)),   # Vaccine passports introduced September 2021
                }
            }
            add_label=True
            for _, date_pairs in restrictions.items():
                if add_label:
                    label='Restrictions'
                    add_label=False
                else:
                    label=None
                for date_pair in date_pairs:
                    plt.axvspan(date_pair[0], date_pair[1], ymax=restriction_max,
                        alpha=restriction_alpha, color=restriction_color, lw=0., label=label)

        # Vaccines
        if plot_vax and df_vax is not None:
            ax.plot([0.,1.], [restriction_max,restriction_max], transform=ax.transAxes, lw=0.5, color='black')
            dg = df_vax[df_vax['prname']==province]
            dg.reset_index(level=None, drop=False, inplace=True)
            for i, day in enumerate(dg['date']):
                start_day = day
                try: 
                    end_day = dg.at[i+1, 'date']
                except:
                    end_day = end_date
                for _ in range(int(dg.at[i, 'proptotal_atleast1dose'])):
                    plt.axvspan(start_day, end_day, ymin=one_dose_min, ymax=one_dose_max, 
                        color=one_dose_color, alpha=alpha_vax, lw=0.)
                for _ in range(int(dg.at[i, 'proptotal_fully'])):
                    plt.axvspan(start_day, end_day, ymin=two_dose_min, ymax=two_dose_max, 
                        color=two_dose_color, alpha=alpha_vax, lw=0.)
                for _ in range(int(dg.at[i, 'proptotal_additional'])):
                    plt.axvspan(start_day, end_day, ymin=thr_dose_min, ymax=thr_dose_max, 
                        color=thr_dose_color, alpha=alpha_vax, lw=0.)
            labels = [
                'one dose: {:.0f}%'.format(dg.at[i, 'proptotal_atleast1dose']),
                'two doses: {:.0f}%'.format(dg.at[i, 'proptotal_fully']),
                'booster: {:.0f}%'.format(dg.at[i, 'proptotal_additional']),
            ]
            yposs = [one_dose_min, two_dose_min, thr_dose_min]
            for ypos, label in zip(yposs, labels):
                plt.text(0.89, ypos, label, va='bottom', size='x-small', transform=ax.transAxes)

    plt.xlabel(None)
    plt.ylim((0., Nmax))
    plt.xlim((start_date, end_date))
    date_label = max(df['date'])
    title = province+'\n%s'%(date_label.strftime("%Y-%m-%d"))
    plt.title(title, x=titx, y=tity, loc='left', fontsize='small', bbox=dict(facecolor='w', edgecolor='k'))
    plt.legend(loc='upper left', framealpha=1.)

    # Sort x-axis with month data
    X = plt.gca().xaxis
    X.set_major_locator(dates.MonthLocator())
    X.set_minor_locator(dates.MonthLocator(bymonthday=15))
    X.set_major_formatter(ticker.NullFormatter())
    fmt = dates.DateFormatter('%b') # Specify the format - %b gives us Jan, Feb...
    X.set_minor_formatter(fmt)
    plt.tick_params(axis='x', which='minor', bottom=False, labelbottom=True)
    plt.xlabel(None)

    # Primary y axis
    ax.set_ylabel(y1label)
    ax.get_yaxis().set_major_formatter(ticker.FuncFormatter(lambda x, _: format(int(x), ',')))

    # Secondary y axis
    ax2 = ax.secondary_yaxis('right', functions=(lambda y: y/y2fac, lambda y: y*y2fac))
    ax2.set_ylabel(y2label)

    # Finalize
    plt.tight_layout()

### ###

### JHU data ###

def read_JHU_data(file):
    '''
    Read and organise JHU data downloaded from https://github.com/CSSEGISandData/COVID-19
    '''
    df = pd.read_csv(file)
    df.drop(['Province/State', 'Lat', 'Long'], axis='columns', inplace=True) # Remove unnecessary columns
    df.rename(columns={'Country/Region':'Region'}, inplace=True) # Rename for convenience
    df = df.groupby(['Region']).sum() # Combine provinces that are part of the same countries
    df.columns = pd.to_datetime(df.columns) # Convert entries to datetime objects
    df.rename(index={'US':'United States', 'UK':'United Kingdom'}, inplace=True)
    return df

def calculate_JHU_daily_from_original(df):
    '''
    Calculate daily data from original cumulative data
    '''
    df_new = df.diff(axis='columns').copy() # Convert from cumulative to daily numbers
    return df_new

def calculate_JHU_rolling_from_daily(df):
    '''
    Calculate rolling average from daily data
    '''
    center = True
    df_new = df.rolling(7, min_periods=1, axis='columns', center=center).sum()/7. # Average over week
    if center: # Remove final three entries that will be anomalously low if centring
        for i in range(-3, 0):
            df_new[df_new.columns[i]] = np.nan
    return df_new

def calculate_JHU_rolling_from_original(df):
    '''
    Calculate rolling average from original cumulative data
    '''
    df1 = calculate_JHU_daily_from_original(df)
    df2 = calculate_JHU_rolling_from_daily(df1)
    return df2

def plot_UK_deaths(df_daily, df_rolling):
    '''
    Plot the UK death rate from COVID-19 over the course of the pandemic.
    Also add information about flu deaths and total deaths.
    '''
    # Parameters
    deaths = { # Comparable deaths
        'All-cause deaths in a typical year': 530841./365., # 2019 total averaged over 365 days
        'Flu deaths in a typical flu year': 15000./365., # Total averaged over year
        'Flu deaths at height of a bad flu year': 4.*25000./365., # Compress total over 3 months (shoddy)
    }
    country = 'United Kingdom'
    bar_alpha = 1.
    dpi = 200

    # Plot
    _, ax = plt.subplots(figsize=(18.,4.), dpi=dpi)
    sort_month_axis(plt)
    plot_month_spans(plt, alpha=0.025)
    plot_lockdown_spans(plt, df_daily, region=country, label='Lockdowns')
    sns.lineplot(data=df_rolling.loc[country], label='UK COVID-19 deaths', color=death_line_color)
    plt.bar(df_daily.loc[country].index, df_daily.loc[country].values, width=1., color=death_bar_color, linewidth=0., alpha=bar_alpha)
    plt.ylabel('Deaths per day during pandemic')
    plt.xlabel('')
    plt.ylim(bottom=0., top=1500)
    plt.xlim([date(2020, 1, 1), max(df_rolling.columns)+relativedelta(months=6)])
    for i, death in enumerate(deaths.keys()):
        if i == 0:
            delta = -85.
        else:
            delta = 40.
        plt.axhline(deaths[death], color='black', alpha=0.8, ls=':')
        plt.text(max(df_rolling.columns)+relativedelta(months=1), deaths[death]+delta, death)
    handles, _ = ax.get_legend_handles_labels()
    patch = mpatches.Patch(alpha=0., label=dt.date.today().strftime("%Y-%m-%d"))
    handles.insert(0, patch) 
    plt.legend(handles=handles, loc='upper left', framealpha=1., edgecolor='k')
    ax.get_yaxis().set_major_formatter(ticker.FuncFormatter(lambda x, _: format(int(x), ',')))
    #plt.title(dt.date.today().strftime("%Y-%m-%d"), x=0.19, y=0.88, loc='Center', bbox=dict(facecolor='w', edgecolor='k'))
    plt.tight_layout()

def plot_world(df, countries, dftype='deaths'):
    ''' 
    World total and rate of cases/deaths in different countries.
    '''
    per = 1e6
    alpha_month = 0.025
    figx = 18.; figy = 8.
    titx = 0.015; tity = 0.87
    dpi = 200

    plt.subplots(2, 1, figsize=(figx,figy), dpi=dpi)

    for iplot in mead.mrange(1, 2):

        if iplot == 1:
            dg = df
            ylab = 'Total '+dftype+' per '+mead.number_name(per)+' population'
        else:
            dg = calculate_JHU_rolling_from_original(df)
            ylab = dftype.capitalize()+' per day per '+mead.number_name(per)+' population'

        ax = plt.subplot(2, 1, iplot)
        plot_month_spans(plt, alpha=alpha_month)
        for country in countries:
            sns.lineplot(data=dg.loc[country]*per/population_countries[country], label=country)
        plt.ylabel(ylab)
        plt.xlabel(None)
        plt.ylim(bottom=0.)
        plt.xlim([date(2020, 1, 1), max(df.columns)+relativedelta(months=3)])
        _, ymax = ax.get_ylim()
        if ymax >= 1e3:
            ax.get_yaxis().set_major_formatter(ticker.FuncFormatter(lambda x, _: format(int(x), ',')))
        if iplot == 1:
            plt.legend(loc='upper right', edgecolor='k')
            date_title = max(df.columns) # Not sure why this works, but it does
            title = dftype.capitalize()+': '+date_title.strftime("%Y-%m-%d")
            plt.title(title, x=titx, y=tity, loc='Left', bbox=dict(facecolor='w', edgecolor='k'))
            xfmt = ticker.NullFormatter()
        else:
            ax.get_legend().remove()
            sort_month_axis(plt)
            xfmt = dates.DateFormatter('%b') # Specify the format - %b gives us Jan, Feb...
        X = plt.gca().xaxis
        X.set_major_locator(dates.MonthLocator())
        X.set_minor_locator(dates.MonthLocator(bymonthday=15))
        X.set_major_formatter(ticker.NullFormatter())
        X.set_minor_formatter(xfmt)
        plt.tick_params(axis='x', which='minor', bottom=False, labelbottom=True)
        plt.xlabel(None)

    plt.tight_layout()

def plot_world_deaths_per_case(df_deaths, df_cases, countries):
    '''
    World deaths per case in different countries.
    '''
    days_offset = 21
    df_cases_offset = df_cases.copy()
    df_cases_offset.columns = df_cases.columns+pd.DateOffset(days_offset)
    df = df_deaths/df_cases_offset
    plt.figure(figsize=(18.,4.))
    sort_month_axis(plt)
    plot_month_spans(plt, alpha=0.025)
    for country in countries:
        sns.lineplot(data=df.loc[country], label=country)
    plt.ylabel('Deaths per confirmed case')
    plt.xlabel('')
    plt.yscale('log')
    plt.ylim((1e-3, 1.))
    plt.xlim([date(2020, 1, 1), max(df.columns)+relativedelta(months=3)])
    plt.legend(loc='upper right', edgecolor='k')
    plt.title(dt.date.today().strftime("%Y-%m-%d"), x=0.015, y=0.88, loc='Left', bbox=dict(facecolor='w', edgecolor='k'))
    plt.tight_layout()

### ###

### UK data ###

def download_data(area, metrics, verbose=True):
    '''
    Download latest UK government data
    '''
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
    for col in ['Cases', 'Hosp', 'Beds', 'Deaths']:
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
    locator_monthstart = dates.MonthLocator() # Start of every month
    locator_monthmid = dates.MonthLocator(bymonthday=15) # Middle of every month
    fmt = dates.DateFormatter('%b') # Specify the format - %b gives us Jan, Feb...
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
            plt.axvspan(dt.date(year, month, 1), 
                        dt.date(year, month, 1)+relativedelta(months=+1), 
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

def plot_bar_data(df, start_date, end_date, regions, 
    outfile=None, pop_norm=True, Nmax=None, plot_type='Square', legend_location='left',
    hosp_fac=hosp_fac, death_fac=death_fac, 
    dpi=200, ylabel=None, y2label=None, y2axis=False):
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
    # Parameters
    days_roll = days_in_roll
    pop_num = pop_norm_num
    bar_width = 1.
    use_seaborn = True
    date = max(df['date'])
    bar_alpha = 1.
    deaths = { # Comparable deaths
        #'All-cause deaths in a typical year': 530841./365., # 2019 total averaged over 365 days
        #'Flu deaths in a typical flu year': 15000./365., # Total averaged over year
        #'Flu deaths at height of a bad flu year': 4.*25000./365., # Compress total over 3 months (shoddy)
    }
    case_label = 'Cases'
    if y2axis:
        hosp_label = 'Hospitalisations'
    else:
        hosp_label = r'Hospitalisations $[\times %d]$'%(int(hosp_fac))
    if y2axis:
        death_label = 'Deaths'
    else:
        death_label = r'Deaths $[\times %d]$'%(int(death_fac))

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
        for col in ['Cases', 'Hosp', 'Beds', 'Deaths']:
            if col in df:
                if col == 'Cases':
                    fac = pop_fac
                    bar_color = case_bar_color
                elif col == 'Hosp':
                    fac = pop_fac*hosp_fac
                    bar_color = hosp_bar_color
                elif col == 'Beds':
                    fac = pop_fac
                    bar_color = bed_bar_color
                elif col == 'Deaths':
                    fac = pop_fac*death_fac
                    bar_color = death_bar_color
                else:
                    raise ValueError('Something went wrong')
                plt.bar(df[df['Region']==region]['date'],
                        df[df['Region']==region][col]*fac,
                        width=bar_width,
                        color=bar_color,
                        linewidth=0.,
                        alpha=bar_alpha,
                       )

        # Lines for weekly average
        for col in ['Cases_roll', 'Hosp_roll', 'Beds_roll', 'Deaths_roll']:
            if col in df:
                if col == 'Cases_roll':
                    fac = pop_fac/days_roll
                    line_color = case_line_color
                    line_label = case_label
                elif col == 'Hosp_roll':
                    fac = pop_fac*hosp_fac/days_roll
                    line_color = hosp_line_color
                    line_label = hosp_label
                elif col == 'Beds_roll':
                    fac = pop_fac/days_roll
                    line_color = bed_line_color
                    line_label = bed_label
                elif col == 'Deaths_roll':
                    fac = pop_fac*death_fac/days_roll
                    line_color = death_line_color
                    line_label = death_label
                else:
                    raise ValueError('Something went wrong')
                plt.plot(pd.to_datetime(df[df['Region']==region]['date'])-dt.timedelta(rolling_offset),
                         df[df['Region']==region][col]*fac,
                         color=line_color, 
                         label=line_label
                        )

        if region=='United Kingdom' and pop_norm==False:
            for i, death in enumerate(deaths.keys()):
                delta = 50.
                plt.axhline(deaths[death]*death_fac, color='black', alpha=0.8, ls=':')
                plt.text(max(df['date'])+relativedelta(days=15), (deaths[death]+delta)*death_fac, death)

        # Ticks and month arragement on x axis
        sort_month_axis(plt)

        # Axes limits
        plt.xlim(left=start_date, right=end_date)
        plt.ylim(bottom=0)
        if Nmax != None: 
            plt.ylim(top=Nmax)
        if pop_norm:
            if y2axis:
                raise ValueError('You need to code up y2 if pop_norm is active')
            if ylabel is None: ylabel = 'Number per day per 100,000 population'
        else:
            if y2axis:
                if ylabel is None: ylabel = 'New cases per day'
                if y2label is None: y2label = 'Deaths/Hospitalisations per day'
            else:
                if ylabel is None: ylabel = 'Number per day'
        plt.ylabel(ylabel)

        # Title that is region
        if n == 1:
            title = region+'\n%s'%(date.strftime("%Y-%m-%d"))
            ytit = 0.85
        else:
            title = region
            if n == 4:
                ytit = 0.93
            elif n == 9:
                ytit = 0.895
        if plot_type == 'Elongated':
            dx = 0.01
        else:
            dx = 0.02
        loc = mead.opposite_side(legend_location)
        if loc == 'right':
            xtit = 1.-dx
        else:
            xtit = dx
        plt.title(title, x=xtit, y=ytit, loc=loc, bbox=dict(facecolor='w', edgecolor='k'))

        # Title that is the date of the plot
        if (n == 4 or n == 9) and i == 0:
            if legend_location == 'right':
                xtit = 0.99
            elif legend_location == 'left':
                xtit = 0.01
            else:
                raise ValueError('legend location not recognised')
            if legend_location == 'right':
                xtit = 1.-dx
            else:
                xtit = dx
            if n == 4:
                ytit = 0.93
            elif n == 9:
                ytit = 0.89
            title = date.strftime("%Y-%m-%d")
            plt.title(title, x=xtit, y=ytit, loc=legend_location, bbox=dict(facecolor='w', edgecolor='k'))

        # Legend
        if (n == 9 and i == 2) or (n == 4 and i == 1) or n==1:
            legend = plt.legend(loc='upper '+legend_location, framealpha=1.)
            legend.get_frame().set_edgecolor('k')

        # Commas on y axis for long numbers
        if not pop_norm: 
            axs.get_yaxis().set_major_formatter(ticker.FuncFormatter(lambda x, _: format(int(x), ',')))

        # Secondary y axis
        if y2axis:
            y2fac = death_fac
            ax2 = axs.secondary_yaxis('right', functions=(lambda y: y/y2fac, lambda y: y*y2fac))
            ax2.set_ylabel(y2label)
            ax2.get_yaxis().set_major_formatter(ticker.FuncFormatter(lambda x, _: format(int(x), ',')))

    #if n != 1: plt.tight_layout()
    plt.tight_layout()
    if outfile != None: plt.savefig(outfile)
    plt.show(block=False)

def plot_rolling_data(df, start_date, end_date, regions, pop_norm=True, plot_type='Cases', log=True):
    '''
    Plot rolling daily data to directly compare region-to-region
    '''
    # Parameters
    days_roll = days_in_roll
    pop_num = pop_norm_num
    use_seaborn = True
    lockdown_region = 'United Kingdom'
    plot_lockdowns = True
    plot_months = True
    figx = 17.; figy = 6.
    date = max(df['date'])
    dpi = 200

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
    plt.subplots(figsize=(figx, figy), dpi=dpi)
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
        plt.title('Daily new '+plot_type.lower()+': %s'%(date.strftime("%Y-%m-%d")))

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
    '''
    Plot doubling-time (or halving time) data to directly compare region-to-region
    '''
    # Parameters
    use_seaborn = True
    lockdown_region = 'United Kingdom'
    plot_lockdowns = True
    plot_months = True
    figx = 17.; figy = 6.
    ymin = 0.; ymax = 30.
    date = max(df['date'])
    dpi = 200

    ### Figure options ###

    # Seaborn
    if use_seaborn:
        sns.set_theme(style='ticks')
    else:
        sns.reset_orig
        matplotlib.rc_file_defaults()

    # Plot
    plt.subplots(figsize=(figx, figy), dpi=dpi)
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
    '''
    Plot rolling daily data to directly compare region-to-region
    '''
    # Parameters
    use_seaborn = True
    lockdown_region = 'United Kingdom'
    figx = 17.; figy = 6.
    plot_lockdowns = True
    plot_months = True
    Rmin = 0.; Rmax = 2.
    date = max(df['date'])
    dpi = 200

    ### Figure options ###

    # Seaborn
    if use_seaborn:
        sns.set_theme(style='ticks')
    else:
        sns.reset_orig
        matplotlib.rc_file_defaults()

    # Plot
    plt.subplots(figsize=(figx, figy), dpi=dpi)
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

### ###