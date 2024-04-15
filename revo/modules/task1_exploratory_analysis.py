import pandas as pd
import matplotlib.pyplot as plt

def task1_exploratory_analysis():
    
    # Import cleaned data
    df=pd.read_csv('output/0_clean_arff/df_task1.csv')
    
    # Line plot grouped by country

    # Group the data by 'Year', 'Quarter', and 'Country' and calculate the average for each group
    df_plot1 = df[['Country','Year','Quarter','Net profit (n)/net profit (n−1)']]
    df_plot1 = df_plot1.groupby(['Year', 'Quarter', 'Country']).mean().reset_index()

    # Create a line plot for each country
    for country, data in df_plot1.groupby('Country'):
            plt.plot(data['Year'] + (data['Quarter'] - 1) / 4, data['Net profit (n)/net profit (n−1)'], label=country)

    # Add labels and legend
    plt.xlabel('Year and Quarter')
    plt.ylabel('Net profit (n)/net profit (n−1)')
    plt.legend()

    # Show plot
    plt.show()
    # Save the plot to a file
    plt.savefig('output/1_task1_exploratory_analysis/Line plot grouped by country.png')

    # Line plot grouped by sector

    # Group the data by 'Year', 'Quarter', and 'description_sector' and calculate the average for each group
    df_plot2 = df[['description_sector','Year','Quarter','Net profit (n)/net profit (n−1)']]
    df_plot2 = df_plot2.groupby(['Year', 'Quarter', 'description_sector']).mean().reset_index()

    # Create a line plot for each description_sector
    for description_sector, data in df_plot2.groupby('description_sector'):
            plt.plot(data['Year'] + (data['Quarter'] - 1) / 4, data['Net profit (n)/net profit (n−1)'], label=description_sector)

    # Add labels and legend
    plt.xlabel('Year and Quarter')
    plt.ylabel('Net profit (n)/net profit (n−1)')
    plt.legend()

    # Show plot
    plt.show()
    # Save the plot to a file
    plt.savefig('output/1_task1_exploratory_analysis/Line plot grouped by sector.png')
    
    print('\nExploratory analysis plots saved in output/1_task1_exploratory_analysis')
    