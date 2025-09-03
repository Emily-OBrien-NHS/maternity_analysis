import os
import re
import folium
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from folium.plugins import HeatMap
from sqlalchemy import create_engine
os.chdir('C:/Users/obriene/Projects/General Analysis/Maternity')


sdmart_engine = create_engine('mssql+pyodbc://@SDMartDataLive2/InfoDB?'\
                            'trusted_connection=yes&driver=ODBC+Driver+17'\
                                '+for+SQL+Server')

deliveries_query = """
SELECT *
  FROM [InfoDB].[dbo].[RL_Maternity_upload_All]
  WHERE [Birth Order_New] = 1 AND [Delivery Date / Time] > '31-07-2024'
  --and [Hospital Delivery Site] = 'Derriford Hospital' --if only wanting births with UHP, not patients who've given birth elsewhere and come to us
"""
deliveries = pd.read_sql(deliveries_query, sdmart_engine)

antinatal_query = """
SELECT *
  FROM [InfoDB].[dbo].[RL_Antinatal_Test]
  WHERE [Boooking Date] > '31-07-2024'
"""
antinatal = pd.read_sql(antinatal_query, sdmart_engine)

IMD_query = """SELECT PostcodeFormatted, LowerSuperOutputArea2011, IndexValue
FROM [PiMSMarts].[Reference].[vw_IndicesOfMultipleDeprivation2019_DecileByPostcode]
"""
IMD = pd.read_sql(IMD_query, sdmart_engine)
IMD['pcode'] = IMD['PostcodeFormatted'].str.replace('  ', ' ')


pcode_LL = pd.read_csv('C:/Users/obriene/Projects/Mapping/ukpostcodes.csv').rename(columns={'postcode':'FullPostCode'})

##################################Patients######################################
current_patients = pd.read_excel('Patient under the care of Maternity - 01_08_2024 to 31_07_2025.xlsx')
#Remove duplicates by keeping most recent EDD row
current_patients = current_patients.sort_values(by='EDD', ascending=False)
current_patients = current_patients.groupby('NHS Number').first()

#Extract postcodes with regex
current_patients['pcode'] = [i[0].replace('  ', ' ') if len(i) > 0 else 'ZZ99 9ZZ'
                             for i in current_patients['Address'].str.findall(
                                 '[A-Z]{1,2}[0-9R][0-9A-Z]? ?[0-9][A-Z]{2}')]
#Add spaces if none in original postce
current_patients['pcode'] = [i[:-3] + ' ' + i[-3:] if ' ' not in i else i
                             for i in current_patients['pcode'].to_list()]
#Join onto lat long data
current_patients = current_patients.merge(pcode_LL, left_on='pcode',
                                          right_on='FullPostCode', how='left')
#Join onto IMD data
current_patients = current_patients.merge(IMD, on='pcode', how='left')

##############################Clinc Locations###################################
clinics = pd.read_excel('Clinic Locations.xlsx')
clinics = clinics.merge(pcode_LL, left_on='pcode', right_on='FullPostCode', how='left')

##############################Team Postcodes####################################
teams = pd.read_excel('Team Locations.xlsx')
current_patients['pcode stem'] = current_patients['pcode'].str.split(' ').apply(lambda x: x[0])
current_patients = current_patients.merge(teams, on='pcode stem', how='left')
current_patients['Team'] = current_patients['Team'].fillna('Other')

############################Heatmap Functions###################################
def create_heatmap(DataFrame, name):
  #Group data to get count by postcode/latlong
  heat_df = (DataFrame.groupby(['pcode', 'latitude', 'longitude'],
                               as_index=False) ['Hospital Patient ID'].count()
                      .rename(columns={'Hospital Patient ID'
                                       :'Number of Patients'}))
  #heatmap
  m = folium.Map(location=[50.4163,-4.11849], zoom_start=10,
                 tiles="cartodbpositron")
  #Repeat by number of patients in each pcode
  heat_df = heat_df.loc[heat_df.index.repeat(heat_df['Number of Patients']),
                        ['latitude','longitude']].dropna()
  #Make a list of values
  heat_data = [[row['latitude'], row['longitude']] for index, row
               in heat_df.iterrows()]
  #create heatmap
  HeatMap(heat_data).add_to(m)

  #Add clinics as markers
  for i, row in clinics[['Site', 'Team', 'latitude', 'longitude']].iterrows():
    folium.Marker([row['latitude'], row['longitude']],
                  tooltip=f'Site: {row['Site']}, Team: {row['Team']}',
                  icon=folium.Icon(color='darkpurple', icon="h-square",
                                   prefix = 'fa')).add_to(m)
  title = "The information in this map may contain Personal Confidential Data (PCD) and must not be sent to another organisation or non-NHS.NET email address without IG consent"
  title_html = f'<h1 style="position:absolute;z-index:100000;left:40vw;color:red;font-size:160%;" >{title}</h1>'
  m.get_root().html.add_child(folium.Element(title_html))
  #save
  m.save(f"Results/{name} Maternity Heatmap.html")


#All Patients Heatmap
create_heatmap(current_patients, 'All Patients')
#IMD Heatmaps
create_heatmap(current_patients.loc[current_patients['IndexValue'].astype(float) <= 2].copy(), 'IMD 1-2')
create_heatmap(current_patients.loc[current_patients['IndexValue'].astype(float) > 2].copy(), 'IMD 3-10')


#########################IMD and Ethniciy breakdowns############################
#Format ethnicity strings and get groups
current_patients['Ethnicity'] = [i.split(' - ')[1] if i else  i for i
                                 in current_patients['Ethnicity'].tolist()]
current_patients['Ethnicity Group'] = [i.split(':')[0] if ((i) and (':' in i))
                                      else  i for i 
                                      in current_patients['Ethnicity'].tolist()]
#Detailed level plot
detailed_ethnicity = pd.DataFrame(current_patients['Ethnicity'].value_counts())
detailed_ethnicity['count'].plot(kind='barh', xlabel='Number of Patients',
                                 title='Number of Maternity Patients by Ethnicity')
plt.savefig('Results/Detailed Ethniciy.png', bbox_inches='tight')
plt.close()
#Group level plot
grouped_ethnicity = pd.DataFrame(current_patients['Ethnicity Group'].value_counts())
grouped_ethnicity['count'].plot(kind='barh', xlabel='Number of Patients',
                                title='Number of Maternity Patients by Ethnicity Group')
plt.savefig('Results/Grouped Ethniciy.png', bbox_inches='tight')
plt.close()

#IMD plot
IMD_cols = [str(i+1) for i in range(10)]
IMD = pd.DataFrame(current_patients['IndexValue'].dropna().astype(int).value_counts()).sort_values(by='IndexValue')
IMD.plot(kind='bar', ylabel='Number of Patients', title='Number of Maternity Patients by IMD')
plt.savefig('Results/IMD.png', bbox_inches='tight')
plt.close()

#IMD and detailed ethnicity
def imd_and_ethnicity_plot(eth_col):
  #Group and pivot data
  detailed_ethIMD = (current_patients.groupby([eth_col, 'IndexValue'],
                                              as_index=False)
                                              ['Hospital Patient ID'].count()
                                    .pivot(index=eth_col,
                                            columns='IndexValue',
                                            values='Hospital Patient ID')
                                            ).fillna(0)[IMD_cols].sort_values(by='1', ascending=False)
  #Get totals
  IMD_total = detailed_ethIMD.sum(axis=0)
  eth_total = detailed_ethIMD.sum(axis=1)

  #get percentages within each ethnicity
  detailed_ethIMD_perc = (detailed_ethIMD[IMD_cols].div(eth_total, axis=0).round(2)*100).astype(int)
  #Add total percentages
  eth_perc = ((eth_total / eth_total.sum()).round(2)*100).astype(int)
  IMD_perc = ((IMD_total / IMD_total.sum()).round(2)*100).astype(int)
  detailed_ethIMD_perc['Total %'] = eth_perc
  IMD_perc['Total %'] = np.nan
  IMD_perc.name = 'Total %'
  detailed_ethIMD_perc = pd.concat([detailed_ethIMD_perc, pd.DataFrame(IMD_perc).T])

  #Create dfs for 2 heatmaps, one for the totals and one for the rest of the data.
  data = detailed_ethIMD_perc.copy()
  data['Total %'] = np.nan
  data.loc['Total %'] = np.nan

  totals = detailed_ethIMD_perc.copy()
  totals.iloc[:-1, :-1] = np.nan

  #create heatmap
  fig, ax = plt.subplots(figsize=(25, 15))
  sns.set_theme(font_scale=2) 
  sns.heatmap(ax=ax, data=data, annot=True, cmap='Blues')
  sns.heatmap(ax=ax, data=totals, annot=True, cmap='Reds')
  for t in ax.texts: t.set_text(t.get_text() + "%")
  ax.set_title(f'Percentages within each {eth_col} by IMD for current Maternity Patients')
  ax.set_xticklabels(ax.get_xticklabels(), fontsize=20)
  plt.savefig(f'Results/IMD and {eth_col}.png', bbox_inches='tight')
  plt.close()

imd_and_ethnicity_plot('Ethnicity')
imd_and_ethnicity_plot('Ethnicity Group')

################################Language Data###################################
#Fill in blanks
current_patients['Interpreter Required'] = current_patients['Interpreter Required'].fillna('No')
current_patients['English is First language'] = current_patients['English is First language'].fillna('Yes')

#Filter to yes/no for if english is first language
eng_first = current_patients.loc[current_patients['English is First language'] == 'Yes'].copy()
no_eng_first = current_patients.loc[current_patients['English is First language'] == 'No'].copy()
#Filter to yes or no if interpreter is required
no_eng_no_int = no_eng_first.loc[no_eng_first['Interpreter Required'] == 'No'].copy()
no_eng_yes_int = no_eng_first.loc[no_eng_first['Interpreter Required'] == 'Yes'].copy()

#Get total number of patients and create a dictionary to plot
total = len(current_patients)
interpretor = {'Interpreter Required' : np.array([0, len(no_eng_yes_int)]),
                'No Interpreter Required' : np.array([len(eng_first), len(no_eng_no_int)])}

#Plot stacked bar based on counts
width = 0.5
fig, ax = plt.subplots(figsize=(15, 20))
bottom = np.zeros(2)
for col, count in interpretor.items():
    p = ax.bar(['English', 'Other'], count, width, label=col, bottom=bottom)
    bottom += count
ax.set_title("Number of Maternity Patients with English as a First Language")
ax.legend(loc="upper right")
ax.set_ylabel('Number of Patients')
plt.savefig(f'Results/Number of English First.png', bbox_inches='tight')
plt.close()

#Plot stacked bar based on percentage
width = 0.5
fig, ax = plt.subplots(figsize=(15, 20))
bottom = np.zeros(2)
for col, count in interpretor.items():
    perc = np.round(count/total * 100, 2)
    p = ax.bar(['English', 'Other'], perc, width, label=col, bottom=bottom)
    bottom += perc
ax.set_title("Percentage of Maternity Patients with English as a First Language")
ax.legend(loc="upper right")
ax.set_ylabel('Percentage of Patients')
plt.savefig(f'Results/Percentage of English First.png', bbox_inches='tight')
plt.close()

#For those that don't have english as first language, which languages come on top
(current_patients['First Language'].value_counts().head(20)
 .plot(kind='barh', title='Top 20 First Languages other than English',
       ylabel='Number of Patients', figsize=(20, 15)))
plt.savefig('Results/Languages - All.png', bbox_inches='tight')
plt.close()

(no_eng_yes_int['First Language'].value_counts().head(20)
 .plot(kind='barh', title='Top 20 First Languages other than English Requiring Interpreter',
       ylabel='Number of Patients', figsize=(20, 15)))
plt.savefig('Results/Languages - Interpreter Required.png', bbox_inches='tight')
plt.close()

(no_eng_no_int['First Language'].value_counts().head(20)
 .plot(kind='barh', title='Top 20 First Languages other than English Not Requiring Interpreter',
       ylabel='Number of Patients', figsize=(20, 15)))
plt.savefig('Results/Languages - No Interpreter Required.png', bbox_inches='tight')
plt.close()

##################################Team Data#####################################
#overall count
(current_patients['Team'].value_counts()
 .plot(kind='bar', title='Number of Maternity Patients per Team', xlabel='Team',
       ylabel='Count of patients', figsize=(20, 15)))
plt.savefig('Results/Number of Maternity Patients per Team.png', bbox_inches='tight')
plt.close()

#heatmap of proportion in each IMD for each team
cols = ['Greenark', 'Fourwoods', 'Nomony', 'Cornwall', 'South Hams', 'Other']
current_patients['IndexValue'] = current_patients['IndexValue'].astype(float)
IMD_teams = (current_patients.groupby(['Team', 'IndexValue'], as_index=False)
             ['Hospital Patient ID'].count()
             .pivot(index='IndexValue', columns='Team',
                    values='Hospital Patient ID').fillna(0).astype(int))[cols]
IMD_teams.index = IMD_teams.index.astype(int)

#create count heatmap
fig, ax = plt.subplots(figsize=(25, 15))
sns.set_theme(font_scale=2) 
sns.heatmap(ax=ax, data=IMD_teams, annot=True, cmap='Blues', fmt='g')
ax.set_title(f'Count within each Team by IMD for current Maternity Patients')
ax.set_xticklabels(ax.get_xticklabels(), fontsize=20)
plt.savefig(f'Results/IMD and Team Count.png', bbox_inches='tight')
plt.close()

#Convert to proportions
cols = ['Greenark', 'Fourwoods', 'Nomony', 'Cornwall', 'South Hams', 'Other']
perc = ((IMD_teams/IMD_teams.sum()) * 100).round(0).astype(int)[cols]

#create proportion heatmap
fig, ax = plt.subplots(figsize=(25, 15))
sns.set_theme(font_scale=2) 
sns.heatmap(ax=ax, data=perc, annot=True, cmap='Blues')
for t in ax.texts: t.set_text(t.get_text() + "%")
ax.set_title(f'Percentages within each Team by IMD for current Maternity Patients')
ax.set_xticklabels(ax.get_xticklabels(), fontsize=20)
plt.savefig(f'Results/IMD and Team percentage.png', bbox_inches='tight')
plt.close()

#English first language y/n
languages = current_patients.groupby(['Team', 'English is First language',
                                      'Interpreter Required'], as_index=False
                                      )['Hospital Patient ID'].count()
languages['Language Requirement'] = np.nan
languages.loc[languages['English is First language'] == 'Yes',
              'Language Requirement'] = 'English First Language'
languages.loc[(languages['English is First language'] == 'No')
              & (languages['Interpreter Required'] == 'Yes'),
              'Language Requirement'] = 'Interpreter Required'
languages['Language Requirement'] = languages['Language Requirement'
                                             ].fillna('No Interpreter Required')
languages = (languages.pivot(index='Team', columns='Language Requirement',
                            values='Hospital Patient ID').fillna(0)
                            .sort_values(by='English First Language',
                                         ascending=False)
                            [['English First Language', 'No Interpreter Required',
                              'Interpreter Required']])

values = {"English First Language": languages['English First Language'],
          "No Interpreter Required": languages['No Interpreter Required'],
          "Interpreter Required":languages['Interpreter Required']}
width = 0.5

fig, ax = plt.subplots(figsize=(20, 10))
bottom = np.zeros(len(languages))
for boolean, value in values.items():
    p = ax.bar(languages.index, value, width, label=boolean, bottom=bottom)
    bottom += value
ax.set_title("Language Requirements by Team")
ax.legend(loc="upper right")
plt.savefig(f'Results/Language Requirements by Team.png', bbox_inches='tight')
plt.close()


#languages spoken
languages = (current_patients.loc[current_patients['Interpreter Required'] == 'Yes']
             .groupby(['Team', 'First Language'], as_index=False)
             ['Hospital Patient ID'].count()
             .pivot(index='Team', #'First Language',
                    columns='First Language',#'Team',
                    values='Hospital Patient ID'))
#create heatmap
fig, ax = plt.subplots(figsize=(30, 15))
sns.set_theme(font_scale=2) 
sns.heatmap(ax=ax, data=languages, annot=True, cmap='Blues')
ax.set_title(f'Count of First Languages of Maternity Patients who require a Translator')
ax.set_xticklabels(ax.get_xticklabels(), fontsize=20)
plt.savefig(f'Results/First Language by Team.png', bbox_inches='tight')
plt.close()




#Ethnicity by team
eth = (current_patients.groupby(['Team', 'Ethnicity Group'], as_index=False)
       ['Hospital Patient ID'].count()
       .pivot(index='Ethnicity Group', columns='Team',
              values='Hospital Patient ID').fillna(0)
        [['South Hams', 'Cornwall', 'Fourwoods', 'Greenark', 'Nomony', 'Other']]
        .sort_values(by='South Hams', ascending=False))

#create count heatmap
fig, ax = plt.subplots(figsize=(25, 15))
sns.set_theme(font_scale=2) 
sns.heatmap(ax=ax, data=eth, annot=True, cmap='Blues', fmt='g')
ax.set_title(f'Count within each Team by ethnicity for current Maternity Patients')
ax.set_xticklabels(ax.get_xticklabels(), fontsize=20)
plt.savefig(f'Results/Ethnicity and Team Count.png', bbox_inches='tight')
plt.close()

#Convert to proportions
cols = ['Cornwall', 'South Hams', 'Fourwoods','Greenark','Nomony',  'Other']
perc = ((eth/eth.sum()) * 100).round(0).astype(int)[cols]

#create proportion heatmap
fig, ax = plt.subplots(figsize=(25, 15))
sns.set_theme(font_scale=2) 
sns.heatmap(ax=ax, data=perc, annot=True, cmap='Blues')
for t in ax.texts: t.set_text(t.get_text() + "%")
ax.set_title(f'Percentages within each Team by ethnicity for current Maternity Patients')
ax.set_xticklabels(ax.get_xticklabels(), fontsize=20)
plt.savefig(f'Results/Ethnicity and Team percentage.png', bbox_inches='tight')
plt.close()
