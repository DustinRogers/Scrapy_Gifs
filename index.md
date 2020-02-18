---
classes: wide
title: " "
excerpt: Using Spotipy to predict song popularity!
header:
  overlay_image: /assets/images/Header.jpg  
---

### Below is a Scrapy spider that I created to pull data from WorldoMeters.info. The first function pulls the country name and the link to that countries website. The second function loops through each countrys' website and takes infomation out a table on that site. The bottom 'yeild' function aggregates the infomation and I export that to a csv.


```python
# -*- coding: utf-8 -*-
import scrapy
import logging 

class CountriesSpider(scrapy.Spider):
    name = 'countries'
    allowed_domains = ['www.worldometers.info']
    start_urls = ['https://www.worldometers.info/world-population/population-by-country/']

    custom_settings = {
        'LOG_LEVEL': logging.WARNING}
  
    def parse(self, response):
        countries=response.xpath("//td/a")
        country_sizes=response.xpath("(//table[@id='example2'])[1]/tbody/tr")
        for country,country_size in zip(countries,country_sizes):
            name=country.xpath(".//text()").get()
            link_only = country.xpath(".//@href").get()
            link=country.xpath(".//@href").get()
            size=country_size.xpath(".//td[7]/text()").get()
            yield response.follow(url=link, callback=self.parse_country,  meta={'country_name': name, 'link_only':link_only,'country_size':size})
               
    def parse_country(self, response):
        name=response.request.meta['country_name']
        size=response.request.meta['country_size']
        regions = response.xpath("//li[4]//a/text()").get()
        subregions = response.xpath("//li[5]//a/text()").get()
        rows = response.xpath("(//table[@class='table table-striped table-bordered table-hover table-condensed table-list'])[1]/tbody/tr")
        for row in rows:
            region=regions
            subregion=subregions
            year=row.xpath(".//td[1]/text()").get()
            population=row.xpath(".//td[2]/strong/text()").get()
            yearly_change=row.xpath(".//td[3]/text()").get()
            migrants=row.xpath(".//td[5]/text()").get()
            age=row.xpath(".//td[6]/text()").get()
            fertility=row.xpath(".//td[7]/text()").get()
            density=row.xpath(".//td[8]/text()").get()
            urban_pop=row.xpath(".//td[10]/text()").get()
            world_pop=row.xpath(".//td[12]/text()").get()
            yield{
                'Country': name,
                'Region': region,
                'Subregion':subregion,
                'Country_size':size,
                'Year': year,
                'Population': population,
                'YoY Pop %':yearly_change,
                'Migrants(net)': migrants,
                'Median Age': age,
                'Fertility Rate': fertility,
                'Density(P/Km²)': density,
                'Urban Population': urban_pop,
                'World Population': world_pop
            }            
```

#### This code runs the crawler.


```python
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings

# Remove logging
logging.getLogger('scrapy').setLevel(logging.WARNING)

process = CrawlerProcess(get_project_settings())

process.crawl('countries')
process.start()
```

## Data Wrangling

#### Load the basic wrangling packages.


```python
import numpy as np
import pandas as pd
```

#### Import the .csv file created above and check out some summary statistics.


```python
df = pd.read_csv("worldometers.csv",)
print(df.shape)
for col in df.columns:
    print(col," - ",df[col].dtype," - ",df[col].nunique(),"unique levels")
```

    (4195, 13)
    Country  -  object  -  235 unique levels
    Region  -  object  -  7 unique levels
    Subregion  -  object  -  22 unique levels
    Country_size  -  object  -  226 unique levels
    Year  -  int64  -  18 unique levels
    Population  -  object  -  4192 unique levels
    YoY Pop %  -  object  -  740 unique levels
    Migrants(net)  -  object  -  2535 unique levels
    Median Age  -  object  -  646 unique levels
    Fertility Rate  -  object  -  1079 unique levels
    Density(P/Km²)  -  object  -  620 unique levels
    Urban Population  -  object  -  3595 unique levels
    World Population  -  object  -  18 unique levels
    

#### I will need to change the format of all the columns that have dtype='Object' to Interger/Float . I will get to that after I check for nulls.

#### FIrst, I need to check for null values and make any corrections.


```python
df.isnull().sum()
```




    Country               0
    Region                0
    Subregion            51
    Country_size          0
    Year                  0
    Population            0
    YoY Pop %             0
    Migrants(net)         0
    Median Age            0
    Fertility Rate        0
    Density(P/Km²)        0
    Urban Population      0
    World Population    595
    dtype: int64



#### From looking at the N/A values, there were 35 small countries that didn't have complete information. I decided to remove them for now.


```python
df.dropna(subset=['World Population'], how='all', inplace=True)
```

#### I noticed that the Urban Population column had some N.A. values so I want turn those into zeroes so that I can change the format from string to numerical.


```python
for col in df.columns:
    print(col," - ",df[df[col]=='N.A.'][col].count())
```

    Country  -  0
    Region  -  0
    Subregion  -  0
    Country_size  -  0
    Year  -  0
    Population  -  0
    YoY Pop %  -  0
    Migrants(net)  -  0
    Median Age  -  0
    Fertility Rate  -  0
    Density(P/Km²)  -  0
    Urban Population  -  48
    World Population  -  0
    


```python
df.replace('N.A.','0',inplace=True)
```

#### I need to make sure that all of the numerical columns are actually numbers. Pandas imported them as strings.


```python
df.columns
```




    Index(['Country', 'Region', 'Subregion', 'Country_size', 'Year', 'Population',
           'YoY Pop %', 'Migrants(net)', 'Median Age', 'Fertility Rate',
           'Density(P/Km²)', 'Urban Population', 'World Population'],
          dtype='object')



#### I will put the columns into categories based on how I want to reformat them into numerical values.


```python
nums_w_commas = ['Country_size','Population','Migrants(net)', 'Urban Population', 'World Population','Density(P/Km²)']
nums_w_percents = ['YoY Pop %']
nums_less_1000 = ['Median Age', 'Fertility Rate','Density(P/Km²)']
year = ['Year']
```

#### For the number columns that have commas, I need to split the string at the comma and rejoin the numbers and set to numeric.


```python
for col in nums_w_commas:
      df[col]=pd.to_numeric(df[col].str.split(',').str.join(''))
```

#### For numbers with % at the end. I need to strip the % and set as numeric


```python
for col in nums_w_percents:
    df[col] = pd.to_numeric(df[col].apply(lambda x:x.split('%')[0]))
```

#### For small numbers that don't have commas.


```python
for col in nums_less_1000:
    df[col] = pd.to_numeric(df[col])
```


```python
df['Year'] = df['Year'].astype(str)
```

#### Check to makes sure all the columns are in the right data type. It looks like everything worked correctly.


```python
df.dtypes
```




    Country              object
    Region               object
    Subregion            object
    Country_size          int64
    Year                 object
    Population            int64
    YoY Pop %           float64
    Migrants(net)         int64
    Median Age          float64
    Fertility Rate      float64
    Density(P/Km²)        int64
    Urban Population      int64
    World Population      int64
    dtype: object



## Data Visualization


```python
df['Urban %'] = df['Urban Population']/df['Population']
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Country</th>
      <th>Region</th>
      <th>Subregion</th>
      <th>Country_size</th>
      <th>Year</th>
      <th>Population</th>
      <th>YoY Pop %</th>
      <th>Migrants(net)</th>
      <th>Median Age</th>
      <th>Fertility Rate</th>
      <th>Density(P/Km²)</th>
      <th>Urban Population</th>
      <th>World Population</th>
      <th>Urban %</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>China</td>
      <td>Asia</td>
      <td>Eastern Asia</td>
      <td>9388211</td>
      <td>2020</td>
      <td>1439323776</td>
      <td>0.39</td>
      <td>-348399</td>
      <td>38.4</td>
      <td>1.69</td>
      <td>153</td>
      <td>875075919</td>
      <td>7794798739</td>
      <td>0.607977</td>
    </tr>
    <tr>
      <th>1</th>
      <td>China</td>
      <td>Asia</td>
      <td>Eastern Asia</td>
      <td>9388211</td>
      <td>2019</td>
      <td>1433783686</td>
      <td>0.43</td>
      <td>-348399</td>
      <td>37.0</td>
      <td>1.65</td>
      <td>153</td>
      <td>856409297</td>
      <td>7713468100</td>
      <td>0.597307</td>
    </tr>
    <tr>
      <th>2</th>
      <td>China</td>
      <td>Asia</td>
      <td>Eastern Asia</td>
      <td>9388211</td>
      <td>2018</td>
      <td>1427647786</td>
      <td>0.47</td>
      <td>-348399</td>
      <td>37.0</td>
      <td>1.65</td>
      <td>152</td>
      <td>837022095</td>
      <td>7631091040</td>
      <td>0.586295</td>
    </tr>
    <tr>
      <th>3</th>
      <td>China</td>
      <td>Asia</td>
      <td>Eastern Asia</td>
      <td>9388211</td>
      <td>2017</td>
      <td>1421021791</td>
      <td>0.49</td>
      <td>-348399</td>
      <td>37.0</td>
      <td>1.65</td>
      <td>151</td>
      <td>816957613</td>
      <td>7547858925</td>
      <td>0.574909</td>
    </tr>
    <tr>
      <th>4</th>
      <td>China</td>
      <td>Asia</td>
      <td>Eastern Asia</td>
      <td>9388211</td>
      <td>2016</td>
      <td>1414049351</td>
      <td>0.51</td>
      <td>-348399</td>
      <td>37.0</td>
      <td>1.65</td>
      <td>151</td>
      <td>796289491</td>
      <td>7464022049</td>
      <td>0.563127</td>
    </tr>
  </tbody>
</table>
</div>



### Correlations

#### First, I will look at how my variables are correlated with each other. Then I can make some graphs that show the correlations more effectively.


```python
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

plt.figure(figsize=(10,7))
sns.heatmap(data=df.corr())
plt.tight_layout(pad=0)
```


![png](WoM%20Analysis_files/WoM%20Analysis_34_0.png)


### It looks like Fertility Rate and Median Age are highly negatively correlated. This makes sense, but let's see how that has changed over time and across different world regions.

#### Fertility Rate has been decreasing since 1955 across all world regions.


```python
plt.figure(figsize=(12,6))
sns.lineplot(data=df,x='Year',y='Fertility Rate',hue='Region')
plt.legend(loc='upper left', prop={'size':7}, bbox_to_anchor=(1,1))
plt.tight_layout(pad=7)
```


![png](WoM%20Analysis_files/WoM%20Analysis_37_0.png)


#### Conversely, the median age of region has been increasing since 1955.


```python
plt.figure(figsize=(12,6))
sns.lineplot(data=df,x='Year',y='Median Age',hue='Region')
plt.legend(loc='upper left', prop={'size':7}, bbox_to_anchor=(1,1))
plt.tight_layout(pad=7)
```


![png](WoM%20Analysis_files/WoM%20Analysis_39_0.png)


#### Below shows a scatterplot of Median Age against Fertility Rate. It's pretty hard to read since each year in the data set is displayed in this graph. So, I am going to show how to create an animated graph that shows how this relationship has changed over time.


```python
plt.figure(figsize=(12,7))
sns.scatterplot(data=df,x='Median Age',y='Fertility Rate',hue='Region')
plt.legend(loc='upper left', prop={'size':6}, bbox_to_anchor=(1,1))
plt.tight_layout(pad=7)
```


![png](WoM%20Analysis_files/WoM%20Analysis_41_0.png)



```python
import matplotlib.animation as animation
plt.rcParams['animation.ffmpeg_path'] = r'C:/Users/roger/ImageMagick/ffmpeg.exe'
plt.rcParams['animation.convert_path'] = r'C:/Users/roger/ImageMagick/convert.exe'
%matplotlib inline 

# Set up the plot to be animated
fig, ax = plt.subplots(figsize=(12,7))

# for convenience: define a function which prepares the data
def get_data(Year = '1955'):
    x = df.loc[df.Year == Year, 'Median Age']
    y = df.loc[df.Year == Year, 'Fertility Rate']
    col = df.loc[df.Year == Year, 'Region'] 
    return x,y,col

def init():
    x,y,col=get_data(Year = '1955')
    scat = sns.scatterplot(x,y,hue=col,legend=False) 
    
# animation function 
def animate(i):  
    global scats
    for scat in scats:
        scat.remove()
    scats=[]
    x,y,col=get_data(i)
    scats.append(sns.scatterplot(x,y,hue=col))
    # Put a legend to the right of the current axis
    plt.legend(loc='upper left', prop={'size':6}, bbox_to_anchor=(1,1))
    plt.tight_layout(pad=7)
    plt.xlim(df['Median Age'].min(), df['Median Age'].max())
    plt.xlabel('Median Age',fontsize=20)
    plt.ylim(df['Fertility Rate'].min(), df['Fertility Rate'].max())
    plt.ylabel('Fertility Rate',fontsize=20)
    plt.title(f'Fertility Rate vs. Median Age in {i}',fontsize=20)

anim = animation.FuncAnimation(fig, animate,
                               init_func=init, 
                               frames=reversed(df['Year'].unique()), 
                               interval=1000,)
anim.save('myanimation.gif', writer='imagemagick')
```


![png](WoM%20Analysis_files/WoM%20Analysis_42_0.png)



```python
Image('myanimation.gif')
```




    <IPython.core.display.Image object>



