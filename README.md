''' Data Set Information:
# 
# This is a transnational data set which contains all the transactions occurring between 01/12/2010 and 09/12/2011 for a UK-based and registered non-store online retail.
# The company mainly sells unique all-occasion gifts. 
# Many customers of the company are wholesalers.(B2B online transactions)<br>
# 
# Attribute Information:
# 
# 1.InvoiceNo: Invoice number. Nominal, a 6-digit integral number uniquely assigned to each transaction. If this code starts with letter 'c', it indicates a cancellation.
# 2.StockCode: Product (item) code. Nominal, a 5-digit integral number uniquely assigned to each distinct product. <br>
# 3.Description: Product (item) name. Nominal. <br>
# 4.Quantity: The quantities of each product (item) per transaction. Numeric. <br>
# 5.InvoiceDate: Invice Date and time. Numeric, the day and time when each transaction was generated.<br> 
# 6.UnitPrice: Unit price. Numeric, Product price per unit in sterling. <br>
# 7.CustomerID: Customer number. Nominal, a 5-digit integral number uniquely assigned to each customer. <br>
# 8.Country: Country name. Nominal, the name of the country where each customer resides.
'''
import matplotlib.pyplot as plt
import pandas as pd
#import numpy as np
#import seaborn as sns#匯入seaborn, 並命名為sns, 爾後以sns 簡稱呼叫之
# # 1. Load Data

df = pd.read_excel('/Users/88693/Desktop/財經大數據 APP/data_association/Online Retail.xlsx', sheet_name='Online Retail')

print('showing the rows of dataset: %s \nand the columns of dataset: %s' % (df.shape[0], df.shape[1]))
 #check data rows and data columns
 
print('\nShow the data type of each column\n',df.dtypes) 
# show the data type of each column in the df

print('\nshowing the first 10 rows of data', df.head(10)) # show the top 10 rows of data

# # 2. Data Clean-Up
# #### - Negative Quantity

print('\nshowing the invalid records whose purchasing quantities are negative',df.loc[df['Quantity'] <= 0].shape)
#check the purchasing quantity, non-positive

df = df.loc[df['Quantity'] > 0]
# excluding the invalid purchasing records: non-positive quantity

print('\nshowing the valid transactions',df.shape)
#check data rows and data columns after screening

##### - Missing CustomerID

print('\nshowing the records with invalid customer IDs:', pd.isnull(df['CustomerID']).sum())
# check the non-existent customers by the 'isnull' function

df = df[pd.notnull(df['CustomerID'])] 
# excluding the records of non-existent customers
# applying the notnull function to extract the qualified transactions from df.

print('\nshowing the valid transactions',df.shape)  
#check data rows and data columns after screening null rows

##### - Excluding Incomplete Month

print('\nDate Range: %s ~ %s' % (df['InvoiceDate'].min(), df['InvoiceDate'].max()))
# checking the beginning date and the end one of transactions

print('\nshowing the records of incomplete month: ',df.loc[df['InvoiceDate'] >= '2011-12-01'].shape)
# counting the rows whose invoicedates are after the date of 2011-12-01

df = df.loc[df['InvoiceDate'] < '2011-12-01'] 
# excluding records of the incomplete month, 2011/12, keeping the rows before 2011-12-01

print('\nshowing the transactions: %s  \nranging from %s ~ %s' % (df.shape[0],
    df['InvoiceDate'].min(), df['InvoiceDate'].max())) #check data rows and data columns after screening

##### - Total Sales

df['Sales'] = df['Quantity'] * df['UnitPrice'] # a product of two columns. 
# create a new column, Sales, to calculate the volumne of sales for each transaction 

print('\nshwoing the Sales of top 10 records:\n', df['Sales'].head(10))

##### - Per Customer Data

customer_df = df.groupby('CustomerID').agg({
    'Sales': sum, 'InvoiceNo': lambda x: x.nunique()})
# grouping by customerID to summarize the associated total order "volumes" 
# and orders(nunique(): 'n'umber of unique)
# creat a new dataframe, customer_df, with the key customer_id, 
# and attached two columns, one is the summation of sales of the sae customer_id
# and the other is the number of unique invoiceNo, that is, the total number of orders of 
# the same customer 

# rename the aggregate_Sales as TotalSales; rename the aggregate_InvoiceNo as OrderCount
customer_df.columns = ['TotalSales', 'OrderCount']

# adding a new column, AvgOrderValue, a division between two columns
customer_df['AvgOrderValue'] = customer_df['TotalSales']/customer_df['OrderCount']

print('\nshowing the top 10 records of customer_df:\n',customer_df.head(10)) # show the top 15 row of customer_df

print('\nshowning the descriptive statistics of customer purchasing:\n',customer_df.describe())
# Show the statistics of continuous variables in customer_df

rank_df = customer_df.rank(method='first') 
# .rank() is a function to rank each column of numberic value.
# The rank is a sequence of a list from the samllest to the maximum with the number
# from 1 to the total size of list.
# In this case, the smallest will be assigned with 1, the largest with 4298 (total record size).
# The 'first' method means that the ranks are assigned in the orders that the values appear 
# in the array when eccountering the same value.
# That original column values had been transformed into ranks.
# Therefore, the rank of each column of purchasing information now ranges from 1 to 4298.

print('\nshowing the transformed ranking data:\n',rank_df.head(10))
# the purpose of rank transformation is to levelize the column values
# the sales, averageorder, and count are very disperse and based on different units.
# Another way is to standardize each column directly. But this way may not remove the bias cauesed by extreme values. 

print('\nshowing the descriptive stastics of transformed ranking data:\n',rank_df.describe())
# All the columns' descriptive statistics are the same.

normalized_df = (rank_df - rank_df.mean()) / rank_df.std() 
# This is a standardization process.
# .mean() is to calculate the mean of numberic variables.
# .std() is to calculate the standard deviation of numberic variables.
print('\nshowing the normalized ranking data: \n',normalized_df.head(10))

print('\nshowing the descriptive stastics ofnormalized ranking data: \n',normalized_df.describe()) # all the same!

## 3. Customer Segmentation via K-Means Clustering
# import the Kmeans module from sklearn.cluster
from sklearn.cluster import KMeans

##### - K-Means Clustering: 
# it is a method of clustering observations into the intended number of groups.

kmeans = KMeans(n_clusters=4).fit(
    normalized_df[['TotalSales', 'OrderCount', 'AvgOrderValue']])
#adopting the three columns into the KMeans model to classify all the 4298 customers into 4 segments
# n_clusters is the intended number of groups.

print('\nshowing the assigned cluster for each customer:\n',kmeans.labels_)
# kmeans.labels_: Labels of each point, that is, assigning the belonged cluster number to each point
# 4298 customers will be assigned 0,1,2,3 groups respectively.

print('\nshowing all the cluster centers:\n', kmeans.cluster_centers_)
#showing the average values (TotalSales', 'OrderCount', 'AvgOrderValue') in the 4 segments

print('\nshowing the Sum of Squared Distance of each customer:\n', kmeans.inertia_)
# Sum of squared distances of each observations to their belonged cluster center. 
# the fit process is to find the best centroid seed by the smallest interia

four_cluster_df = normalized_df[['TotalSales', 'OrderCount', 'AvgOrderValue']].copy(deep=True)
four_cluster_df['Cluster'] = kmeans.labels_
#copy the original normalized-df to four_cluster_df and append the fourth column: cluster(segments)

print('\nshowing the top 10 customer with the assigned cluster:\n',four_cluster_df.head(10))

print('\nshowing the number of customers each cluster:\n',
      four_cluster_df.groupby('Cluster')['TotalSales'].count())
#groupby cluster and count the number of records occurring in the Totalsales column, indicating each segment size.

# predict the most likely cluster for a given input: 
# kmeans.predict(normalized_df[['TotalSales', 'OrderCount', 'AvgOrderValue']])

colors=['blue', 'red', 'orange', 'green', 'black']
for i in range(0,4):
    plt.scatter(
    four_cluster_df.loc[four_cluster_df['Cluster'] == i]['OrderCount'], 
    four_cluster_df.loc[four_cluster_df['Cluster'] == i]['TotalSales'],
    color=colors[i])
# scatter cluster i with the x-axis of ordercount and the y-axis of totalsales, indicating by the blue color.
plt.title('TotalSales vs. OrderCount')
plt.xlabel('Order Count')
plt.ylabel('Total Sales')
plt.grid()
plt.show()

for i in range(0,4):
    plt.scatter(
    four_cluster_df.loc[four_cluster_df['Cluster'] == i]['OrderCount'], 
    four_cluster_df.loc[four_cluster_df['Cluster'] == i]['AvgOrderValue'],
    c=colors[i])
plt.title('AvgOrderValue vs. OrderCount')
plt.xlabel('Order Count')
plt.ylabel('Avg Order Value')
plt.grid()
plt.show()

for i in range(0,4):
    plt.scatter(
    four_cluster_df.loc[four_cluster_df['Cluster'] == i]['TotalSales'], 
    four_cluster_df.loc[four_cluster_df['Cluster'] == i]['AvgOrderValue'],
    color=colors[i])
plt.title('AvgOrderValue vs. TotalSales')
plt.xlabel('Total Sales')
plt.ylabel('Avg Order Value')
plt.grid()
plt.show()

from sklearn.metrics import silhouette_score 
# evaluate the best number of clusters (groups)
# silhouette_score is the ratio of (b:the mean distance to the outside nearest center 
# minus a:mean intra distance to the cluster center)/ max(a,b)
# "+" means within this group is better than to assign to the other group
# "-" means the other case
# the value is near to 1, the better grouping.
# the worst case is the value near to -1, indicating a wrong grouping.

SSD_elbow=[]
Silhouette_score=[]
n_cluster=range(2,9) # assign values from 2,3,4,...8 to n_cluster
for i in n_cluster: 
# perform kmeans clustering analysis by categorizing into 2,3,4,... 8 clusters
    kmeans = KMeans(n_clusters=i).fit(
        normalized_df[['TotalSales', 'OrderCount', 'AvgOrderValue']])
    print('SSD within %i Clusters: %0.4f' % (i, kmeans.inertia_)) # 誤差平方和 (SSE)
    SSD_elbow.append(kmeans.inertia_)
    silhouette_avg = silhouette_score(
        normalized_df[['TotalSales', 'OrderCount', 'AvgOrderValue']], 
        kmeans.labels_) 
# this is a function of calculating silhousette_score by inputing 
# associated attributes and # of cluster  
    print('Silhouette Score for %i Clusters: %0.4f' % (i, silhouette_avg))
    Silhouette_score.append(silhouette_avg)

# Theoretically, find the max silhouette score to define the best number of clustering
selected_n = Silhouette_score.index(max(Silhouette_score)) + 2 
# at lease two clusters; the first clustering analysis begins from categorizing into two clusters

# Plot elbow ans Silhouette 
plt.subplot(121)  #plot the left fig in a one-row & two-column figure
plt.title('SSD (elbow method)')
plt.plot(n_cluster, SSD_elbow)
plt.plot(selected_n, SSD_elbow[selected_n]-2, marker='o', color='green') 

plt.subplot(122)  #plot the right fig in a one-row & two-column figure
plt.title('Silhouette score')
plt.plot(n_cluster, Silhouette_score)
plt.plot(selected_n, Silhouette_score[selected_n-2], marker='o', color='green') 
# The index of Silhouette score starts from 0,
# while the least selected n is 2. So, the selected n is 
# larger than the corresponding index by 2. 
plt.tight_layout()
plt.show()

# for practical clustering, select the cluster number 
# which the SSE begings to stop decreasing dramatically.

selected_n=4

kmeans = KMeans(n_clusters=selected_n).fit(
     normalized_df[['TotalSales', 'OrderCount', 'AvgOrderValue']])
centers=kmeans.cluster_centers_ # keeping the centers of four clusters

plt.subplot(121)  #plot the left fig in a one-row & two-column figure
plt.title('SSD (elbow method)')
plt.plot(n_cluster, SSD_elbow)
plt.plot(selected_n, SSD_elbow[selected_n - 2], marker='o', color='red')
#plot the cluster number with a red point

plt.subplot(122)  #plot the right fig in a one-row & two-column figure
plt.title('Silhouette score')
plt.plot(n_cluster, Silhouette_score)
plt.plot(selected_n, Silhouette_score[selected_n - 2], marker='o', color='red') 
plt.tight_layout()
plt.show()

#### segmentation by ranking values
four_cluster_df = normalized_df[['TotalSales', 'OrderCount', 'AvgOrderValue']].copy(deep=True)
four_cluster_df['Cluster'] = kmeans.labels_
#copy the original normalized-df to four_cluster_df and append the fourth column: cluster(segments)

for i in range(0, selected_n):
    plt.scatter(
    four_cluster_df.loc[four_cluster_df['Cluster'] == i]['OrderCount'], 
    four_cluster_df.loc[four_cluster_df['Cluster'] == i]['TotalSales'],
    color=colors[i])
    ax = plt.gca()# 鎖定前繪圖之figure axes
    plt.text(0.04+i*0.2,-0.2,s='cluster '+str(i+1),fontsize=10, 
             color=colors[i],weight='bold',transform=ax.transAxes)
plt.scatter(centers.T[1], centers.T[0], marker='^', color=colors[4])
# scatter cluster i with the x-axis of ordercount and the y-axis of totalsales, 
# indicating by the blue color.
# centers.T: row are variables/column are clusters 
# --> center.T[1] means the second variable of four clusters 
# --> center.T[0] means the first variable of four clusters 
plt.title('TotalSales vs. OrderCount')
plt.xlabel('Order Count')
plt.ylabel('Total Sales')
plt.grid()
plt.show()

for i in range(0, selected_n):
    plt.scatter(
    four_cluster_df.loc[four_cluster_df['Cluster'] == i]['OrderCount'], 
    four_cluster_df.loc[four_cluster_df['Cluster'] == i]['AvgOrderValue'],
    color=colors[i])
    ax = plt.gca()# 鎖定前繪圖之figure axes
    plt.text(0.04+i*0.2,-0.2,s='cluster '+str(i+1),fontsize=10, 
             color=colors[i],weight='bold',transform=ax.transAxes)
plt.scatter(centers.T[1], centers.T[2], marker='^', color=colors[4])
plt.title('AvgOrderValue vs. OrderCount')
plt.xlabel('Order Count')
plt.ylabel('Avg Order Value')
plt.grid()
plt.show()

for i in range(0, selected_n):
    plt.scatter(
    four_cluster_df.loc[four_cluster_df['Cluster'] == i]['TotalSales'], 
    four_cluster_df.loc[four_cluster_df['Cluster'] == i]['AvgOrderValue'],
    color=colors[i])
    ax = plt.gca()# 鎖定前繪圖之figure axes
    plt.text(0.04+i*0.2,-0.2,s='cluster '+str(i+1),fontsize=10, 
             color=colors[i],weight='bold',transform=ax.transAxes)
plt.scatter(centers.T[0], centers.T[2], marker='^', color=colors[4])
plt.title('AvgOrderValue vs. TotalSales')
plt.xlabel('Total Sales')
plt.ylabel('Avg Order Value')
plt.grid()
plt.show()

#### segmentation by the mean values of original purchasing data
customer_df['Cluster'] = kmeans.labels_
for i in range(0, selected_n):
    center_x=customer_df.loc[customer_df['Cluster'] == i]['OrderCount'].mean()
    center_y=customer_df.loc[customer_df['Cluster'] == i]['TotalSales'].mean()
    plt.scatter(center_x, center_y, marker='^', color=colors[i],s=60)
    ax = plt.gca()# 鎖定前繪圖之figure axes
    plt.text(0.04+i*0.2,-0.2,s='cluster '+str(i+1),fontsize=10, 
             color=colors[i],weight='bold',transform=ax.transAxes)
    # scatter cluster i with the x-axis of ordercount and the y-axis of totalsales, indicating by the blue color.
plt.title('TotalSales vs. OrderCount')
plt.xlabel('Order Count')
plt.ylabel('Total Sales')
plt.grid()
plt.show()

for i in range(0, selected_n):
    center_x=customer_df.loc[customer_df['Cluster'] == i]['OrderCount'].mean()
    center_y=customer_df.loc[customer_df['Cluster'] == i]['AvgOrderValue'].mean()
    plt.scatter(center_x, center_y, marker='^', color=colors[i],s=60)
    ax = plt.gca()# 鎖定前繪圖之figure axes
    plt.text(0.04+i*0.2,-0.2,s='cluster '+str(i+1),fontsize=10, 
             color=colors[i],weight='bold',transform=ax.transAxes)
plt.title('AvgOrderValue vs. OrderCount')
plt.xlabel('Order Count')
plt.ylabel('Avg Order Value')
plt.grid()
plt.show()

for i in range(0, selected_n):
    center_x=customer_df.loc[customer_df['Cluster'] == i]['TotalSales'].mean()
    center_y=customer_df.loc[customer_df['Cluster'] == i]['AvgOrderValue'].mean()
    plt.scatter(center_x, center_y, marker='^', color=colors[i],s=60)
    ax = plt.gca()# 鎖定前繪圖之figure axes
    plt.text(0.04+i*0.2,-0.2,s='cluster '+str(i+1),fontsize=10, 
             color=colors[i],weight='bold',transform=ax.transAxes)
plt.title('AvgOrderValue vs. TotalSales')
plt.xlabel('Total Sales')
plt.ylabel('Avg Order Value')
plt.grid()
plt.show()

##### - Interpreting Customer Segments
kmeans = KMeans(n_clusters=4).fit(
    normalized_df[['TotalSales', 'OrderCount', 'AvgOrderValue']])

four_cluster_df = normalized_df[['TotalSales', 'OrderCount', 'AvgOrderValue']].copy(deep=True)
four_cluster_df['Cluster'] = kmeans.labels_
print('\nshowing the cluster centers:\n',kmeans.cluster_centers_)
print('\nshowing the top 10 cutomers:\n',four_cluster_df.head(10))
print('\nshowing the number of each cluster:\n', 
      four_cluster_df.groupby('Cluster')['TotalSales'].count())

first_cluster=four_cluster_df.loc[four_cluster_df['Cluster'] == 0]
second_cluster=four_cluster_df.loc[four_cluster_df['Cluster'] == 1]
third_cluster=four_cluster_df.loc[four_cluster_df['Cluster'] == 2]
fourth_cluster=four_cluster_df.loc[four_cluster_df['Cluster'] == 3]
## show the descriptive statistics of each segment
print('\nthe first cluster statistics:\n',first_cluster.describe())
print('\nthe second cluster statistics:\n',second_cluster.describe())
print('\nthe third cluster statistics:\n',third_cluster.describe()) 
print('\nthe fourth cluster statistics:\n',fourth_cluster.describe()) 

#### boxplot
four_cluster_df.boxplot(by='Cluster',column=['TotalSales', 'OrderCount', 'AvgOrderValue'],
                        grid=True)
#plt.ylabel(four_cluster_df.columns[i], size=14)
plt.xlabel('Cluster', size=14)
plt.title('Boxplot of purchasing information by clusters', size=18)
plt.show() #把目前繪製的圖表,展示出來

high_value_cluster = second_cluster
#screening the customers in the segment 2 into hogh_value_cluster, 
#that is , to check who the most important customers are
high_value_cluster.head() #index by customerID

print('\nThe statistics of high-vlaue segment:\n',second_cluster.describe())
#showing the descriptive statistics of high_value_segment from the data in the original customer_df

print('\nThe top items purchased by the high-vlaue segment:\n',pd.DataFrame(
    df.loc[df['CustomerID'].isin(high_value_cluster.index)
    ].groupby('Description')['StockCode'].count().sort_values(ascending=False).head()))
#screening the high_value_segment customers from df, grouping by what they bought, 
#counting the number of stockCode occurrance, and sorting the occurances
# the top categories of products brought by the most important customers

low_value_cluster = first_cluster
print('\nThe statistics of low-vlaue segment:\n',first_cluster.describe()) 

print('\nThe top items purchased by the low-vlaue segment:\n',pd.DataFrame(
    df.loc[df['CustomerID'].isin(low_value_cluster.index)
    ].groupby('Description')['StockCode'].count().sort_values(ascending=False).head()))
# the top categories of products brought by the least important customers
# The types and amounts of goods buying of high_value_segment are really different from that of low_value_segment

##### another clustering approach by centering with real data
from sklearn_extra.cluster import KMedoids
# this module must be installed first. "pip install scikit-learn-extra"
# The k-medoids problem is a clustering problem similar to k-means.
# The algorithm of k-medoids randomly chooses actual points as centers,
# while k-means does not necessary choose one of the input data points as the center of a cluster. 
# The latter is usually adopting the average between the points in the cluster.
KMed = KMedoids(n_clusters=4).fit(
    normalized_df[['TotalSales', 'OrderCount', 'AvgOrderValue']])
centers=KMed.cluster_centers_
four_cluster_df = normalized_df[['TotalSales', 'OrderCount', 'AvgOrderValue']].copy(deep=True)
four_cluster_df['Cluster'] = KMed.labels_
print('\nthe cluster centers:\n',KMed.cluster_centers_)

# predict the most likely cluster for a given input: 
# KMed.predict(normalized_df[['TotalSales', 'OrderCount', 'AvgOrderValue']])

#### segmentation by ranking values

for i in range(0, selected_n):
    plt.scatter(
    four_cluster_df.loc[four_cluster_df['Cluster'] == i]['OrderCount'], 
    four_cluster_df.loc[four_cluster_df['Cluster'] == i]['TotalSales'],
    color=colors[i])
    ax = plt.gca()# 鎖定前繪圖之figure axes
    plt.text(0.04+i*0.2,-0.2,s='cluster '+str(i+1),fontsize=10, 
             color=colors[i],weight='bold',transform=ax.transAxes)
plt.scatter(centers.T[1], centers.T[0], marker='^', color=colors[4])
    # scatter cluster i with the x-axis of ordercount and the y-axis of totalsales, indicating by the blue color.
plt.title('TotalSales vs. OrderCount')
plt.xlabel('Order Count')
plt.ylabel('Total Sales')
plt.grid()
plt.show()

for i in range(0, selected_n):
    plt.scatter(
    four_cluster_df.loc[four_cluster_df['Cluster'] == i]['OrderCount'], 
    four_cluster_df.loc[four_cluster_df['Cluster'] == i]['AvgOrderValue'],
    color=colors[i])
    ax = plt.gca()# 鎖定前繪圖之figure axes
    plt.text(0.04+i*0.2,-0.2,s='cluster '+str(i+1),fontsize=10, 
             color=colors[i],weight='bold',transform=ax.transAxes)
plt.scatter(centers.T[1], centers.T[2], marker='^', color=colors[4])
plt.title('AvgOrderValue vs. OrderCount')
plt.xlabel('Order Count')
plt.ylabel('Avg Order Value')
plt.grid()
plt.show()

for i in range(0, selected_n):
    plt.scatter(
    four_cluster_df.loc[four_cluster_df['Cluster'] == i]['TotalSales'], 
    four_cluster_df.loc[four_cluster_df['Cluster'] == i]['AvgOrderValue'],
    color=colors[i])
    ax = plt.gca()# 鎖定前繪圖之figure axes
    plt.text(0.04+i*0.2,-0.2,s='cluster '+str(i+1),fontsize=10, 
             color=colors[i],weight='bold',transform=ax.transAxes)
plt.scatter(centers.T[0], centers.T[2], marker='^', color=colors[4])
plt.title('AvgOrderValue vs. TotalSales')
plt.xlabel('Total Sales')
plt.ylabel('Avg Order Value')
plt.grid()
plt.show()
