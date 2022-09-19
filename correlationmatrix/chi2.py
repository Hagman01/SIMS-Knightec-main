import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn import preprocessing
from scipy import stats

#---------------------------
# The results of this needs
# to be overlooked, not sure
# if it is correct
#---------------------------

#Chi-square test of independence
def chi2(df, col1, col2):
    #Create the contingency table
    df_cont = pd.crosstab(index = df[col1], columns = df[col2])
    
    #Calculate degree of freedom
    degree_f = (df_cont.shape[0]-1) * (df_cont.shape[1]-1)

    #Sum up the totals for row and columns
    df_cont.loc[:,'Total'] = df_cont.sum(axis=1)
    df_cont.loc['Total'] = df_cont.sum()
    
    #create the expected value dataframe
    df_exp = df_cont.copy()
    df_exp.iloc[:,:] = np.multiply.outer(df_cont.sum(1).values, df_cont.sum().values) / df_cont.sum().sum()

    #calculate chi-square values
    df_chi2 = ((df_cont - df_exp)**2) / df_exp
    df_chi2.loc[:,'Total'] = df_chi2.sum(axis=1)
    df_chi2.loc['Total'] = df_chi2.sum()

    #get chi-square score
    chi_square_score = df_chi2.iloc[:-1,:-1].sum().sum()

    #calculate the p-value
    p = stats.distributions.chi2.sf(chi_square_score, degree_f)

    return chi_square_score, degree_f, p

#Read dataset
df = pd.read_csv("data.csv")

#Change target feature to numerical
df["Closed Reason"] = df["Closed Reason"].replace(["No action required"], 1)
df["Closed Reason"] = df["Closed Reason"].replace(["Customer resolved on own"], 1)
df["Closed Reason"] = df["Closed Reason"].replace(["Duplicate event"], 1)
df["Closed Reason"] = df["Closed Reason"].replace(["Service Case Created"], 0)
df["Closed Reason"] = df["Closed Reason"].fillna(0)

#Encode dataset
for x in df:
    le = preprocessing.LabelEncoder()
    le.fit(df[x])
    df[x] = le.transform(df[x])

#H0, the 2 categorical variables to be compared are independent of each other
#H1, the 2 categorical variables to be compared are dependent of each other
risk = 0.05 #risk of concluding that the two variables are independent when in reality they are not

#Conduct chi-square test on every feature
corr_matrix = []
corr_list = []
for i in df:
    for j in df:
        chi_score, degree_f, p = chi2(df, i, j)
        if p < risk: #Reject H0 and accept H1
            corr_list.append(1) #Are dependent
        elif p > risk: #Accept H0 and reject H1
            corr_list.append(0) #Are independent
    if corr_list:
        corr_matrix.append(corr_list)
    corr_list = []

#Create heatmap of correlation matrix, 1 means it is correlated and 0 means it is not
labels = df.columns.values
sns.heatmap(corr_matrix, linewidths=0.5, xticklabels=labels, yticklabels=labels)
plt.title('Results of chi square test showing correlation')
plt.show()