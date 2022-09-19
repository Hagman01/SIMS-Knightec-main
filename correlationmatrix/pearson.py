#!/usr/bin/env python3

from matplotlib import pyplot
from pandas import read_csv
import numpy
import csv
from pandas import *
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_excel("optiRun-data.ods")

dataType = ['Instrument Type', 'Installed Product ID', 'System Name', 'Event Description', 'Component', 'Error ID', 'Source of Event', 
'Location', 'Severity', 'Alarm/Alert', 'ServiceMax Case/WO URL', 'Status', 'Exp Date', 'Region', 'Last Service', 
'Last Maintenance', 'Alert to Acknowledge Time', 'Triage Time', 'Closed Reason', 'Comments']

print(data[:10])
for x in dataType:
	data[x]=data[x].astype('category').cat.codes

#data = pd.get_dummies(data)
corr = data.corr()
data.info()
sns.heatmap(corr, annot=True);
pyplot.show()