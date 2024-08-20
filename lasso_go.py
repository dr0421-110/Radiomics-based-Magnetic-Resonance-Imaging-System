import sklearn
import  pandas as pd
data=pd.read_csv(open('/mnt/data6/mvi_go/try.csv','r'))

model=sklearn.linear_model.LassoLarsIC('bic')
model.fit(data)