from sklearn.cross_validation import train_test_split
import pandas as pd

dta = pd.read_csv('/Users/pengfeiwang/Desktop/dogkfc/Project3_poodleKFC_train/sift_features.csv')
dta = pd.DataFrame.transpose(dta)
def Obj_Type(x):
    if str(x).find('chicken') != -1: return('1')
    else: return(0)
    
dta['dtype'] = list(dta.index)
dta['dtype'] = dta['dtype'].map(lambda x: Obj_Type(x))

y = dta['dtype']
x = dta.drop(['dtype'],axis=1)
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size = 0.1, random_state = 2006)

train = pd.merge(train_x,train_y.to_frame(),left_index=True,right_index=True)
test = pd.merge(test_x,test_y.to_frame(),left_index=True,right_index=True)
train.to_csv('/Users/pw2406/Desktop/Project3_poodleKFC_train/train_set.csv')
test.to_csv('/Users/pw2406/Desktop/Project3_poodleKFC_train/test_set.csv')
