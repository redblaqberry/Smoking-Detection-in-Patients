import numpy as np
import pandas as pd
from sklearn.utils import column_or_1d
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.utils import column_or_1d
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras import layers, models
from sklearn.metrics import accuracy_score



df = pd.read_csv('data/train.csv')
df_test = pd.read_csv('data/test.csv')
samples = pd.read_csv('data/sample_submission.csv')

print(df.head())
nan_count = df.isnull().sum().sum()
print('Number of NaN values:', nan_count)

columns = df.columns.tolist()
print(columns)
#['id', 'age', 'height(cm)', 'weight(kg)', 'waist(cm)', 'eyesight(left)',
# 'eyesight(right)', 'hearing(left)', 'hearing(right)', 'systolic', 'relaxation',
# 'fasting blood sugar', 'Cholesterol', 'triglyceride', 'HDL', 'LDL', 'hemoglobin',
# 'Urine protein', 'serum creatinine', 'AST', 'ALT', 'Gtp', 'dental caries', 'smoking']

# smoking_categories = df['smoking'].unique()
# for column in columns:
#     data_to_plot = [df[df['smoking'] == category][column].dropna() for category in smoking_categories]
#     plt.figure(figsize=(10, 6))
#     plt.boxplot(data_to_plot, labels=smoking_categories)
#     plt.title(f'{column} vs. Smoking')
#     plt.ylabel(column)
#     plt.xlabel('Smoking')
#     plt.show()

features_num = ['age', 'height(cm)', 'weight(kg)', 'waist(cm)', 'eyesight(left)',
                'eyesight(right)', 'systolic', 'relaxation', 'fasting blood sugar',
                'Cholesterol', 'triglyceride', 'HDL', 'LDL', 'hemoglobin',
                'Urine protein', 'serum creatinine', 'AST', 'ALT', 'Gtp', 'bmi']

features_cat = ['hearing(left)', 'hearing(right)', 'dental caries']


df['bmi'] = df['weight(kg)'] / pow(df['height(cm)']/100,2)
df_test['bmi'] = df_test['weight(kg)'] / pow(df_test['height(cm)']/100,2)
corr_pearson = df[features_num].corr(method='pearson')
plt.figure(figsize=(12,10))
sns.heatmap(corr_pearson, annot=True, cmap='RdYlGn',
            vmin=-1, vmax=+1, fmt='.2f',
            linecolor='black', linewidths=0.5)
plt.title('Pearson Correlation')
plt.show()





df = df.drop(['height(cm)','weight(kg)'],axis=1)
df_test = df_test.drop(['height(cm)','weight(kg)'],axis=1)

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['smoking'])

train_data = df.drop(['smoking','id'],axis=1)
test_data = df_test.drop(['id'],axis=1)

x_train, x_val, y_train, y_val = train_test_split(train_data, y, test_size=0.2,random_state=42, stratify=y)

clf = xgb.XGBClassifier()
clf.fit(x_train, y_train)
pred = clf.predict(x_val)

preds_test = clf.predict(test_data)
samples['smoking'] = preds_test
samples['smoking'] = samples["smoking"].astype(int)
samples.to_csv('data/submission.csv',index=False)

scores = accuracy_score(y_val, pred)
print("Accuracy: ",scores)
