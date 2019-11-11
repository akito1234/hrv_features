
import pandas as pd

df = pd.read_excel(r"C:\Users\akito\Desktop\stress\03.Analysis\Analysis_Features\biosignal_datasets.xlsx",index_col=0)
print(df.index)

df2 = pd.read_excel(r"C:\Users\akito\Desktop\stress\03.Analysis\Analysis_Features\biosignal_datasets.xlsx",index_col=0)
df_summary = pd.concat([df,df2])

print(df_summary)