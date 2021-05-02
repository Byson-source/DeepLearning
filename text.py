list1=[1,2,3,4,5]
import pandas as pd
df=pd.DataFrame({'test':list1,
                1:[3,4,5,6,7]})
df.set_index("a",inplace=True)
print(df)