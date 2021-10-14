import pandas as pd

df=pd.read_csv(".\data1.csv")

df.drop_duplicates(subset ="Title",
                     keep = False, inplace = True)

# df.set_index('Title')



word="Materials for architectural design"
a=df[df['Title'].str.contains(word,case=False)][['Title']]
df.index.name="SNo"
# print(df.index.name)
for i in a["Title"]:
    print(i)
    b=df.index[df["Title"]==i]
    b=b.tolist()
    ind=b[0]
    df.iat[ind,1]=df.iat[ind,1]+1
# print(ind)
print(df)


df.to_csv("data1.csv", index=False)
