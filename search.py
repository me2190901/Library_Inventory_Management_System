import pandas as pd

df=pd.read_excel(".\data.xlsx")

df.drop_duplicates(subset ="Title",
                     keep = False, inplace = True)
dic={}

fl=open("recognized.txt",'r')
data = fl.readlines()
for lines in data:
    dic={}
    line=lines.strip().split()
    for word in line:
        a=df[df['Title'].str.contains(word)][['Title']]


        for i in a["Title"]:
            if i not in dic:
                dic[i]=0
            dic[i]=dic[i]+len(word)
    # print(dic)
    try:
        maxtitle = max(zip(dic.values(), dic.keys()))[1]
        print(maxtitle)
    except:
        print("No Title Found")
