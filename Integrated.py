import spine_segment
import glob as gb
import text_recognize as tr
import pandas as pd
df=pd.read_csv(".\data1.csv")

df.drop_duplicates(subset ="Title",
                     keep = False, inplace = True)

img_segmentation=spine_segment.get_book_lines("./images/1.jpg")
for img_s in img_segmentation:
    title=tr.text_detection_spines(img_s,False)
    if(len(title)>0):
        print("Original : ",title)                 
    dic={}
    line=title.strip().split()
    for word in line:
        a=df[df['Title'].str.contains(word,case=False)][['Title']]


        for i in a["Title"]:
            if i not in dic:
                dic[i]=0
            else:
                dic[i]=dic[i]+len(word)
    # print(dic)
    try:
        maxtitle = max(zip(dic.values(), dic.keys()))[1]
        b=df.index[df["Title"]==maxtitle]
        b=b.tolist()
        ind=b[0]
        df.iat[ind,1]=df.iat[ind,1]+1
        print("Pridicted : ",maxtitle)
    except:
        print("No Title Found")
    print("------------------------------------")
df.to_csv("data1.csv", index=False)
# print(df.iat[0,0])