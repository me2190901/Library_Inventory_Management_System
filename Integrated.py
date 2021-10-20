import spine_segment
import glob as gb
import text_recognize as tr
import pandas as pd
# from IPython.display import display, HTML
def prediction(loc,debug=False):
    df=pd.read_csv(".\data1.csv")
    df.drop_duplicates(subset ="Title",
                     keep = False, inplace = True)
    if (debug):
        l=[["Afewwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww",1],["B",2]]
        return l
    img_segmentation=spine_segment.get_book_lines(loc)
    l=[]
    total={}
    for img_s in img_segmentation:
        title=tr.text_detection_spines(img_s,False)
        # if(len(title)>0):
        #     print("Original : ",title)                 
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
            if(maxtitle in total):
                total[maxtitle]=total[maxtitle]+1
            else:
                total[maxtitle]=1
            # l.append([maxtitle,df.iat[ind,1]])
            # print("Pridicted : ",maxtitle)
        except:
            # print("No Title Found")
            print("No Title Found")
    for key in total:
        l.append([key,total[key]])
    df.to_csv("data1.csv", index=False)
        # print("------------------------------------")
    return l

def search(line,number,debug=False):
    df=pd.read_csv(".\data1.csv")
    df.drop_duplicates(subset ="Title",
                     keep = False, inplace = True)
    dic={}
    line1=line
    line=list(line.strip().split())
    for word in line:
        # print(word)
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
        if(len(line1)/len(maxtitle)<0.6):
            raise Exception("Title not detected correctly")
        b=b.tolist()
        ind=b[0]
        df.iat[ind,1]=df.iat[ind,1]+number
        if(not debug):
            df.to_csv("data1.csv", index=False)
        # l.append([maxtitle,df.iat[ind,1]])
        # print("Pridicted : ",maxtitle)
    except:
        # print("No Title Found")
        # print(line1,number)
        df1 = pd.DataFrame([(line1, number)],
                columns=('Title', 'Checked')
                        )
        df3 = pd.concat([df, df1], ignore_index = True)
        df3.reset_index()
        if(not debug):
            df3.to_csv("data1.csv", index=False)
    
    # print(dic)
        # print("------------------------------------")
