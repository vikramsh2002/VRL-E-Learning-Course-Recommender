import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
st.set_page_config(
        page_title="VRL Course Recommender",
        page_icon="MyLogoLight.png",
        layout="centered"
    )

st.image("MyLogoLight.png")
head='''
        <style>
        h1{
            color: aquamarine;
            font-size:30;
            
        }
        h2{
            color:red;
            text-align:center;
            font-size:20;
        }
        h1:hover{
               color:blue;
               font-size:40px;
        }
        h2:hover{
            color: blue;
            font-size:50px;
        }
        #cname{
            
            margin:2px;
            padding:3px;
            margin-bottom:5px;
            border-radius:10%;
            background-color: aqua;
            color: blue;
            animation: 20s mt infinite;
            text-align:center;
        }
        @keyframes mt
        {
            0%{
               background-color: aqua; 
            }
            30%{
                background-color: pink; 
            }
            60%{
                background-color: lime; 
            }
        }
        #cname:hover{
            margin:6px;
            padding:9px;
            margin-bottom:9px;
            font-size:10px;
        }
        button{
        border: 1px solid grey;
        border-radius: 10%;
        padding: 10px 50px; 
        background-color: white;
        text-align: center;
        font-size: 14px;
        }
    button:hover{
        border:1px solid red;
        color: red;
        }

    </style>
    <body>
    <center> <h1> E-Learning Courses </h1> </center>  <h2>Recommender Engine</h2>  
    </body>
        
'''
st.markdown(head,unsafe_allow_html=True)

@st.cache
def Genearte_simm(df):
    # Code to make matrix
    
    no_feat=4000
    cv= CountVectorizer(max_features=no_feat,stop_words="english")
    word_arr = cv.fit_transform(df["Tags"]).toarray()
    simmi_mat = cosine_similarity(word_arr)
    simmi_df = pd.DataFrame(simmi_mat, columns=df["Course Name"],index=df["Course Name"])
    return simmi_df

def Recommender(name,df):
    
    simmi_df = Genearte_simm(df)
    top12_=simmi_df[name].reset_index()
    top12_=top12_.sort_values(by=name,ascending=False)[1:13]["Course Name"]
    #print("Top 12 ",top12_)
    top12_details=df[df["Course Name"].isin(top12_)]
    top12_details = top12_details.sort_values(by="Rating",ascending=False)
    return top12_details
def sublayout(cname):
    s = '''
       <div id="cname">
       <blockquote> {} </blockquote><div>
       </body>
    
    
    '''.format(cname)
    return s
def button(url):
    s='''
    <a href="{}" target="_blank" ><button> Go to Course </button></a>
    
    '''.format(url)
    return s


def layout(info,ncol=2):
    x = st.columns(ncol)
    try:
        for i in range(0,12,ncol):
            for j in range(ncol):
                with x[j]:
                    data=info.iloc[i+j]
                    cname =sublayout(data["Course Name"])
                    st.markdown(cname,unsafe_allow_html=True)
                    
                    # more info
                    with st.expander("More Info"):
                        # info
                        st.markdown("###### University Name/ Company Name")
                        st.markdown("{}".format(data["University"])) 
                        st.markdown("###### Level of Course")
                        st.markdown( "{}".format(data["Difficulty Level"]))
                        st.markdown("###### Skills")
                        sk=[i.capitalize() for i in (data["Skills"]).split(" ")]
                        skills=", ".join(sk)
                        st.markdown( "{}".format(skills))
                        
                        #Rating
                        s=""
                        for r in range(int(data["Rating"])):
                            s+=":star: "
                        st.markdown("######  Course Rating")
                        st.markdown("###### "+s)

                        #button
                        url=data["Course URL"]
                        st.markdown(button(url),unsafe_allow_html=True)
    except:
        pass




            
df = pd.read_csv("CourseDetails.csv",index_col=0)
diff = list(df["Difficulty Level"].unique())
unv= list(df["University"].unique())
diff.append("All")
unv.append("All") 



# filters

with st.expander("Filters"):
    level = st.selectbox("Select the Level of Course ",diff,index=len(diff)-1,key="lvl_1")    
    if level not in ["All"]:
        df_diff = df[df["Difficulty Level"]==level]
        df = df_diff
        
    unv_name=st.selectbox("Select the University/Company",unv,index=len(unv)-1,key="unv_1")
        
    if unv_name not in ["All"]:
        df= df[df["University"]==unv_name]
    
    order=st.selectbox("Sort by Rating",["High-to-Low","Low-to-High","None"],index=2,key="ord_1")
    
    x = st.columns(3)    
    if x[1].button("Reset Filters"):
        st.experimental_rerun()
    

    



course_name=st.selectbox("Previous Completed or Liked Course or Topic of Interest",df["Course Name"].unique())
if course_name is not None:
    c=st.columns(3)

    if c[1].button("Show Similar \n & \n Interesting Courses"):
        similar_courses=Recommender(course_name,df)
        if len(similar_courses)>0:
            if order=="High-to-Low":
                similar_courses=similar_courses.sort_values("Rating",ascending=False)            
            elif order=="Low-to-High":
                similar_courses=similar_courses.sort_values("Rating",ascending=True)

            layout(similar_courses)


        else:
            st.warning("No similar course available under this filter. Please retry with something else")
            print("error")
else:
    st.warning("No Course Available Under this Filter Try something else Please")

    
    # layout
