#!/usr/bin/env python
# coding: utf-8

# ### Import needed packages to wrangle data

# In[1]:


import pandas as pd
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import RegexpTokenizer
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np


# In[ ]:





# # Clinton vs. Trump Dataset

# In[2]:


ClintonTrump = pd.read_csv("C:/Users/deand/Desktop/Courtney/ClintonTrump.csv")
ClintonTrump.head()


# ### Renaming columns

# In[3]:


ClintonTrump.rename(columns={'is_retweet':'retweets', 'handle':'username','text':'tweet','entities':'source'},inplace=True)


# In[4]:


ClintonTrump.dtypes


# In[5]:


ClintonTrump['retweets'] = ClintonTrump['retweets'].astype(str)


# ### Recoding 'retweets' column

# In[6]:


def retweets (series):
    if series == "True":
        return 0
    if series == "False":
        return 1


# In[7]:


ClintonTrump['retweetsR'] = ClintonTrump['retweets'].apply(retweets)


# In[8]:


ClintonTrump.head()


# ### Subsetting data to keep only the necessary columns

# In[9]:


ClintonTrump1 = ClintonTrump[['id','username','tweet','time','source','retweetsR','retweet_count']]


# In[10]:


ClintonTrump1.head()


# ### Splitting 'time' column

# In[11]:


ClintonTrump2 = ClintonTrump1['time'].str.split('-',expand=True).rename(columns = lambda x: "date"+str(x+1))


# In[12]:


ClintonTrump2.head()


# In[13]:


dateTime = ClintonTrump2['date3'].str.split('T',expand=True).rename(columns = lambda x: "time"+str(x+1))
dateTime.head()


# In[14]:


ClintonTrump3 = ClintonTrump2[['date1','date2']]


# ### Adding time and date columns back to original dataframe

# In[15]:


ClintonTrump4 = pd.concat([ClintonTrump1,ClintonTrump3,dateTime],axis=1)


# In[16]:


ClintonTrump4.head()


# ### Renaming date and time columns

# In[17]:


ClintonTrump5 = ClintonTrump4.rename(columns={'date1':'year','date2':'month','time1':'date','time2':'Time'})


# In[18]:


ClintonTrump5 = ClintonTrump5.drop(['time'], axis=1)


# ### Final version of Clinton vs. Trump dataset

# In[19]:


ClintonTrump5.head()


# # Biden Dataset

# In[20]:


Biden = pd.read_csv("C:/Users/deand/Desktop/Courtney/Biden.csv")
Biden.head()


# ### Splitting 'timestamp' column

# In[21]:


Biden1 = Biden['timestamp'].str.split('-',expand=True).rename(columns = lambda x: "date"+str(x+1))


# In[22]:


Biden1.tail()


# In[23]:


Biden2 = Biden1['date3'].str.split(' ',expand=True).rename(columns = lambda x: "time"+str(x+1))


# In[24]:


Biden2.head()


# In[25]:


Biden3 = Biden1[['date1','date2']]


# ### Adding time and date columns back to original dataframe

# In[26]:


Biden4 = pd.concat([Biden,Biden3,Biden2],axis=1)


# In[27]:


Biden4.head()


# ### Renaming columns and dropping unneeded columns

# In[28]:


Biden5 = Biden4.rename(columns={'date1':'year','date2':'month','time1':'date','time2':'Time','link':'source'})


# In[29]:


Biden5 = Biden5.drop(columns=['timestamp','likes'])


# ### Duplicating 'retweets' column data into new column

# In[30]:


Biden5['retweet_count'] = Biden5['retweets']


# ### Recoding 'retweets' column

# In[31]:


def Retweets (series):
    if series > 0:
        return 0
Biden5['retweetsR'] = Biden5['retweets'].apply(Retweets)


# In[32]:


Biden5 = Biden5.drop(columns=['retweets'])


# ### Final version of Biden dataset

# In[33]:


Biden5.head()


# # Appending the dataframes

# In[34]:


AllTweets = pd.concat([Biden5, ClintonTrump5], ignore_index=True)


# In[35]:


AllTweets.head()


# # Final dataframes

# In[36]:


AllTweets


# ### Dropping Colomns I don't need in each Dataset

# In[37]:


FinalAllTweets = AllTweets.drop(columns=['source', 'month','date', 'Time' ])


# In[38]:


FinalAllTweets


# In[ ]:





# In[39]:


Biden5


# In[40]:


FinalBiden5 = Biden5.drop(columns=['source', 'month','date', 'Time'])


# In[41]:


FinalBiden5


# In[42]:


ClintonTrump5


# In[43]:


FinalClintonTrump5 = ClintonTrump5.drop(columns=['month','date', 'Time', 'source'])


# In[44]:


FinalClintonTrump5


# # --------- NLP Sentiment Analysis ----------

# ### import packages 
# ### Terminal / Anaconda Navigator: conda install -c conda-forge textblob

# In[45]:


from textblob import TextBlob
from bs4 import BeautifulSoup
import re
import nltk
import nltk.data
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer,PorterStemmer
from nltk.corpus import stopwords
lemmatizer = WordNetLemmatizer()


# ### Remove HTML, Emails, Stop Words, Weblinks, Whitespace, Numbers, Special Characters, NAN, and Lowercase Text

# In[46]:


def preprocess(sentence):
   sentence=str(sentence)
   sentence = sentence.lower()
   sentence=sentence.replace('{html}',"") 
   cleanr = re.compile('<.*?>')
   cleantext = re.sub(cleanr, '', sentence)
   rem_url=re.sub(r'http\S+', '',cleantext)
   rem_num = re.sub('[0-9]+', '', rem_url)
   tokenizer = RegexpTokenizer(r'\w+')
   tokens = tokenizer.tokenize(rem_num)  
   filtered_words = [w for w in tokens if len(w) > 2 if not w in stopwords.words('english')]
   stem_words=[stemmer.stem(w) for w in filtered_words]
   lemma_words=[lemmatizer.lemmatize(w) for w in stem_words]
   return " ".join(filtered_words)
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import TweetTokenizer
stemmer = PorterStemmer() 


# ### Adding cleanText to all datasets

# In[47]:


FinalAllTweets['cleanText']=FinalAllTweets['tweet'].map(lambda s:preprocess(s)) 


# In[48]:


FinalBiden5['cleanText']=FinalBiden5['tweet'].map(lambda s:preprocess(s))


# In[49]:


FinalClintonTrump5['cleanText']=FinalClintonTrump5['tweet'].map(lambda s:preprocess(s))


# In[50]:


FinalAllTweets.head()


# In[51]:


FinalBiden5.head()


# In[52]:


FinalClintonTrump5.head()


# ### Removing Short Words from cleanText

# In[53]:


FinalAllTweets['cleanText'] = FinalAllTweets['cleanText'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))

FinalAllTweets.head(10)


# In[54]:


FinalBiden5['cleanText'] = FinalBiden5['cleanText'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))

FinalBiden5.head(10)


# In[55]:


FinalClintonTrump5['cleanText'] = FinalClintonTrump5['cleanText'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))

FinalClintonTrump5.head(10)


# ### Tokenize Clean Tweets

# In[56]:


tokenizedAllTweets = FinalAllTweets['cleanText'].apply(lambda x: x.split())
tokenizedAllTweets.head()


# In[57]:


tokenizedTweetBiden = FinalBiden5['cleanText'].apply(lambda x: x.split())
tokenizedTweetBiden.head()


# In[58]:


tokenizedTweetClintTrump = FinalClintonTrump5['cleanText'].apply(lambda x: x.split())
tokenizedTweetClintTrump.head()


# ### Steeming the veggies better known as the cleanText :)

# In[59]:


tokenizedAllTweets = tokenizedAllTweets.apply(lambda x: [stemmer.stem(i) for i in x])

tokenizedAllTweets.head()


# In[60]:


tokenizedTweetBiden = tokenizedTweetBiden.apply(lambda x: [stemmer.stem(i) for i in x])

tokenizedTweetBiden.head()


# In[61]:


tokenizedTweetClintTrump = tokenizedTweetClintTrump.apply(lambda x: [stemmer.stem(i) for i in x])

tokenizedTweetClintTrump.head()


# ## Bar Graphing Words

# In[62]:


FinalAllTweets['cleanText'].value_counts()[:20].plot(kind='barh')


# In[63]:


FinalBiden5['cleanText'].value_counts()[:20].plot(kind='barh')


# In[64]:


FinalClintonTrump5['cleanText'].value_counts()[:20].plot(kind='barh')


# # Sentiment

# In[ ]:





# In[65]:


pol = lambda x: TextBlob(x).sentiment.polarity
sub = lambda x: TextBlob(x).sentiment.subjectivity


# ### All Tweets Combined

# In[66]:


FinalAllTweets['polarity'] = FinalAllTweets['tweet'].apply(pol)
FinalAllTweets['subjectivity'] = FinalAllTweets['tweet'].apply(sub)


# In[67]:


FinalAllTweets.tail()


# In[68]:


FinalBiden5['polarity'] = FinalBiden5['tweet'].apply(pol)
FinalBiden5['subjectivity'] = FinalBiden5['tweet'].apply(sub)


# In[69]:


FinalBiden5.tail()


# In[70]:


FinalClintonTrump5['polarity'] = FinalClintonTrump5['tweet'].apply(pol)
FinalClintonTrump5['subjectivity'] = FinalClintonTrump5['tweet'].apply(sub)


# In[71]:


FinalClintonTrump5.tail()


# In[72]:


FinalAllTweets.describe()


# In[ ]:





# In[ ]:





# In[73]:


FinalAllTweets.dtypes


# ## Polarity V Subjectivity

# In[83]:


x = FinalAllTweets.polarity
y = FinalAllTweets.subjectivity
fig, ax = plt.subplots()


ax.scatter(x, y, alpha=0.3, cmap = 'viridis')
ax.set_title('Polarity and Subjectivity')
ax.set_xlabel('<-- Negative -------- Positive -->')
ax.set_ylabel('<-- Facts -------- Opinions -->')



plt.show()


# In[84]:


x = FinalBiden5.polarity
y = FinalBiden5.subjectivity
fig, ax = plt.subplots()


ax.scatter(x, y, alpha=0.3, cmap = 'viridis')
ax.set_title('Polarity and Subjectivity')
ax.set_xlabel('<-- Negative -------- Positive -->')
ax.set_ylabel('<-- Facts -------- Opinions -->')



plt.show()


# In[85]:


x = FinalAllTweets.polarity
y = FinalAllTweets.subjectivity
fig, ax = plt.subplots()


ax.scatter(x, y, alpha=0.3, cmap = 'viridis')
ax.set_title('Polarity and Subjectivity')
ax.set_xlabel('<-- Negative -------- Positive -->')
ax.set_ylabel('<-- Facts -------- Opinions -->')



plt.show()


# In[ ]:





# ### Making a Word Cloud for Retweets (with Twitter symbol)

# #### Import Packages 

# In[77]:


from wordcloud import WordCloud,ImageColorGenerator
from PIL import Image
import urllib
import requests


# In[107]:


allRetweets = ' '.join(text for text in FinalAllTweets['cleanText'][FinalAllTweets['retweetsR']==1])


# #### combinig the image with the dataset

# In[108]:


Mask = np.array(Image.open(requests.get('http://clipart-library.com/image_gallery2/Twitter-PNG-Image.png', stream=True).raw))


# #### We use the ImageColorGenerator library from Wordcloud 
# #### Here we take the color of the image and impose it over our wordcloud

# In[109]:


image_colors = ImageColorGenerator(Mask)


# #### Now we use the WordCloud function from the wordcloud library 

# In[110]:


wc = WordCloud(background_color='black', height=1500, width=4000,mask=Mask).generate(allRetweets)


# ## Plotting Word Cloud (wish me luck!)

# In[112]:


plt.figure(figsize=(10,20))

plt.imshow(wc.recolor(color_func=image_colors),interpolation="hamming")

plt.axis('off')
plt.show()


# In[113]:


allRetweets1 = ' '.join(text for text in FinalAllTweets['cleanText'][FinalAllTweets['retweetsR']==0])


# In[114]:


wc1 = WordCloud(background_color='black', height=1500, width=4000,mask=Mask).generate(allRetweets1)


# In[115]:


plt.figure(figsize=(10,20))

plt.imshow(wc1.recolor(color_func=image_colors),interpolation="hamming")

plt.axis('off')
plt.show()


# In[ ]:





# In[ ]:





# In[116]:


allRetweetsClintTrump = ' '.join(text for text in FinalClintonTrump5['cleanText'][FinalClintonTrump5['retweetsR']==1])


# In[117]:


wcClintTrump = WordCloud(background_color='black', height=1500, width=4000,mask=Mask).generate(allRetweetsClintTrump)


# In[118]:


plt.figure(figsize=(10,20))

plt.imshow(wcClintTrump.recolor(color_func=image_colors),interpolation="hamming")

plt.axis('off')
plt.show()


# In[119]:


allRetweetsClintTrump1 = ' '.join(text for text in FinalClintonTrump5['cleanText'][FinalClintonTrump5['retweetsR']==0])


# In[120]:


wcClintTrump1 = WordCloud(background_color='black', height=1500, width=4000,mask=Mask).generate(allRetweetsClintTrump1)


# In[121]:


plt.figure(figsize=(10,20))

plt.imshow(wcClintTrump1.recolor(color_func=image_colors),interpolation="hamming")

plt.axis('off')
plt.show()


# In[ ]:





# In[ ]:





# In[122]:


allRetweetsBiden = ' '.join(text for text in FinalBiden5['cleanText'][FinalBiden5['retweetsR']==0])


# In[123]:


wcBiden = WordCloud(background_color='black', height=1500, width=4000,mask=Mask).generate(allRetweetsBiden)


# In[124]:


plt.figure(figsize=(10,20))

plt.imshow(wcBiden.recolor(color_func=image_colors),interpolation="hamming")

plt.axis('off')
plt.show()


# ### extract pos and neg hashtags from  combined tweets 

# In[125]:


def Hashtags_Extract(x):
    hashtags=[]
    
  
    for i in x:
        ht = re.findall(r'#(\w+)',i)
        hashtags.append(ht)
    
    return hashtags


# In[126]:


positiveAllTweets = Hashtags_Extract(FinalAllTweets['tweet'][FinalAllTweets['retweetsR']==0])

positiveAllTweets


# In[127]:


positiveUnnestAll = sum(positiveAllTweets,[])

positiveUnnestAll


# In[128]:


negativeAllTweet = Hashtags_Extract(FinalAllTweets['tweet'][FinalAllTweets['retweetsR']==1])

negativeAllTweet


# In[129]:


negativeUnnestAll = sum(negativeAllTweet,[])
negativeUnnestAll


# ### counting frequency of pos hashtags then graphing

# In[130]:


positiveFreqAll = nltk.FreqDist(positiveUnnestAll)

positiveFreqAll


# In[131]:


positiveDF = pd.DataFrame({'Hashtags':list(positiveFreqAll.keys()),'Count':list(positiveFreqAll.values())})

positiveDF.head(10)


# In[132]:


positivePlotAll = positiveDF.nlargest(20,columns='Count')

sns.barplot(data=positivePlotAll,y='Hashtags',x='Count')
sns.despine()


# ### now, doing the same for negative

# In[133]:


negativeFreqAll = nltk.FreqDist(negativeUnnestAll)

negativeFreqAll


# In[134]:


negativeDF = pd.DataFrame({'Hashtags':list(negativeFreqAll.keys()),'Count':list(negativeFreqAll.values())})

negativeDF.head(10)


# In[135]:


negativePlotAll = negativeDF.nlargest(20,columns='Count')

sns.barplot(data=negativePlotAll,y='Hashtags',x='Count')
sns.despine()


# In[ ]:





# In[ ]:





# ## Hovering Scatterplot

# #### import packages 

# In[136]:


import plotly
import plotly.express as px


# In[137]:


df = px.data.gapminder()
fig = px.scatter(FinalAllTweets, x="polarity", y="subjectivity", size="retweet_count", color="username",
           hover_name="cleanText", log_x=True, size_max=80)
fig.show()


# In[ ]:





# In[138]:


df = px.data.gapminder()
fig = px.scatter(FinalBiden5, x="polarity", y="subjectivity", animation_frame="year", animation_group="tweet",
           size="retweet_count", color="username", hover_name="cleanText", facet_col="username",
           log_x=True, size_max=80, range_x=[1,1], range_y=[1,1])
fig.show()


# In[139]:


df = px.data.gapminder()
fig = px.scatter(FinalClintonTrump5, "polarity", y="subjectivity", animation_frame="year", animation_group="tweet",
           size="retweet_count", color="username", hover_name="cleanText", facet_col="username",
           log_x=True, size_max=80, range_x=[1,1], range_y=[1,1])
fig.show()


# ## Sentiment Analysis

# ### importing packages 

# In[140]:


import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import collections

import tweepy as tw
import nltk
from nltk.corpus import stopwords
import re
import networkx


# ### gaining access to my consimer keys and access tokens

# In[141]:


consumerKey = "w4pKhVvWCjCbY32tNmzCimIJj"
consumerSecret = "jqjiVlfRBe3n03xJZtgY6suqMVNGqzjjYiSTYoADkdm0AwVaaW"
accessToken = "1301574480037392390-2yrsX3PY1wlHHeQjfWuLAILXjhMl82"
accessTokenSecret = "IeIMX7ub7VxgD44ProDrLIx9PlqSgbaq4iRqHxL2fk48c"


# In[142]:


auth = tw.OAuthHandler(consumerKey, consumerSecret)
auth.set_access_token(accessToken, accessTokenSecret)
api = tw.API(auth, wait_on_rate_limit=True)


# #### Looking for Joe Biden Tweets

# In[143]:


search_term = "@JoeBiden"

tweets = tw.Cursor(api.search,
                   q=search_term,
                   lang="en",
                   since='2012-01-01').items(1000)

allTweetsBiden6 = [tweet.text for tweet in tweets]

allTweetsBiden6[:10]


# #### remove URLS

# In[144]:


def removeUrl(txt):
    """Replace URLs found in a text string with nothing 
    (i.e. it will remove the URL from the string).

    Parameters
    ----------
    txt : string
        A text string that you want to parse and remove urls.

    Returns
    -------
    The same txt string with url's removed.
    """

    return " ".join(re.sub("([^0-9A-Za-z \t])|(\w+:\/\/\S+)", "", txt).split())


# In[145]:


allTweetsNoURLS = [removeUrl(tweet) for tweet in allTweetsBiden6]
allTweetsNoURLS[:10]


# #### text cleanup

# In[146]:


allTweetsNoURLS[0].split()


# In[147]:


allTweetsNoURLS[0].lower().split()


# In[148]:


stopWords = set(stopwords.words('english'))


list(stopWords)[0:10]


# ### Calculate and Plot word Freq

# In[149]:


wordsTweets = [tweet.lower().split() for tweet in allTweetsNoURLS]
wordsTweets[:2]


# In[150]:


wordsTweets[0]


# In[151]:


allWords = list(itertools.chain(*wordsTweets))

counts = collections.Counter(allWords)

counts.most_common(15)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




