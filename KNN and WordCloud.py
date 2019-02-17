import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame 
import nltk
import regex as re
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn import neighbors
from scipy.spatial.distance import cosine
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import string
from wordcloud import WordCloud, STOPWORDS
from sklearn.metrics import mean_squared_error
df=pd.read_csv("reviews.csv")#Reading the file from csv
count = df.groupby("asin", as_index=False).count()#counting the id
mean = df.groupby("asin", as_index=False).mean()#calculating the mean
dfMerged = pd.merge(df, count, how='right', on=['asin'])#merging the data with the count
dfMerged["totalReviewers"] = dfMerged["reviewerID_y"]
dfMerged["overallScore"] = dfMerged["overall_x"]
dfMerged["summaryReview"] = dfMerged["summary_x"]
dfNew = dfMerged[['asin','summaryReview','overallScore',"totalReviewers"]]
dfMerged = dfMerged.sort_values(by='totalReviewers', ascending=False)
dfCount = dfMerged[dfMerged.totalReviewers >= 100]# sorting data whose reviews are above 100
dfProductReview = df.groupby("asin", as_index=False).mean()
ProductReviewSummary = dfCount.groupby("asin")["summaryReview"].apply(list)
ProductReviewSummary = pd.DataFrame(ProductReviewSummary)
ProductReviewSummary.to_csv("ProductReviewSummary.csv")#Summarizing the review given by a user for a product
df3 = pd.read_csv("ProductReviewSummary.csv")
df3 = pd.merge(df3, dfProductReview, on="asin", how='inner')
df3 = df3[['asin','summaryReview','overall']]#calculating the rating between summary and productReview
regEx = re.compile('[^a-z]+')#Taking all the text between A to Z 
def cleanReviews(reviewText):
    reviewText = reviewText.lower()
    reviewText = regEx.sub(' ', reviewText).strip()#Removing the extra space from the text
    return reviewText
df3["summaryClean"] = df3["summaryReview"].apply(cleanReviews)
df3 = df3.drop_duplicates(['overall'], keep='last')
df3 = df3.reset_index()
reviews = df3["summaryClean"]
countVector = CountVectorizer(max_features = 300, stop_words='english')#countvectorize create tokens in the form of matrix   
transformedReviews = countVector.fit_transform(reviews)# Transforming the data into big bag of words
dfReviews = DataFrame(transformedReviews.A, columns=countVector.get_feature_names())
dfReviews = dfReviews.astype(int)
dfReviews.to_csv("dfReviews.csv")#Contains all the common words occurring and how many times it is present 
X = np.array(dfReviews)
print len(X) # create train and test
tpercent = 0.9
tsize = int(np.floor(tpercent * len(dfReviews)))
dfReviews_train = X[:tsize]
dfReviews_test = X[tsize:]
lentrain = len(dfReviews_train)
lentest = len(dfReviews_test)
neighbor = NearestNeighbors(n_neighbors=3, algorithm='ball_tree').fit(dfReviews_train)
# Let's find the k-neighbors of each point in object X. To do that we call the kneighbors() function on object X.
distances, indices = neighbor.kneighbors(dfReviews_train)
#find most related products
for i in range(lentest):
    a = neighbor.kneighbors([dfReviews_test[i]])
    related_product_list = a[1]
    first_related_product = [item[0] for item in related_product_list]
    first_related_product = str(first_related_product).strip('[]')
    first_related_product = int(first_related_product)
    second_related_product = [item[1] for item in related_product_list]
    second_related_product = str(second_related_product).strip('[]')
    second_related_product = int(second_related_product)
    print ("Based on product reviews, for ", df3["asin"][lentrain + i] ," average rating is ",df3["overall"][lentrain + i])
    print ("The first similar product is ", df3["asin"][first_related_product] ," average rating is ",df3["overall"][first_related_product])
    print ("The second similar product is ", df3["asin"][second_related_product] ," average rating is ",df3["overall"][second_related_product])
    print ("-----------------------------------------------------------")
df5_train_target = df3["overall"][:lentrain]
df5_test_target = df3["overall"][lentrain:lentrain+lentest]
df5_train_target = df5_train_target.astype(int)
df5_test_target = df5_test_target.astype(int)
n_neighbors = 3
knnclf = neighbors.KNeighborsClassifier(n_neighbors, weights='distance')
knnclf.fit(dfReviews_train, df5_train_target)
knnpreds_test = knnclf.predict(dfReviews_test)
print classification_report(df5_test_target, knnpreds_test)
print accuracy_score(df5_test_target, knnpreds_test)
print mean_squared_error(df5_test_target, knnpreds_test)
cluster = df.groupby("overall")["summary"].apply(list)
cluster = pd.DataFrame(cluster)
cluster.to_csv("cluster.csv")
cluster1 = pd.read_csv("cluster.csv")
cluster1["summaryClean"] = cluster1["summary"].apply(cleanReviews)
stopwords = set(STOPWORDS)
def show_wordcloud(data, title = None):
    wordcloud = WordCloud(background_color='white', stopwords=stopwords, max_words=500, max_font_size=30, scale=3,
    random_state=1 # chosen at random by flipping a coin; it was heads
    ).generate(str(data))
    fig = plt.figure(1, figsize=(8, 8))
    plt.axis('off')
    if title: 
        fig.suptitle(title, fontsize=20)
        fig.subplots_adjust(top=2.3)
    plt.imshow(wordcloud)
    plt.show()
show_wordcloud(cluster1["summaryClean"][0], title = "")
show_wordcloud(cluster1["summaryClean"][1], title = "")
show_wordcloud(cluster1["summaryClean"][2], title = "")
show_wordcloud(cluster1["summaryClean"][3], title = "")
show_wordcloud(cluster1["summaryClean"][4], title = "")
