import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.preprocessing import Normalizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer

categories = ['comp.sys.mac.hardware', 'comp.graphics', 'sci.med', 'soc.religion.christian', 'talk.religion.misc']
text_data = fetch_20newsgroups(subset='all', categories=categories, shuffle=True, random_state=62)

print(len(text_data.data), len(text_data.target_names))


#Taking 10 files for Clustering
dataset = []
target = []
for i in range(0, 5):
    files=0;
    for j in range(0, 4853):
        if(text_data.target[j] == i):
            dataset.append(text_data.data[j])
            target.append(i)
            files = files + 1
        if(files == 2):
            break
print('No. of filess:', len(dataset))


vectorizer = TfidfVectorizer(max_df = 0.95, min_df = 0.05, max_features = 10000, stop_words = 'english', use_idf = True)
X = vectorizer.fit_transform(text_data.data)
print("No. of samples : ", X.shape[0])
print("No. of feature : ", X.shape[1])

classes = text_data.target
clusters = np.unique(classes).shape[0]
km = KMeans (n_clusters = clusters, init = 'k-means++', n_init = 1, verbose = 8, max_iter = 100)
km.fit(X)

print("Homogeneity : ",metrics.homogeneity_score(classes, km.labels_))
print("V-measure: ", metrics.v_measure_score(classes, km.labels_))