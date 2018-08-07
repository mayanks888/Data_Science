#https://radimrehurek.com/gensim/models/word2vec.html
import tensorflow as tf
import  nltk
from nltk import word_tokenize,sent_tokenize
import gensim
from gensim.models.word2vec import  Word2Vec
import  pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from bokeh.io import output_notebook
from bokeh.plotting import show, figure

'''my_model=Word2Vec.load('my_gutenberg_word_to_vec.w2v')
print(my_model['dog'])#print value vector for dog which is 64 dimension (previously defined remember)

print(my_model.most_similar('dog'))#list of token which are near to vector dog
print(my_model.most_similar('day'))
print(my_model.similarity('dog','cat'))#
print(my_model.most_similar(positive=["king",'man'],negative=['queen'],topn=30))#to check into top 30 list of scores
# husband-man+woman=daughter
print(my_model.most_similar(positive=["husband",'women'],negative=['man']))

total_token=my_model.wv.vocab
print(len(total_token))#total number of tokens/ word in wordtovec file
all_word=my_model[my_model.wv.vocab]#extract all 64 dimension of all the word/tokens form model
1
# print('total keys are',total_token.keys())
# print(total_token.items())
# lets converts 64 dimnesion into small dimension with help of tsne
tsne=TSNE(n_components=2,n_iter=1000)#n_components=convert into dimensions
x_2d=tsne.fit_transform(all_word)

cord_df=pd.DataFrame(x_2d,columns=['X','Y'])
cord_df['token']=my_model.wv.vocab.keys()#its a great way to extract name from dict or anything use it in future
cord_df.head()
cord_df.to_csv('myraw_word_tovec_2d.csv',index=False)'''



#now lets visulaise your 2 d graph
my_data=pd.read_csv('myraw_word_tovec_2d.csv')
my_data.plot.scatter('X','Y',figsize=(8,8),marker='.',s=10,alpha=0.2)
# plt.show()

# This is a great way to visualise wordtovec
sample_data=my_data.sample(n=400)
p = figure(plot_width=800, plot_height=800)
_ = p.text(x=sample_data.X, y=sample_data.Y, text=sample_data.token)
show(p)
