from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler,MaxAbsScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from scipy.sparse import csr_matrix
import copy
import numpy as np

class DataModelingCrossFold:
    
    def __init__( self, corpus_cross: list, tense_cross: list, labels_cross: list,  skmodel, test_set_num=100 ):
        self.vectorizer = TfidfVectorizer()
        corp_flatten = []
        corp_tuple_inds = []
        tense_flatten = []
        ind_count = 0
       
        for i,x in enumerate(corpus_cross):
            x.apply(lambda x: corp_flatten.append(x))
            tense_cross[i].apply(lambda x: tense_flatten.append(x))
            corp_tuple_inds.append((ind_count,ind_count+len(x)))
            ind_count += len(x)
        
        self.X = self.vectorizer.fit_transform(corp_flatten)  
        if True:
            ## add tense count
            tense_arr = np.array([tense_flatten]).reshape(-1,len(tense_flatten[0]))
            tmax = tense_arr.max(0)
            tmin = tense_arr.min(0)
            tense_arr_scaled = (tense_arr-tmin)/(tmax-tmin)
            self.X = csr_matrix(np.hstack((self.X.toarray(),tense_arr_scaled)))
        self.y = labels_cross
        
        self.X_ind = [ self.X[i[0]:i[1]] for i in corp_tuple_inds]
        folds = len(corpus_cross)
        self.models = []
        for i in range(folds):
            x = csr_matrix(np.vstack([ self.X[corp_tuple_inds[jj][0]:corp_tuple_inds[jj][1]].toarray() for jj,x in enumerate(corpus_cross) if i!=jj]))
            y = np.hstack([ x.tolist() for jj,x in  enumerate(self.y) if i!=jj])
            skmodel_old = copy.deepcopy(skmodel)
            inds = np.arange(len(y))
            np.random.shuffle(inds)
            skmodel_old.fit(x[inds],y[inds])
            self.models.append(skmodel_old)
    
    def predict(self,vectors_cross):
        return [ self.models[i].predict(v) for i,v in enumerate(vectors_cross) ]

    def predict_probs(self,vectors_cross):
        return [ self.models[i].predict_proba(vectors) for v in vectors_cross ]



        # self.X_train = self.X[:-test_set_num]
        # self.y_train = self.y[:-test_set_num]

        # self.X_test = self.X[-test_set_num:]
        # self.y_test = self.y[-test_set_num:]

class DataModeling:
    
    def __init__( self, corpus: list, labels: list, skmodel, test_set_num=100 ):
        self.vectorizer = TfidfVectorizer()
        self.X = self.vectorizer.fit_transform(corpus)
        self.y = labels
        
        self.X_train = self.X[:-test_set_num]
        self.y_train = self.y[:-test_set_num]

        self.X_test = self.X[-test_set_num:]
        self.y_test = self.y[-test_set_num:]

        self.skmodel = skmodel.fit(self.X_train,self.y_train)


class KNNModeling(DataModelingCrossFold):
    def __init__(self, corpus: tuple,labels: list, n_neighbors=1,test_set_num=100):
        corpus, tense = corpus
        self.skmodel = KNeighborsClassifier( n_neighbors = n_neighbors )
        super().__init__(corpus , tense, labels,self.skmodel, test_set_num )

        # self.skmodel.fit(self.X_train, self.y_train)
    # def predict(self,vectors):
    #     return self.skmodel.predict(vectors)

    # def predict_probs(self,vectors):
    #     return self.skmodel.predict_proba(vectors)


class RandomForestModeling(DataModelingCrossFold):
    def __init__(self, corpus: list,labels: list, n_estimators=10,test_set_num=100):
        corpus, tense = corpus
        self.skmodel = RandomForestClassifier(n_estimators=n_estimators)
        super().__init__(corpus , tense,labels,self.skmodel,test_set_num )

    
    # def predict(self,vectors):
    #     return self.skmodel.predict(vectors)

    # def predict_probs(self,vectors):
    #     return self.skmodel.predict_proba(vectors)


class SVMModeling(DataModelingCrossFold):
    def __init__(self, corpus: list,labels: list, kernel='linear',gamma='auto',test_set_num = 100):
        corpus, tense = corpus
        self.skmodel = make_pipeline(MaxAbsScaler(), SVC(gamma=gamma,kernel=kernel))
        super().__init__(corpus , tense,labels,self.skmodel,test_set_num )

class GradBoostingModeling(DataModelingCrossFold):
    def __init__(self, corpus: list,labels: list, n_estimators=10,test_set_num=100):
        corpus, tense = corpus
        self.skmodel = GradientBoostingClassifier(n_estimators=5)
        super().__init__(corpus , tense,labels,self.skmodel,test_set_num )
    
    # def predict(self,vectors):
    #     return self.skmodel.predict(vectors)

    # def predict_probs(self,vectors):
    #     return self.skmodel.predict_proba(vectors)




    