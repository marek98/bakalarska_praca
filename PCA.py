import sklearn
from sklearn.decomposition import PCA

'''
Wrapper pre implementáciu PCA v sklearn
'''
class PCA:

    def __init__(self, numberOfComponents=2):
        self.pca = sklearn.decomposition.PCA(n_components=numberOfComponents)


    '''
    Vráti upravené dáta (majú dimenziu podľa toho, ako bolo PCA initnuté)
    '''
    def fitAndTransform(self, data):
        return self.pca.fit_transform(data)