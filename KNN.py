from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

'''
Wrapper pre implementáciu KNN v sklearn
'''
class KNN:

    def __init__(self, nNeighbors):
        self.knn = KNeighborsClassifier(n_neighbors=nNeighbors)


    '''
    Trénovanie na dátach z daného datasetu
    '''
    def train(self, dataset):
        # Get training data and labels
        trainingData, labels = dataset.getTrainingData()

        # Train
        self.knn.fit(trainingData, labels)


    '''
    Klasifikácia dát z daného datasetu
    '''
    def makePrediction(self, testData):
        predictions = []
        for parkingSpaceHOGDescriptor in testData:
            predictions.append(self.knn.predict([parkingSpaceHOGDescriptor])[0])

        return predictions


    '''
    Spraví klasifikáciu, tú nastaví parkovacím miestam v datasete a vypíše úspešnosť klasifikácie
    '''
    def predictAndSetPredictions(self, dataset):
        testData, actualOccupancy = dataset.getTrainingData()

        predictOccupancy = self.makePrediction(testData)
        dataset.setPredictions(predictOccupancy)

        # Model Accuracy: how often is the classifier correct?
        print("     - Uspesnost: ", metrics.accuracy_score(actualOccupancy, predictOccupancy))