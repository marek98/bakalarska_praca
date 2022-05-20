from Dataset import Dataset
from SVM import SVM
from PCA import PCA
from KNN import KNN
import time
import os


# Cesta, kde sa budu ukladat obrazky
RESULT_DIR = "results"


# Inicializacia casu
startTime = time.time()
stepStartTime = startTime


# Pomocna funkcia, ktora vytvori priecinok ak neexistuje
def createDirIfNotExist(path):
    if(not os.path.exists(path)):
        os.makedirs(path)


# Pomocna funkcia, ktora pre dany dataset robi predikciu za pomoci SVM
def trainAndPredictSVM(trainingDataset, testingDataset, info):
    for kernelFunction in ["linear", "poly", "sigmoid", "rbf"]:

        print(f"SVM_{kernelFunction}_{info}")
        svm = SVM(kernelFunction)

        stepStartTime = time.time()
        svm.train(trainingDataset)
        print("     - Trening za {:.2f} sekund/y".format(time.time() - stepStartTime))

        stepStartTime = time.time()
        svm.predictAndSetPredictions(testingDataset)
        print("     - Predikcia za {:.2f} sekund/y".format(time.time() - stepStartTime))

        path = f"{RESULT_DIR}/SVM_{kernelFunction}_{info}"
        createDirIfNotExist(path)
        testingDataset.saveWorstImage("PREDICTION", path, "png")


# Pomocna funkcia, ktora pre dany dataset robi predikciu za pomoci KNN
def trainAndPredictKNN(trainingDataset, testingDataset, info):
    for kNeighbors in [1,3,5,7,9,11]:
        
        print(f"KNN_{kNeighbors}_{info}")
        svm = KNN(kNeighbors)

        stepStartTime = time.time()
        svm.train(trainingDataset)
        print("     - Trening za {:.2f} sekund/y".format(time.time() - stepStartTime))

        stepStartTime = time.time()
        svm.predictAndSetPredictions(testingDataset)
        print("     - Predikcia za {:.2f} sekund/y".format(time.time() - stepStartTime))

        path = f"{RESULT_DIR}/KNN_{kNeighbors}_{info}"
        createDirIfNotExist(path)
        testingDataset.saveWorstImage("PREDICTION", path, "png")



# Nacitanie datasetu
print("Nacitavam datasety")
tDataset = Dataset("datasets/training")    # Trenovacia 
print("     - Trenovaci dataset nacitany")
pDataset = Dataset("datasets/validating")        # Validacna/Testovacia
print("     - Validacny/Testovaci dataset nacitany")

# Volame getTrainingData() aby sme pri trenovani/predikcii mali uz vyratane HOG deskriptory
print("     - Zacinam vypocitavat HOG deskriptory")
tDataset.getTrainingData()
print("     - HOG deskriptory pre trenovaci dataset vyratane")  
pDataset.getTrainingData()
print("     - HOG deskriptory pre validacny/testovaci dataset vyratane")  
print("------ Datasety nacitane za {:.2f} sekund/y".format(time.time() - stepStartTime))
stepStartTime = time.time()

print("------ Pocet parkovacich miest v treningovom datasete - " + str(len(tDataset.getParkingSpaces())))
print("------ Pocet parkovacich miest v validacnom/testovacom datasete - " + str(len(pDataset.getParkingSpaces())))




# Zakladne
trainAndPredictSVM(tDataset, pDataset, "")
trainAndPredictKNN(tDataset, pDataset, "")


# S PCA
for i in [1,2,4,8,16,32,64,128,256,512,1024,2048]:
    print(f"\nNastavuje PCA({i})")
    tDataset.setPCA(PCA(i))
    pDataset.setPCA(PCA(i))

    print("     - Zacinam vypocitavat PCA data")
    tDataset.getTrainingData()
    print("     - PCA data pre trenovaci dataset vyratane")
    pDataset.getTrainingData()
    print("     - PCA data pre validacny/testovaci dataset vyratane")

    trainAndPredictSVM(tDataset, pDataset, f"PCA_{i}")
    trainAndPredictKNN(tDataset, pDataset, f"PCA_{i}")


