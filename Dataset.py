from cProfile import label
import os

import cv2
from Image import Image

'''
Trieda Dataset reprezentuje množinu obrázkov
'''
class Dataset:

    def __init__(self, path):
        self.pca = None

        # For storing data, help with time when one dataset is used for multiple times
        self.parkingSpaces = None
        self.data = None
        self.dataWithPCA = None
        
        # Load all images from {path} directory
        self.loadImages(path)


    '''
    Načíta všetky obrázky z {path} priečinka
    '''
    def loadImages(self, path):
        self.images = []

        # Get all files in directory
        # And get only JPG images, because every image in PUCPR is in .jpg format
        allFilesInDirectory = os.listdir(path)
        for fileNameWithExtension in allFilesInDirectory:
            fileName, extension = os.path.splitext(fileNameWithExtension)
            if(extension == ".jpg"):
                try:    
                    self.images.append(Image(path, fileName))
                except:
                    continue


    '''
    Ukáže všetky obrázky v datasete
    '''
    def showImages(self):
        for image in self.images:
            image.drawParkingSpacesOnImageBasedOnPrediction()
            image.showImage()
            cv2.waitKey(0)


    '''
    Vráti všetky parkovacie miesta v datasete
    '''
    def getParkingSpaces(self):
        if(self.parkingSpaces is None):
            self.parkingSpaces = []
            for image in self.images:
                self.parkingSpaces.extend(image.getParkingSpaces())
        
        return self.parkingSpaces


    '''
    Spočíta a na konci vypíše štatistiku o datasete
    '''
    def getStatistics(self):
        totalCounter = 0
        occupiedCounter = 0
        unoccupiedCounter = 0
        for image in self.images:
            total, occupied, unoccupied = image.getStatistics()
            totalCounter += total
            occupiedCounter += occupied
            unoccupiedCounter += unoccupied
        
        print(f"Celkovo má dataset parkovacích miest: {totalCounter}")
        print(f"    - Z toho obsadených: {occupiedCounter}")
        print(f"    - Z toho neobsadených: {unoccupiedCounter}")


    '''
    Vráti najhoršie hodnotený obrázok v datasete
    '''
    def getWorstImage(self):
        worstImage = None
        worstImageWrongPredictionsCounter = -1
        for image in self.images:
            actualImageWrongPredictionsCounter = image.howManyWrongPredictions()
            if(worstImageWrongPredictionsCounter == -1 or worstImageWrongPredictionsCounter < actualImageWrongPredictionsCounter):
                worstImageWrongPredictionsCounter = actualImageWrongPredictionsCounter
                worstImage = image
        
        return worstImage


    '''
    Uloží najhoršie hodnotený obrázok v datasete do {dir} priečinku a vo formáte {format}
    '''
    def saveWorstImage(self, how="EMPTY", dir="", format="png"):
        worstImage = self.getWorstImage()
        worstImage.saveImageAsWorst(how, dir, format)


    '''
    Vráti dáta v podobe (<vektor>, <obsadenosť>)
    Ak je nastavený PCA, najprv tieto dáta preženie cez PCA a až potom ich vráti
    '''
    def getTrainingData(self):
        def getTrainingData():
            # Get training data
            trainingData = []       # HOGDescriptor
            labels = []             # Occupancy

            for parkingSpace in self.getParkingSpaces():
                trainingData.append(parkingSpace.getHOGDescriptor())
                labels.append(int(parkingSpace.isOccupied()))

            return (trainingData, labels)


        # PCA is not set
        if(self.pca is None):
            if(self.data is None):
                self.data = getTrainingData()

            return self.data

        # PCA is set
        if(self.dataWithPCA is None):
            if(self.data is None):
                self.data = getTrainingData()
            trainingData, labels = self.data
            trainingData = self.pca.fitAndTransform(trainingData)
            self.dataWithPCA = (trainingData, labels)

        return self.dataWithPCA


    '''
    Nastav PCA, ktoré sa bude používať na redukciu dimenzie dát
    Pri nastavení nastav aj dataWithPCA = None, keďže sa pri behu môže zmeniť na iné PCA a potom by vracalo zlé výsledky (pre skorej nastavené)
    '''
    def setPCA(self, PCA):
        self.pca = PCA
        self.dataWithPCA = None


    '''
    Parkovacím miestam nastaví triedu, do ktorej ich klasifikoval klasifikátor
    '''
    def setPredictions(self, predictions):
        for index, parkingSpace in enumerate(self.getParkingSpaces()):
            parkingSpace.setPredictOccupied(predictions[index])
        



