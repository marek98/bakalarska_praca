import cv2
from skimage import io
import xml.etree.ElementTree as ET
import numpy as np

from ParkingSpace import ParkingSpace


'''
Trieda Image reprezentuje jeden obrázok parkoviska
    - Združuje v sebe všetky parkoviacie miesta, ktoré sa na obrázku nachádzajú
'''
class Image:

    RED = (0, 0, 255)
    GREEN = (0, 255, 0)
    YELLOW = (0, 255, 255)

    def __init__(self, path, name):
        # Set fileName
        fileName = f"{path}/{name}"
        self.fileName = fileName
        self.name = name

        # Get image (path, mode 0 = grayscale)
        self.loadImage(fileName)

        # Get parking spaces for image
        try:
            self.loadParkingSpaces(fileName)
        except Exception as e:
            print(f"{fileName} - {e}")


    '''
    Načíta obrázok s {fileName} adresov a {ext} formátom
    '''
    def loadImage(self, fileName, ext="jpg"):
            self.image = cv2.imread(f"{fileName}.{ext}", cv2.IMREAD_COLOR)

    
    '''
    Načíta ParkingSpaces pre odpovedajúci obrázok
    '''
    def loadParkingSpaces(self, fileName):
        # Set parkingSpaces to empty array
        self.parkingSpaces = []

        # Error information
        errorCounter = 0

        # Get root of XML file which contains all parking spaces for image
        # Then get all data for parking space and save them in ParkingSpace class
        root = ET.parse(f"{fileName}.xml").getroot()

        for space in root:
            try:
                self.parkingSpaces.append(ParkingSpace(space, self.image, self.name))
            except:
                errorCounter += 1
                continue

        # If was error, raise error to get info
        if(errorCounter):
            raise Exception(f"Pocet chyb - {errorCounter}")


    '''
    Vráti všetky parkovacie miesta, ktoré má pre daný obrázok načítané
    '''
    def getParkingSpaces(self):
        return self.parkingSpaces

    
    '''
    Vráti všetky miesta, pre ktoré spravil klasifikátor zlú klasifikáciu
    '''
    def getParkingSpacesWithWrongPrediction(self):
        parkingSpacesWithWrongPrediction = []
        for parkingSpace in self.parkingSpaces:
            if(not parkingSpace.isPredictionCorrect()):
                parkingSpacesWithWrongPrediction.append(parkingSpace) 
        
        return parkingSpacesWithWrongPrediction


    '''
    Vráti počet parkovacích miest z chybnou klasifikáciou
    '''
    def howManyWrongPredictions(self):
        return len(self.getParkingSpacesWithWrongPrediction())


    '''
    Vráti štatistiku pre daný obrázok (konkrétne ju potom ráta Dataset)
    '''
    def getStatistics(self):
        total = len(self.parkingSpaces)
        occupied = 0
        for parkingSpace in self.parkingSpaces:
            if(parkingSpace.isOccupied()):
                occupied += 1
        
        return (total, occupied, total - occupied)


    '''
    Vykreslí okraje okolo parkovacích miest
    '''
    def drawParkingSpacesOnImage(self):
        for parkingSpace in self.parkingSpaces:
            color = self.GREEN
            if(parkingSpace.isOccupied()):
                color = self.RED
            cv2.drawContours(self.image, [parkingSpace.getRotatedRectangleBox()], 0, color, 2)


    '''
    Vykreslí okraje okolo parkovacích miest na základe klasifikácie
    '''
    def drawParkingSpacesOnImageBasedOnPrediction(self):
        for parkingSpace in self.parkingSpaces:
            color = self.GREEN
            if(parkingSpace.isOccupiedByPrediction()):
                color = self.RED
            if(not parkingSpace.isPredictionCorrect()):
                color = self.YELLOW
            cv2.drawContours(self.image, [parkingSpace.getRotatedRectangleBox()], 0, color, 2)


    '''
    Ukáže obrázok parkoviska
    '''
    def showImage(self):
        cv2.imshow(self.fileName, self.image)
        cv2.waitKey(0)


    '''
    Uloží obrázok parkoviska
    '''
    def saveImage(self, how="EMPTY", dir="", format="png"):
        if(how == "PREDICTION"):
            self.drawParkingSpacesOnImageBasedOnPrediction()
        if(how == "CORRECT"):
            self.drawParkingSpacesOnImage()

        fileName = f"{dir}/{self.getImageName()}.{format}" 
        cv2.imwrite(fileName, self.image)

    
    '''
    Uloží obrázok parkoviska do {dir} priečinka a zároveň tam uloží aj obrázky pre všetky jeho chybne klasifikované parkovacie miesta
    '''
    def saveImageAsWorst(self, how="EMPTY", dir="", format="png"):
        self.saveImage(how, dir, format)

        parkingSpacesWithWrongPrediction = self.getParkingSpacesWithWrongPrediction()
        for parkingSpace in parkingSpacesWithWrongPrediction:
            parkingSpace.saveHOGImage(dir, format)


    '''
    Vráti meno obrázku
    '''
    def getImageName(self):
        return self.name







