import cv2
import numpy as np

from skimage.transform import rotate
from skimage.feature import hog
from skimage import data, exposure
import matplotlib.pyplot as plt

'''
Trieda ParkingSpace reprezentuje jedno konkrétne parkovacie miesto
    - Drží v sebe o ňom údaje vo formáte rotatedRectangle => má (x,y) koordináty stredu, šírku a výšku a uhoľ otočenia
    - Tak isto si drží obrázok tohto parkovacieho miesta (ten je vyseknutý z obrázku parkoviska + v prípade potreby je otočený, aby bol na výšku)
'''
class ParkingSpace:
    
    def __init__(self, parkingSpaceInfo, image, imageName):
        self.predictOccupied = None
        self.HOGDescriptor = None

        # Get parameters about parking space
        self.setInfo(parkingSpaceInfo)

        # Get cropped and resize image of parking space from parking lot image
        self.imageName = imageName
        self.setImage(image)


    '''
    Pre dané parkovacie miesto získa údaje z XML (dostáva príslušný node z XML súbora)
    '''
    def setInfo(self, parkingSpaceInfo):
        try:
            self.occupied = bool(int(parkingSpaceInfo.attrib["occupied"]))
            self.id = int(parkingSpaceInfo.attrib["id"])

            rotatedRectangleInfo = parkingSpaceInfo.find("rotatedRect")
            for atribute in rotatedRectangleInfo:
                if(atribute.tag == "center"):
                    self.center = (int(atribute.get("x")), int(atribute.get("y")))
                if(atribute.tag == "size"):
                    self.size = (int(atribute.get("w")), int(atribute.get("h")))
                if(atribute.tag == "angle"):
                    self.angle = int(atribute.get("d"))
        except:
            raise Exception("ParkingSpace.setInfo - Nastala chyba")
            

    '''
    Pre dané parkovacie miesto na základe údajov, ktoré o mieste má, získa výsek z obrázku parkoviska
    '''
    # https://jdhao.github.io/2019/02/23/crop_rotated_rectangle_opencv/
    # https://theailearner.com/tag/cv2-getperspectivetransform/
    def setImage(self, image, resizeSize=(64,128)):

        def getDestinationCoordinates(width, height):
            return np.array([[0, height],
                             [0, 0],
                             [width, 0],
                             [width, height]], dtype="float32")


        # Get rotated rectangle box around parking space
        box = self.getRotatedRectangleBox()

        # Get source and destination coordinates
        width, height = self.size   # Format (width, height)
        sourceCordinates = box.astype("float32")
        destinationCoordinates = getDestinationCoordinates(width, height)

        # Get perspective transformation matrix
        M = cv2.getPerspectiveTransform(sourceCordinates, destinationCoordinates)

        # Directly warp the rotated rectangle to get the straightened rectangle
        parkingSpaceImage = cv2.warpPerspective(image, M, (width, height))

        # Rotate image, if height < width [I want to have cars in same direction]
        if(height < width):
            parkingSpaceImage = cv2.rotate(parkingSpaceImage, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # Set image of parking space
        self.image = cv2.resize(parkingSpaceImage, resizeSize)

        # Get image as grayscale (Because HOG compute with grayscale image)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)


    '''
    Pre dané parkovacie miesto na základe údajov, ktoré o mieste má, vypočíta obdĺžnik okolo neho
    '''
    def getRotatedRectangleBox(self):
        # Get rotated rectangle around parking space
        box = cv2.boxPoints((self.center, self.size, self.angle))

        # Get values in array as integers
        return np.int0(box)


    '''
    Vráti údaj o tom, či je miesto obsadené alebo nie (načítaný údaj z XML o parkovisku)
    '''
    def isOccupied(self):
        return self.occupied


    '''
    Vráti údaj o tom, či je miesto obsadené alebo nie (na základe klasifikácie)
    '''
    def isOccupiedByPrediction(self):
        return self.predictOccupied


    '''
    Nastaví hodnotu predictOccupied podľa toho, kam parkovacie miesto zaradil daný klasifikátor
    '''
    def setPredictOccupied(self, prediction):
        self.predictOccupied = prediction


    '''
    Vráti true/false podľa toho, či sa klasifikácia zhoduje s tým, čo si o mieste načítal z XML súboru
    '''
    def isPredictionCorrect(self):
        return self.occupied == self.predictOccupied


    '''
    Pre dané parkovacie miesto vypočíta HOG deskriptor
    '''
    def getHOGDescriptor(self):
        if(self.HOGDescriptor is None):
            HOGDescriptor, _ = hog(self.image, orientations=9, pixels_per_cell=(8, 8),
                        cells_per_block=(2, 2), visualize=True)
            self.HOGDescriptor = HOGDescriptor

        return self.HOGDescriptor


    '''
    Vráti vizualizáciu HOG deskriptora
    '''
    def getHOGImage(self):
        _, HOGImage = hog(self.image, orientations=9, pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2), visualize=True)

        return HOGImage


    '''
    Zobrazí obrázok pre parkovacie miesto a aj jeho HOG vizualizáciu
    '''
    def showHOGImage(self):
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

        ax1.axis('off')
        ax1.imshow(self.image, cmap=plt.cm.gray)
        ax1.set_title('Parkovacie miesto')

        ax2.axis('off')
        ax2.imshow(self.getHOGImage(), cmap=plt.cm.gray)
        ax2.set_title('Histogram orientovaných gradientov')

        plt.show()

    
    '''
    Uloží obrázok pre parkovacie miesto a aj jeho HOG vizualizáciu
    '''
    def saveHOGImage(self, dir="", format="png"):
        plt.close('all')
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

        ax1.axis('off')
        ax1.imshow(self.image, cmap=plt.cm.gray)
        ax1.set_title('Parkovacie miesto')

        ax2.axis('off')
        ax2.imshow(self.getHOGImage(), cmap=plt.cm.gray)
        ax2.set_title('Histogram orientovaných gradientov')

        fileName = f"{dir}/{self.getParkingSpaceName()}.{format}"
        plt.savefig(fileName)


    '''
    Vráti meno parkovacieho miesta (názov obrázku _ id parkovacieho miesta)
    '''
    def getParkingSpaceName(self):
        return f"{self.imageName}_{self.id}"