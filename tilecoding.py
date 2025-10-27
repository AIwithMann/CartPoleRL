import numpy as np 
 
class TileCoding:
    def __init__(self,num_tiles:int,low:float,high:float)->None:
        self.nTiles = num_tiles
        self.low = low  
        self.high = high 
        self.I = (abs(self.low) + abs(self.high)) / self.nTiles
        self.tilings = []

    def constructTilings(self):
        l = self.low
        while len(self.tilings)!=self.nTiles:
            self.tilings.append(np.array([l,l+self.I]))
            l += self.I
            
    def getFeature(self,x:float):
        if x < self.low: x = self.low
        elif x > self.high: x = self.high
    
        features = np.zeros(self.nTiles)
        for i,bound in enumerate(self.tilings):
            if bound[0] <= x < bound[1]:
                features[i] = 1
        return features

tc = TileCoding(5,0,10)
print(tc.I)
tc.constructTilings()
print(tc.tilings)
print(tc.getFeature(9.5))
print(tc.getFeature(0.1872))
print(tc.getFeature(5.66))