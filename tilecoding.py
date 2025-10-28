import numpy as np

class TileCoding:
    def __init__(self, num_tiles: int,num_tilings: int, low: np.ndarray, high: np.ndarray, offset: np.ndarray) -> None:
        if low.size != high.size:
            raise IndexError("size mismatch between self.low and self.high")
        self.nTiles = num_tiles
        self.low = low
        self.high = high
        self.nTilings = num_tilings
        self.offset = offset
        self.I = (self.high - self.low) / self.nTiles
        self.tilings = []

    def constructTilings(self):
        for dim in range(self.low.size):
            dimTilings = []
            for tiling_idx in range(self.nTilings):
                tiling = []
                l = self.low[dim] + tiling_idx * self.offset[dim]
                for _ in range(self.nTiles):
                    tiling.append(np.array([l, l + self.I[dim]]))
                    l += self.I[dim]
                dimTilings.append(tiling)
            self.tilings.append(dimTilings)

    def getFeature(self, x: np.ndarray):
        x = np.clip(x, self.low, self.high - 1e-9)
        features = np.zeros((self.low.size, self.nTilings,self.nTiles))
        for i,dimension in enumerate(self.tilings):
            for j,tiling in enumerate(dimension):
                for k, bound in enumerate(tiling):
                    if bound[0] <= x[i] < bound[1]:
                        features[i][j][k] = 1
        return features

    '''def getFeatureIndices(self, x: float):
        x = np.clip(x, self.low, self.high - 1e-9)
        indices = []
        for i, bound in enumerate(self.tilings):
            if bound[0] <= x < bound[1]:
                indices.append(i)
        return indices'''

tc = TileCoding(10,5,np.array([-10,0]),np.array([0,10]),np.array([2,2]))
print(tc.I)
tc.constructTilings()
print(tc.tilings)
print(tc.getFeature(np.array([0,5])))
