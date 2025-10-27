import numpy as np

class TileCoding:
    def __init__(self, num_tiles: int, low: float, high: float, num_tilings: int, offset: float) -> None:
        self.nTiles = num_tiles
        self.low = low
        self.high = high
        self.nTilings = num_tilings
        self.offset = offset
        self.I = (self.high - self.low) / self.nTiles
        self.tilings = []

    def constructTilings(self):
        for tiling_idx in range(self.nTilings):
            l = self.low + tiling_idx * self.offset
            for _ in range(self.nTiles):
                self.tilings.append(np.array([l, l + self.I]))
                l += self.I

    def getFeature(self, x: float):
        x = np.clip(x, self.low, self.high - 1e-9)
        features = np.zeros(self.nTiles * self.nTilings)
        for i, bound in enumerate(self.tilings):
            if bound[0] <= x < bound[1]:
                features[i] = 1
        return features

    def getFeatureIndices(self, x: float):
        x = np.clip(x, self.low, self.high - 1e-9)
        indices = []
        for i, bound in enumerate(self.tilings):
            if bound[0] <= x < bound[1]:
                indices.append(i)
        return indices

tc = TileCoding(5,0,10,2,1)
print(tc.I)
tc.constructTilings()
print(tc.tilings)
print(tc.getFeature(9.5))
print(tc.getFeature(0.1872))
print(tc.getFeature(5.66))
print(tc.getFeatureIndices(6.99))
