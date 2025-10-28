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
        self.I = np.divide(np.subtract(self.high,self.low),self.nTiles)
        self.tilings = []
        self.dim_ranges = []

    def constructTilings(self):
        for dim in range(self.low.size):
            start_idx = len(self.tilings)
            for tiling_idx in range(self.nTilings):
                l = self.low[dim] + tiling_idx * self.offset[dim]
                for _ in range(self.nTiles):
                    self.tilings.append(np.array([l, l + self.I[dim]]))
                    l += self.I[dim]
            self.dim_ranges.append(np.array([start_idx,len(self.tilings)]))

    def getFeatureIndices(self, x: float):
        x = np.clip(x, self.low, self.high - 1e-9)
        indices = []
        for dim_idx, value in enumerate(x):
            start,end = self.dim_ranges[dim_idx]
            for i in range(start,end):
                if self.tilings[i][0] <= value < self.tilings[i][1]:
                    indices.append(i)
        return indices

#driver code
tc = TileCoding(5,5,np.array([-10,0]),np.array([0,10]),np.array([2,2]))
tc.constructTilings()
print(tc.tilings)
print(tc.getFeatureIndices(np.array([0,5])))
