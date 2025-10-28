import numpy as np

class TileCoding:
    def __init__(self, num_tiles: int,num_tilings: int, low: np.ndarray, high: np.ndarray,N:int) -> None:
        if low.size != high.size:
            raise IndexError("size mismatch between self.low and self.high")
        self.nTiles = num_tiles
        self.low = low
        self.high = high
        self.tileWidth = (self.high - self.low)/self.nTiles
        self.nTilings = num_tilings
        self.N = N

    def tileIndices(self, x):
        indices = []
        for t in range(self.nTilings):
            offsets = (t / self.nTilings) * self.tileWidth
            tile = np.floor((x + offsets - self.low) / self.tileWidth).astype(int)
            hashVal = hash((t, *tile)) % self.N
            indices.append(hashVal)
        return indices
