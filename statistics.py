import math

class Histogram(object):
    def __init__(self):
        pass
    
    def make(self, entries, bins=None):
        sortedEntries = sorted(entries)
        if not bins:
            bins = math.sqrt(len(sortedEntries))
        binWidth = (sortedEntries[-1] + 1e-12 - sortedEntries[0]) / bins
        population = [0] * bins
        for entry in sortedEntries:
            binIndex = int((entry - sortedEntries[0]) / binWidth)
            population[binIndex] += 1
        return population
