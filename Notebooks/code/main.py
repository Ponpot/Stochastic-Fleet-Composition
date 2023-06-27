
from algorithms import Algorithms
from data_gen import DataGeneration

data = DataGeneration(20,200,50,1)
solution = Algorithms(data)

print(solution.cutting_plane())
print(solution.linearization())