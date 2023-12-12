
class DistributedMemory:
    def __init__(self):
        self.rank = 0
        self.size = 1


    def __str__(self):
        return f"Rank:\t{self.rank}\nSize:\t{self.size}"
