class IrisColumn:
    variance = 0
    name = ''

    def __init__(self, variance, name):
        self.variance = variance
        self.name = name

    def __lt__(self, other):
        result = self.variance > other.variance
        return result

    def __str__(self):
        return 'Nazwa kolumny: ' + self.firstName
