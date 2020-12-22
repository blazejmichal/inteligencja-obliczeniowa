import math


class Lab6Task1:

    @classmethod
    def run(cls):
        print("\nLab6Task1")
        testArguments = (
            (23, 75, 176), (25, 67, 180), (28, 120, 175), (22, 65, 165), (46, 70, 187), (50, 68, 180), (48, 97, 178))
        for row in testArguments:
            cls.forwardPass(row[0], row[1], row[2])

    @classmethod
    def forwardPass(cls, age, weight, height):
        hidden1 = age * (-0.46122) + weight * (0.97314) + height * (-0.39203)
        hidden2 = age * (0.78548) + weight * (2.10584) + height * (-0.57847)

        hidden1 += 0.80109
        hidden2 += 0.43529

        lastNeuron = hidden1 * (-0.81546) + hidden2 * (0.43529)
        lastNeuron += -0.2369

        result = cls.activationFunction(lastNeuron)
        print(result)
        return result

    @classmethod
    def activationFunction(cls, value):
        value = value * (-1)
        result = 1 / (1 + math.exp(value))
        return result
