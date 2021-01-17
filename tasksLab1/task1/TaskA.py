class TaskA:
    a = 123
    b = 321

    @staticmethod
    def execute():
        result = TaskA.a * TaskA.b
        print('Rezultat mnozenia: ' + str(result))
