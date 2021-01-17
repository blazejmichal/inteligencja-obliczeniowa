from TaskA import Task2a


class Task2b:

    @staticmethod
    def execute():
        csvRow = '\n2010,460,555,405'
        with open('miasta.csv', 'a') as miasta:
            miasta.write(str(csvRow))
        Task2a.execute()
