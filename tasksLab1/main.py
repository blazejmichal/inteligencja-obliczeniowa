from task1.TaskA import TaskA
from task1.TaskB import TaskB
from task1.TaskC import TaskC
from task1.TaskD import TaskD
from task1.TaskE import TaskE
from task1.TaskF import TaskF
from task1.TaskG import TaskG
from task2.TaskA import Task2a
from task2.TaskB import Task2b
from task2.TaskC import Task2c
from task2.TaskD import Task2d


def run():
    TaskA.execute()
    TaskB.execute()
    TaskC.execute()
    taskD = TaskD()
    taskD.execute()
    taskE = TaskE()
    v = taskE.execute()
    taskF = TaskF()
    taskF.execute(v)
    taskG = TaskG()
    taskG.execute(v)
    Task2a.execute()
    Task2b.execute()
    Task2c.execute()
    Task2d.execute()


run()
