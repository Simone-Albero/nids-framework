from pipeline import Pipeline

pipeline = Pipeline()

# Register tasks with the pipeline
@pipeline.register(priority=2)
def task1():
    print("Task 1")

@pipeline.register(priority=1)
def task2():
    print("Task 2")

@pipeline.register(priority=3)
def task3():
    print("Task 3")

# Execute the pipeline
if __name__ == "__main__":
    pipeline.execute()

