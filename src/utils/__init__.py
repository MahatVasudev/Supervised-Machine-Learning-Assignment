import sys
import os

ROOT_DIR: str = '../../'
def DIR_APPENDER(x, y): return f'{x}{y}'


CURRENT_DIR: str = DIR_APPENDER(ROOT_DIR, 'ML_Model_Building/')
DATASET_DIR: str = DIR_APPENDER(ROOT_DIR, 'datasets/')
CURR_DATASET_DIR: str = DIR_APPENDER(CURRENT_DIR, 'datasets/')
SAVED_MODELS_DIR: str = DIR_APPENDER(CURRENT_DIR, 'saved_models/')
MODELS_DIR: str = DIR_APPENDER(CURRENT_DIR, 'models/')
UTILS_DIR: str = DIR_APPENDER(CURRENT_DIR, 'utils/')


if __name__ == '__main__':
    try:
        print("Checking Whether Files are accessible")
        for i in [ROOT_DIR, CURRENT_DIR, DATASET_DIR]:

            print(" ", i)
            for j in os.listdir(i):
                print(f'\t{j}')

            print()
    except Exception as e:
        print("\n\nProblem With Folder Management...\n", e)
