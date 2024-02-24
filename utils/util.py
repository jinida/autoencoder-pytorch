import os
import datetime

def createIncrementalPath(rootPath: str) -> str:
    datasetNum = 0
    while True:
        folderName = f"{datasetNum:03d}"
        datasetPath = os.path.join(rootPath, folderName)

        if not os.path.exists(datasetPath):
            os.makedirs(datasetPath)
            logFile = os.path.join(datasetPath, 'log.txt')
            with open(logFile, 'w') as log:
                log.write(f"Folder: {folderName}, Created Date: {datetime.datetime.now().strftime('%Y-%m-%d')}\n")
            break
        datasetNum += 1
    return datasetPath