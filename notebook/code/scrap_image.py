# scrapper les images train
import queue
import threading
import time
import pandas as pd
import numpy as np 
import requests 
import shutil 
import threading
import sys
import os


class myThread (threading.Thread):
    def __init__(self, threadID, name, q):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.q = q

    def run(self):
        print("Starting " + self.name)
        process_data(self.name, self.q)
        print("Exiting " + self.name)


# fonction pour télécharger une image
def process_data(threadName, q):
    while not exitFlag:
        queueLock.acquire()
        if not workQueue.empty():
            loic, nom = q.get()
        queueLock.release()
        r = requests.get(loic, stream=True)
        if r.status_code != 200:
            with open("image_non_telecharger.txt", "w") as fichier:
                fichier.write(f"image not télécharger code : {nom}\n")
        filename = "../../data/image/" + str(nom) + "." + loic.split('.')[-1]
        with open(filename, 'wb') as f:
            shutil.copyfileobj(r.raw, f)


df = pd.read_csv("../../data/to_scrap/final.csv")
list_url = df['image_url'].to_numpy()
list_code = df['code'].to_numpy()
exitFlag = 0


threadList = ["Thread-1", "Thread-2", "Thread-3", "Thread-4", "Thread-5",
              "Thread-6", "Thread-7", "Thread-8"]
queueLock = threading.Lock()
queueLock.acquire()
workQueue = queue.Queue(len(list_code))
threads = []
threadID = 1

print("Create new threads")
# Create new threads
for tName in threadList:
    thread = myThread(threadID, tName, workQueue)
    thread.start()
    threads.append(thread)
    threadID += 1

print("Fill the queue")
# Fill the queue
#queueLock.acquire()
for coup in zip(list_url, list_code):
    workQueue.put(coup)
queueLock.release()

print("Wait the queue")
# Wait for queue to empty
while not workQueue.empty():
    pass

# Notify threads it's time to exit
exitFlag = 1

# Wait for all threads to complete
for t in threads:
    t.join()
print("Exiting Main Thread")
