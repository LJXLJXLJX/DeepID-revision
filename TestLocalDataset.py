import os
import compare
import pickle
import matplotlib.pyplot as plt
import random

path = 'F:/Project/LBPface/224pics_nolens/'
fileList = os.listdir(path)
picNum = len(fileList)
sameList = []
diffList = []


def getID(people):
    posA = people.find('A')
    posB = people.find('B')
    posC = people.find('C')
    posDot = people.find('.')
    if posA != -1:
        id = people[0:posA]
    elif posB != -1:
        id = people[posB + 1:posDot]
    else:
        id = people[posC + 1:posDot]
    return id


def samePeople(people1, people2):
    id1 = getID(people1)
    id2 = getID(people2)
    return id1 == id2


sameList_full_flag=False
diffList_full_flag=False
while(not (sameList_full_flag and sameList_full_flag) ):
    m=random.randint(0,picNum-1)
    n=random.randint(0,picNum-1)

    if samePeople(fileList[m],fileList[n]):
        if len(sameList)<250:
            sim = compare.compare(path + fileList[m], path + fileList[n])
            sameList.append(sim)
            sameList=list(set(sameList)) #set去重复
            print('same: ',len(sameList))
        else:
            sameList_full_flag=True
    else:
        if len(diffList)<250:
            sim = compare.compare(path + fileList[m], path + fileList[n])
            diffList.append(sim)
            diffList=list(set(diffList)) #set去重复
            print('diff: ',len(diffList))
        else:
            diffList_full_flag=True


plt.hist(sameList, 50, normed=1, facecolor='g', alpha=0.75, histtype='step')
plt.hist(diffList, 50, normed=1, facecolor='r', alpha=0.75, histtype='step')
plt.show()

with open('data/localTest.pkl', 'wb') as f:
    pickle.dump(sameList, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(diffList, f, pickle.HIGHEST_PROTOCOL)
