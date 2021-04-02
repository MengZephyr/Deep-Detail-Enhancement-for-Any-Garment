import numpy as np
import os
import math
from BaseMap import readOBJFile


def readB2OMap(fName):
    if not (os.path.exists(fName)):
        return None
    mapArray = []
    file = open(fName, "r")
    for line in file:
        values = line.split()
        if values[0] == '#':
            numBV = int(values[1])
        else:
            mm = [int(x) for x in values[1:int(values[0])+1]]
            mapArray.append(mm)
    if not (len(mapArray) == numBV):
        print("Error:: Data wrong in mapping file.")
        return None
    return mapArray


def genBaseVertData(ori_Verts, vertMapping, ifNormal=False):
    base_Vert = []
    for i in range(len(vertMapping)):
        mID = vertMapping[i]
        pA = ori_Verts[mID, :]
        cc = [0, 0, 0]
        for j in range(len(pA)):
            cc = [sum(x) for x in zip(cc, pA[j])]
        cc = [x / float(len(mID)) for x in cc]
        if ifNormal:
            d = math.sqrt(cc[0]*cc[0] + cc[1]*cc[1]+cc[2]*cc[2])
            cc = [cc[0]/d, cc[1]/d, cc[2]/d]
        base_Vert.append(cc)
    return base_Vert


def saveArrayFile(array, name):
    with open(name, 'w') as f:
        f.write(str(array.shape[0]) + "\n")
        for c in range(array.shape[0]):
            f.write(str(array[c][0]) + " " + str(array[c][1]) + " " + str(array[c][2]) + "\n")
        f.close()


def readVertArrayFile(name):
    array = []
    file = open(name, "r")
    for line in file:
        values = line.split()
        if len(values) > 1:
            v = [float(x) for x in values]
            array.append([v[0], v[1], v[2]])
    file.close()
    return array


def BaseAAADataList(ddSet, frame0, frame1, b2oMap, bFace, froots, postfix, savefix, ifNorm=False):
    for dname in ddSet:
        setDirPref = froots + dname + "/" + postfix
        saveDirPref = froots + dname + "/" + savefix
        if not os.path.exists(saveDirPref):
            os.makedirs(saveDirPref)
        for id in range(frame0, frame1+1):
            fileName = setDirPref + str(id).zfill(7) + ".obj"
            saveName = saveDirPref + str(id).zfill(7) + ".txt"

            m_Vert, m_Norm, m_Face = readOBJFile(fileName)
            d_Vert = genBaseVertData(m_Vert, b2oMap, False)
            saveArrayFile(np.array(d_Vert), saveName)

            if ifNorm:
                saveNN = saveDirPref + str(id).zfill(7)+"_n.txt"
                d_Norm = genBaseVertData(m_Norm, b2oMap, False)
                saveArrayFile(np.array(d_Norm), saveNN)

        saveArrayFile(bFace, saveDirPref + "Face.txt")


def CombFromTxt(ddSet, frame0, frame1, b2oMap, froots, postfix, savefix):
    for dname in ddSet:
        setDirPref = froots + dname + "/" + postfix
        saveDirPref = froots + dname + "/" + savefix
        if not os.path.exists(saveDirPref):
            os.makedirs(saveDirPref)
        for id in range(frame0, frame1+1):
            fileName = setDirPref + str(id).zfill(7) + ".txt"
            saveName = saveDirPref + str(id).zfill(7) + ".txt"

            m_Vert = readVertArrayFile(fileName)
            m_Vert = np.array(m_Vert)
            d_Vert = genBaseVertData(m_Vert, b2oMap, False)
            saveArrayFile(np.array(d_Vert), saveName)


BASEDATA_SET = ["Chamuse_tango/skirt", "Chamuse", "Chiffon", "CottonVoile", "DenimLight",
                "Knit_Terry", "Linen", "NylonCarvas_l", "WoolMelton_l"]


if __name__ == '__main__':
    froots = 'D:/models/MD/DataModel/DressOri/case_3/'
    BaseName = froots + BASEDATA_SET[0] + '/uv/Base30.obj'
    b_Vert, b_Norm, b_Face = readOBJFile(BaseName)

    BOMap = readB2OMap(froots + BASEDATA_SET[0] + '/uv/ind_B2O_30.txt')
    if BOMap is None:
        exit(1)
    postfix = '/10_Ds_L/'
    savefix = '/10_Ds_C/'
    # BaseAAADataList(ddSet=[BASEDATA_SET[0]], frame0=1, frame1=1, b2oMap=BOMap,
    #                 bFace=b_Face, froots=froots, postfix=postfix, savefix=savefix, ifNorm=False)
    CombFromTxt(ddSet=[BASEDATA_SET[0]], frame0=1, frame1=864, b2oMap=BOMap,
                froots=froots, postfix=postfix, savefix=savefix)

    print("Done.")

    # AdjMap = genAdjGraph(b_Face, b_Vert.shape[0])
    # print(np.where(AdjMap[0, :] == 1))
    # print(AdjMap[0][0])
    #
    # saveAdjMap(AdjMap, "../baseDress/AAA/Chama_10/adjGraph.txt")
