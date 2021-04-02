import numpy as np
import os


def saveArrayFile(array, name):
    with open(name, 'w') as f:
        f.write(str(array.shape[0]) + "\n")
        for c in range(array.shape[0]):
            f.write(str(array[c][0]) + " " + str(array[c][1]) + " " + str(array[c][2]) + "\n")
        f.close()


def readOBJFile(fname):
    if not(os.path.exists(fname)):
        return None, None, None
    vertArray = []
    normArray = []
    faceArray = []
    file = open(fname, "r")
    for line in file:
        if line.startswith('#'):
            continue
        values = line.split()
        if not values:
            continue
        if values[0] == 'v':
            v = [float(x) for x in values[1:4]]
            vertArray.append(v)
        if values[0] == 'vn':
            vn = [float(x) for x in values[1:4]]
            normArray.append(vn)
        if values[0] == 'f':
            f = [int(x.split('/')[0]) for x in values[1:4]]
            faceArray.append(f)
    vertArray = np.array(vertArray, dtype=np.float64)
    normArray = np.array(normArray)
    faceArray = np.array(faceArray)

    return vertArray, normArray, faceArray


def genAdjGraph(faceA, numVert):
    numF = faceA.shape[0]
    print('face shape: ', faceA.shape, numVert)

    Adj = np.zeros([numVert, numVert], dtype=int)
    for i in range(numF):
        vIDs = faceA[i]
        vIDs = [x-1 for x in vIDs]
        for j in range(faceA.shape[1]):
            vIDs = np.roll(vIDs, 1)
            #Adj[vIDs[0]][vIDs[0]] = Adj[vIDs[0]][vIDs[0]] + 1
            Adj[vIDs[0]][vIDs[1:4]] = 1
    return Adj


def saveAdjMap(adj, fName):
    #print('kk..')
    with open(fName, 'w') as f:
        f.write("# " + str(adj.shape[0]) + " " + str(adj.shape[1]) + "\n")
        for i in range(adj.shape[0]):
            aa = np.where(adj[i, :] == 1)
            # print(adj[i][i])
            # print(aa)
            # print(adj[i, aa[0]])
            f.write(str(len(aa[0])))
            for j in range(len(aa[0])):
                f.write(" " + str(aa[0][j]))
            f.write("\n")
        f.close()


def crossEdge_Graph(FaceArray, numV):
    vertEdge = [[] for _ in range(numV)]
    numF = FaceArray.shape[0]
    print(len(vertEdge))
    for i in range(numF):
        vIDs = FaceArray[i]
        vIDs = [x - 1 for x in vIDs]
        for j in range(FaceArray.shape[1]):
            vIDs = np.roll(vIDs, 1)
            vertEdge[vIDs[0]].append(vIDs[1:3])
    return vertEdge


def saveCrossEdges(pDir, vertEdges, numV):
    with open(pDir, 'w') as f:
        f.write(str(numV) + "\n")
        for v in range(numV):
            Edges = vertEdges[v]
            f.write(str(len(Edges)))
            for e in Edges:
                f.write(" " + str(e[0]) + " " + str(e[1]))
            f.write("\n")
        f.close()


if __name__ == '__main__':
    caseName = 'Larkspur/hood_denimLight/'
    prefRoot = 'D:/models/MD/DataModel/DressOri/case_7/'

    BaseName = prefRoot + caseName + 'uv/Base10.obj'
    saveRoot = prefRoot + caseName + 'uv/10_'
    b_Vert, b_Norm, b_Face = readOBJFile(BaseName)
    saveArrayFile(b_Face, saveRoot + "Face.txt")

    # generate 1-ring graph
    AdjMap = genAdjGraph(b_Face, b_Vert.shape[0])
    print(np.where(AdjMap[0, :] == 1))
    print(AdjMap[0][0])

    saveAdjMap(AdjMap, saveRoot + 'adjGraph.txt')

    # generate cross edge
    vertEdges = crossEdge_Graph(b_Face, b_Vert.shape[0])
    saveCrossEdges(saveRoot + 'crossEdge.txt', vertEdges, b_Vert.shape[0])
