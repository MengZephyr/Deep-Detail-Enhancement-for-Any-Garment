import numpy as np


def readCrossEdges(pDir, numV):
    file = open(pDir, "r")
    line = file.readline()
    values = line.split()
    readV = int(values[0])
    assert (readV == numV)
    vertEdges_0 = []
    vertEdges_1 = []
    EdgeCounts = []
    for v in range(numV):
        line = file.readline()
        values = line.split()
        numE = int(values[0])
        EdgeCounts.append(numE)
        for e in range(numE):
            vertEdges_0.append(int(values[e*2+1]))
            vertEdges_1.append(int(values[e*2+2]))
    file.close()
    return vertEdges_0, vertEdges_1, EdgeCounts


def readVertCCFlag(pDir):
    file = open(pDir, "r")
    line = file.readline()
    values = line.split()
    ccFlag = []
    numV = int(values[0])
    for v in range(numV):
        line = file.readline()
        values = line.split()
        fl = int(values[0])
        ccFlag.append(fl)
    file.close()
    ccFlag = np.array(ccFlag)
    ccInds = np.where(ccFlag > 0)
    return ccInds[0].tolist()

def readFaceIndex(name):
    array = []
    file = open(name, "r")
    for line in file:
        values = line.split()
        if len(values) > 1:
            v = [int(x) - 1 for x in values]
            array.append(v)
    return array


def readAdjFile(name):
    file = open(name, "r")
    line = file.readline()
    values = line.split()
    numV = int(values[1])
    Adj = np.zeros([numV, numV], dtype=int)
    for i in range(numV):
        line = file.readline()
        values = line.split()
        A = [int(x) for x in values[1:int(values[0])+1]]
        Adj[i][i] = 1
        Adj[i][A] = 1
    file.close()
    return Adj, numV


def getLaplacianMatrix(Adj):
    LapM = np.zeros_like(Adj)
    LapM = LapM - Adj
    LapM = LapM.astype(float)
    a = np.sum(Adj, axis=1).astype(float)
    for i in range(LapM.shape[0]):
        LapM[i, i] = a[i]-1.
        LapM[i, :] = LapM[i, :] / (a[i] - 1)
    return LapM


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


def norm_color(normals):
    colors = (normals + np.ones_like(normals)) * 0.5 * 255
    return colors


def writePlyV_F_N_C(pDir, verts, normals, colors, faces):
    numVerts = verts.shape[0]
    numFace = faces.shape[0]
    with open(pDir, 'w') as f:
        f.write("ply\n" + "format ascii 1.0\n")
        f.write("element vertex " + str(numVerts) + "\n")
        f.write("property float x\n" + "property float y\n" + "property float z\n")
        f.write("property float nx\n" + "property float ny\n" + "property float nz\n")
        f.write("property uchar red\n" + "property uchar green\n"
                + "property uchar blue\n" + "property uchar alpha\n")
        f.write("element face " + str(numFace) + "\n")
        f.write("property list uchar int vertex_indices\n" + "end_header\n")
        for p in range(numVerts):
            v = verts[p]
            c = colors[p]
            n = normals[p]
            f.write(str(v[0]) + " " + str(v[1]) + " " + str(v[2]) + " "
                    + str(n[0]) + " " + str(n[1]) + " " + str(n[2]) + " "
                    + str(int(c[0])) + " " + str(int(c[1])) + " " + str(int(c[2])) + " " + "255\n")
        for p in range(numFace):
            fds = faces[p]
            f.write("3 " + str(fds[0]) + " " + str(fds[1])
                    + " " + str(fds[2]) + "\n")
        f.close()


def saveOutVerts(Outs, pDir):
    with open(pDir, 'w') as f:
        for p in range(Outs.shape[0]):
            f.write("v " + str(Outs[p][0]) + " " + str(Outs[p][1]) + " " + str(Outs[p][2]) + "\n")
        f.close()
