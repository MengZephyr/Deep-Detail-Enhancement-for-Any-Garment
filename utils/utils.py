import os
import numpy as np


def readVertArrayFile(name, aa=[0., 0., 0.]):
    array = []
    file = open(name, "r")
    for line in file:
        values = line.split()
        if len(values) > 1:
            v = [float(x) for x in values]
            array.append([v[0] + aa[0], -v[2] + aa[1], v[1] + aa[2]])
    file.close()
    return array


def readVertOBjFile(name):
    array = []
    file = open(name, "r")
    for line in file:
        values = line.split()
        if values[0] == 'v':
            v = [float(x) for x in values[1:4]]
            array.append([v[0], -v[2], v[1]])
    return array


def readFaceArrayFile(name):
    array = []
    file = open(name, "r")
    for line in file:
        values = line.split()
        if len(values) > 1:
            v = [int(x)-1 for x in values]
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
    return Adj


def genKRingAdj(adj_1, K=3):
    aa = adj_1.copy()
    numV = aa.shape[0]
    Rst = [aa]
    for k in range(K-1):
        kk = Rst[-1].copy()
        for v in range(numV):
            vrow = kk[v, :]
            indd = np.where(vrow > 0)
            b = adj_1[indd[0], :]
            for i in range(len(b)):
                vrow = vrow + b[i, :]
            kk[v, :] = vrow
        kadj = np.zeros_like(adj_1)
        kadj[np.where(kk > 0)] = 1
        Rst.append(kadj)
    return Rst


def getLaplacianMatrix(Adj):
    LapM = np.zeros_like(Adj)
    LapM = LapM - Adj
    LapM = LapM.astype(float)
    a = np.sum(Adj, axis=1).astype(float)
    for i in range(LapM.shape[0]):
        LapM[i, i] = a[i]-1.
        LapM[i, :] = LapM[i, :] / (a[i] - 1)
    return LapM


class _ModelMesh(object):
    def __init__(self, vertName, faceName, adjName, K=3, ss='obj'):
        if ss == 'obj':
            self.verts = readVertOBjFile(vertName)
        elif ss == 'txt':
            self.verts = readVertArrayFile(vertName)
        self.faces = readFaceArrayFile(faceName)
        self.colors = [[192, 192, 192] for _ in range(len(self.verts))]
        #self.colors = [[192, 0, 0] for _ in range(len(self.verts))]
        self.numVerts = len(self.verts)
        self.numFace = len(self.faces)
        print("read adj file...")
        adj = readAdjFile(adjName)
        print(adj.shape)
        self.KAdj = genKRingAdj(adj, K)

    def savePly(self, pDir):
        with open(pDir, 'w') as f:
            f.write("ply\n" + "format ascii 1.0\n")
            f.write("element vertex " + str(self.numVerts) + "\n")
            f.write("property float x\n" + "property float y\n" + "property float z\n")
            f.write("property uchar red\n" + "property uchar green\n"
                    + "property uchar blue\n" + "property uchar alpha\n")
            f.write("element face " + str(self.numFace) + "\n")
            f.write("property list uchar int vertex_indices\n" + "end_header\n")
            for p in range(self.numVerts):
                v = self.verts[p]
                c = self.colors[p]
                f.write(str(v[0]) + " " + str(v[2]) + " " + str(-(v[1])) + " "
                        + str(c[0]) + " " + str(c[1]) + " " + str(c[2]) + " " + "255\n")
            for p in range(self.numFace):
                fds = self.faces[p]
                f.write("3 " + str(fds[0]) + " " + str(fds[1])
                        + " " + str(fds[2]) + "\n")
            f.close()


if __name__ == '__main__':
    a = np.array([[1, 1, 0, 0], [1, 1, 1, 0], [0, 1, 0, 1], [0, 0, 1, 1]])
    print(a)
    b = genKRingAdj(a, 2)
    print(b[0])
    print(b[1])
    lap = getLaplacianMatrix(b[0])
    print(lap)
    # oneMesh = _ModelMesh(vertName="0000170.obj", faceName="../baseDress/Face.txt", ss='obj')
    # oneMesh.savePly("test_rs.ply")
