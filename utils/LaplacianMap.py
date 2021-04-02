import torch
import torch.nn as nn
import numpy as np
from utils import readAdjFile, getLaplacianMatrix


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


def calcLapNorms(vertName, graphFile, saveFile):
    vertArray = readVertArrayFile(vertName)
    vertArray = np.array(vertArray)

    adj = readAdjFile(graphFile)
    LapM = getLaplacianMatrix(adj)
    LapM = torch.from_numpy(LapM).type(torch.FloatTensor)
    vertArray = torch.from_numpy(vertArray).type(torch.FloatTensor)
    print(LapM.size())
    print(vertArray.size())

    LVert = torch.matmul(LapM, vertArray)
    print(LVert.size())

    Len = torch.bmm(LVert.unsqueeze(1), LVert.unsqueeze(-1))
    Len = torch.sqrt(Len)
    Len = Len.squeeze(-1)
    print(Len.size())

    LVert = torch.div(LVert, Len)
    print(LVert.size())

    LVert = LVert.numpy()
    saveArrayFile(LVert, saveFile)


def readFaceArrayFile(name):
    array = []
    file = open(name, "r")
    for line in file:
        values = line.split()
        if len(values) > 1:
            v = [int(x) for x in values]
            array.append(v)
    return array


def savePly(pDir, verts, norms, colors, faces):
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
            n = norms[p]
            f.write(str(v[0]) + " " + str(v[1]) + " " + str(v[2]) + " "
                    + str(n[0]) + " " + str(n[1]) + " " + str(n[2]) + " "
                    + str(int(c[0])) + " " + str(int(c[1])) + " " + str(int(c[2])) + " " + "255\n")
        for p in range(numFace):
            fds = faces[p]
            f.write("3 " + str(fds[0]-1) + " " + str(fds[1]-1)
                    + " " + str(fds[2]-1) + "\n")
        f.close()


if __name__ == '__main__':
    faces = readFaceArrayFile("./data/Face.txt")
    verts = readVertArrayFile("./data/0000195.txt")
    norms = readVertArrayFile("./data/0000195_l.txt")
    refes = readVertArrayFile("./data/0000195_n.txt")
    color = []
    for i in range(len(norms)):
        r = refes[i]
        n = norms[i]
        a = r[0]*n[0] + r[1]*n[1] + r[2]*n[2]
        if a < 0.0:
            n = [-n[0], -n[1], -n[2]]
            norms[i] = n
        c = [int((n[0] + 1.) * 0.5 * 255), int((n[1] + 1.) * 0.5 * 255), int((n[2] + 1.) * 0.5 * 255)]
        color.append(c)

    savePly("./data/test_ll.ply", np.array(verts), np.array(norms), np.array(color), np.array(faces))
