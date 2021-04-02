import numpy as np
import os


def readTextureOBJFile_WithMTL(fname, MTLF, faceID=1):
    if not(os.path.exists(fname)):
        return None, None, None, None
    vertArray = []
    normArray = []
    faceArray = []
    textArray = []
    file = open(fname, "r")
    readFF = ""
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
        if values[0] == 'vt':
            vt = [float(x) for x in values[1:3]]
            textArray.append(vt)
        if values[0] == 'usemtl':
            readFF = values[1]
        if values[0] == 'f' and readFF == MTLF:
            f = [int(x.split('/')[faceID]) for x in values[1:4]]
            faceArray.append(f)
    vertArray = np.array(vertArray, dtype=np.float64)
    normArray = np.array(normArray)
    faceArray = np.array(faceArray)
    textArray = np.array(textArray)

    return vertArray, normArray, faceArray, textArray


def readTextureOBJFile(fname, faceID=1):
    if not(os.path.exists(fname)):
        return None, None, None, None
    vertArray = []
    normArray = []
    faceArray = []
    textArray = []
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
        if values[0] == 'vt':
            vt = [float(x) for x in values[1:3]]
            textArray.append(vt)
        if values[0] == 'f':
            f = [int(x.split('/')[faceID]) for x in values[1:4]]
            faceArray.append(f)
    vertArray = np.array(vertArray, dtype=np.float64)
    normArray = np.array(normArray)
    faceArray = np.array(faceArray)
    textArray = np.array(textArray)

    return vertArray, normArray, faceArray, textArray


def savePly(pDir, verts, colors, faces):
    numVerts = verts.shape[0]
    numFace = faces.shape[0]
    with open(pDir, 'w') as f:
        f.write("ply\n" + "format ascii 1.0\n")
        f.write("element vertex " + str(numVerts) + "\n")
        f.write("property float x\n" + "property float y\n" + "property float z\n")
        f.write("property uchar red\n" + "property uchar green\n"
                + "property uchar blue\n" + "property uchar alpha\n")
        f.write("element face " + str(numFace) + "\n")
        f.write("property list uchar int vertex_indices\n" + "end_header\n")
        for p in range(numVerts):
            v = verts[p]
            c = colors[p]
            f.write(str(v[0]) + " " + str(v[1]) + " " + str(v[2]) + " "
                    + str(int(c[0])) + " " + str(int(c[1])) + " " + str(int(c[2])) + " " + "255\n")
        for p in range(numFace):
            fds = faces[p]
            f.write("3 " + str(fds[0]-1) + " " + str(fds[1]-1)
                    + " " + str(fds[2]-1) + "\n")
        f.close()


def readFaceArrayFile(name):
    array = []
    file = open(name, "r")
    for line in file:
        values = line.split()
        if len(values) > 1:
            v = [int(x) for x in values]
            array.append(v)
    return array


def readVertArrayFile(name):
    array = []
    file = open(name, "r")
    for line in file:
        values = line.split()
        if len(values) > 1:
            v = [float(x) for x in values]
            array.append([v[0], v[1], v[2]])
    return array


def textureInfoGrab():
    # caseName = 'case_1/Chamuse_Updress/'
    # OrigName = 'D:/models/MD/DataModel/DressOri/' + caseName + 'uv/uv.obj'
    caseName = 'template/'
    prefRoot = 'D:/models/NR/Data/anotherTemplate/case_1/'
    OrigName = prefRoot + caseName + 'uv/ori.obj'
    _, _, _faces, _texts = readTextureOBJFile(OrigName, 1)
    #_, _, _faces, _texts = readTextureOBJFile_WithMTL(OrigName, 'f201_body', 1)
    print(_texts.shape)
    z_t = np.zeros((_texts.shape[0], 1))
    z_texts = np.concatenate([_texts, z_t], axis=-1)
    print(z_texts.shape)
    colors = 192 * np.ones_like(z_texts)
    savePly(prefRoot + caseName + 'uv/uv.ply', z_texts, colors, _faces)  # face v/vt/vn
    _verts, _, _faces1, _ = readTextureOBJFile(OrigName, 0)
    #_verts, _, _faces1, _ = readTextureOBJFile_WithMTL(OrigName, 'f201_body', 0)
    savePly(prefRoot + caseName + 'uv/geo.ply', _verts, colors, _faces1)  # face v/vt/vn


def GrabGeoFromObj():
    caseName = 'fblack/'
    prefRoot = 'D:/models/MD/FuseCharac/'
    frame0 = 1
    frame1 = 191
    for f in range(frame0, frame1+1):
        objName = prefRoot + caseName + '/Meshes/M_' + str(f).zfill(6) + '.obj'
        #_verts, _, _faces1, _ = readTextureOBJFile(objName, 0)
        _verts, _, _faces1, _ = readTextureOBJFile_WithMTL(objName, 'f201_body', 0)
        colors = 192 * np.ones_like(_verts)
        savePly(prefRoot + caseName + '/Meshes/' + str(f).zfill(6) + '.ply', _verts, colors, _faces1)


if __name__ == '__main__':
    textureInfoGrab()
    #GrabGeoFromObj()
    exit(1)

    FInds = readFaceArrayFile('../baseDress/AAA/Chama_10_n/Face.txt')
    verts = readVertArrayFile('../baseDress/AAA/Chama_10_n/0000195.txt')
    norms = readVertArrayFile('../baseDress/AAA/Chama_10_n/0000195_n.txt')
    color = []
    for n in norms:
        c = [int((n[0] + 1.)*0.5*255), int((n[1] + 1.)*0.5*255), int((n[2] + 1.)*0.5*255)]
        color.append(c)
    savePly('colorTest.ply', np.array(verts), np.array(color), np.array(FInds))
