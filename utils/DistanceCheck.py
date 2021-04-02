from utils import *
from utils import _ModelMesh
import math

rstMesh = _ModelMesh(vertName="0000195.obj", faceName="../baseDress/Face.txt",
                    adjName='../baseDress/adjGraph.txt', K=1, ss='obj')
rstMesh.savePly("test.ply")
exit(1)
GrtMesh = _ModelMesh(vertName="0000170.txt", faceName="../baseDress/Face.txt",
                     adjName='../baseDress/adjGraph.txt', K=5, ss='txt')


def f_L1Dist(v0, v1, dim):
    x = [abs(v0[0][i] - v1[0][i]) for i in range(dim)]
    x = sum(x)
    return x


def f_L2Dist(v0, v1):
    assert (len(v0) == len(v1))
    x = sum((v0[i] - v1[i])*(v0[i] - v1[i]) for i in range(len(v0)))
    return x


def f_ColorMeld(coef, color1, color2):
    c = [int((1.-coef) * color1[i] + coef * color2[i]) for i in range(len(color1))]
    return c


def compVertDist(rsM, gtM):
    assert (rsM.numVerts == gtM.numVerts)
    dist = [f_L1Dist(rsM.verts[i], gtM.verts[i], 3) for i in range(rsM.numVerts)]
    min_d = min(dist)
    max_d = max(dist)
    cdd = [(d-min_d)/(max_d-min_d) for d in dist]

    for p in range(rsM.numVerts):
        rsM.colors[p] = f_ColorMeld(cdd[p], [0., 0., 255.], [255., 0., 0.])


def calcLapFeatures(Mesh):
    lap_1 = getLaplacianMatrix(Mesh.KAdj[0])
    verts = np.array(Mesh.verts)
    lapV = np.matmul(lap_1, verts)
    norV = np.linalg.norm(lapV, axis=-1)
    for v in range(Mesh.numVerts):
        lapV[v, :] = lapV[v, :] / norV[v]
    llapV = np.matmul(lap_1, lapV)
    feat = np.concatenate([lapV, llapV], axis=1)
    return feat


def calGarmm(Feat):
    nF = Feat.shape[0]
    nC = Feat.shape[1]
    TF = np.transpose(Feat, (1, 0))
    GG = np.matmul(TF, Feat) / nF
    return np.reshape(GG, (1, nC*nC))


def compFGarmmaDist(rsM, rsF, gtF):
    neiMatrix = rsM.KAdj[-1]
    dist = []
    for vi in range(rsM.numVerts):
        indN = np.where(neiMatrix[vi] > 0)[0]
        ffr = rsF[indN, :]
        ffg = gtF[indN, :]
        rGarmm = calGarmm(ffr)
        gGarmm = calGarmm(ffg)
        dist.append(f_L1Dist(rGarmm, gGarmm, ffr.shape[1]*ffr.shape[1]))

    min_d = min(dist)
    print(min_d)
    max_d = max(dist)
    print(max_d)
    cdd = [(d - min_d) / (max_d - min_d) for d in dist]
    for p in range(rsM.numVerts):
        rsM.colors[p] = f_ColorMeld(cdd[p], [0., 0., 255.], [255., 0., 0.])


def calcLapVariance(Mesh):
    lap_1 = getLaplacianMatrix(Mesh.KAdj[0])
    verts = np.array(Mesh.verts)
    lapV = np.matmul(lap_1, verts)
    print(lapV.shape)
    VLap = np.expand_dims(lapV, axis=1)
    TLap = np.transpose(VLap, (0, 2, 1))
    DD = np.matmul(VLap, TLap)
    DD = np.sqrt(np.squeeze(DD, axis=-1))
    print(DD.shape)
    for v in range(Mesh.numVerts):
        lapV[v, :] = lapV[v, :] / DD[v]

    vvar = []
    for vi in range(Mesh.numVerts):
        indN = np.where(Mesh.KAdj[-1][vi] > 0)
        #cLap = np.mean(lapV[indN[0], :], axis=0)
        cLap = lapV[vi, :]
        var = [f_L2Dist(cLap, lapV[k, :]) for k in indN[0]]
        vv = math.sqrt(sum(var) / (len(indN[0]) - 1))
        vvar.append(vv)
    print(len(vvar))

    min_d = min(vvar)
    max_d = max(vvar)
    cdd = [(d - min_d) / (max_d - min_d) for d in vvar]
    for p in range(Mesh.numVerts):
        Mesh.colors[p] = f_ColorMeld(cdd[p], [0., 0., 255], [255., 0., 0.])
    return


GtFeat = calcLapFeatures(GrtMesh)
RsFeat = calcLapFeatures(rstMesh)

print(GtFeat.shape)

compFGarmmaDist(rstMesh, RsFeat, GtFeat)
rstMesh.savePly("DistGarm.ply")

#GrtMesh.savePly("VarC.ply")

#compVertDist(rstMesh, GrtMesh)
#rstMesh.savePly("DistC.ply")
