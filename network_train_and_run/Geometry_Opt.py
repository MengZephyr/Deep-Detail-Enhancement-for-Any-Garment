from DataIO import *
import torch
import numpy as np
from torch import optim
import os
import torch.nn as nn
from Losses import Loss_NormalCrossVert, LapSmooth_loss
import time

#USE_CUDA = False
USE_CUDA = torch.cuda.is_available()
print('balin-->', USE_CUDA)
device = torch.device("cuda:0" if USE_CUDA else "cpu")

num_iter = 500

cEdgeWeight = 1.e2
distWeight = 8.e2
smoothWeight = 5.e5


def geo_opt(gtVertName, rrVertName, rrNormName, ccFlagName, vertEdges_0, vertEdges_1, EdgeCounts, numV, LapM, faceArray):
    gtVertArray = np.array(readVertArrayFile(gtVertName))
    rrVertArray = np.array(readVertArrayFile(rrVertName))
    rrNormArray = np.array(readVertArrayFile(rrNormName))
    rrColorArray = norm_color(rrNormArray)
    ccInds = readVertCCFlag(ccFlagName)
    print(len(ccInds))

    gtVertTensor = torch.from_numpy(gtVertArray).type(torch.FloatTensor).to(device)
    rrNormTensor = torch.from_numpy(rrNormArray).type(torch.FloatTensor).to(device)
    #print(gtVertTensor.size())

    #lapNorm_Corr_activations = LaplacianCorrNorm(LapM=LapM, vert=gtVertTensor, normal=gtNormTensor)

    # ini vert position
    noise = torch.from_numpy(rrVertArray.copy()).type(torch.FloatTensor).to(device)
    noise.requires_grad = True

    Func_lossNormalCrossVert = Loss_NormalCrossVert(vertEdges_0, vertEdges_1, EdgeCounts, numV, device).to(device)
    Func_lossVertToGtVert = nn.L1Loss(reduction='mean').to(device)

    adam = optim.Adam(params=[noise], lr=0.001, betas=(0.9, 0.999), amsgrad=True)
    oldLoss = 0.
    t = time.time()
    for iteration in range(num_iter + 1):
        adam.zero_grad()
        loss_geo = Func_lossNormalCrossVert(normalArray=rrNormTensor, vertArray=noise)
        loss_dist = Func_lossVertToGtVert(gtVertTensor[ccInds, :], noise[ccInds, :])
        loss_smooth = LapSmooth_loss(LapM, noise)
        # betaCorr = 0.8
        # noise_lapNorm_Corr_activations = LaplacianCorrNorm(LapM=LapM, vert=noise, normal=rrNormTensor)
        # loss_smooth = betaCorr * LapNorm_CorrLoss(lapNorm_Corr_activations, noise_lapNorm_Corr_activations, numV) \
        #               + (1.-betaCorr) * LapSmooth_loss(LapM, noise)

        total_loss = cEdgeWeight * loss_geo + smoothWeight * loss_smooth + distWeight * loss_dist
        #total_loss = cEdgeWeight * loss_geo + smoothWeight * loss_smooth
        if iteration % 10 == 0:
            print("Iteration: {}, total Loss: {:.3f}, Geo Loss: {:.3f}, disLoss: {:.3f}, Smooth Loss: {:.3f}".
                  format(iteration, total_loss.item(), cEdgeWeight * loss_geo.item(), distWeight * loss_dist,
                         smoothWeight * loss_smooth.item()))

        # if not os.path.exists('../test/generated_geo/'):
        #     os.mkdir('../test/generated_geo/')

        #if iteration % 10 == 0:
        #    rst = noise.clone().detach()
        #    writePlyV_F_N_C(pDir='../test/generated_geo/iter_{}.ply'.format(iteration),
        #                    verts=rst.cpu().numpy(), normals=rrNormArray, colors=rrColorArray, faces=faceArray)

        if total_loss.item()-oldLoss < 0.01 and iteration > 50:
        #if total_loss.item() < 2.5:
            return noise.clone().detach(), rrNormArray, rrColorArray, time.time()-t

        oldLoss = total_loss.item()

        # backprop
        total_loss.backward()
        # update parameters
        adam.step()

    return noise.clone().detach(), rrNormArray, rrColorArray, time.time()-t


if __name__ == '__main__':
    caseName = 'Chamuse_tango/skirt/'
    prefRoot = '../Data/case_3/'

    adjName = prefRoot + caseName + '/uv/10_adjGraph.txt'
    adj, numV = readAdjFile(adjName)
    LapM = getLaplacianMatrix(adj)
    LapM = torch.from_numpy(LapM).type(torch.FloatTensor).to(device)

    EdgeName = prefRoot + caseName + '/uv/10_crossEdge.txt'
    vertEdges_0, vertEdges_1, EdgeCounts = readCrossEdges(EdgeName, numV)
    vertEdges_0 = np.array(vertEdges_0)
    vertEdges_1 = np.array(vertEdges_1)
    EdgeCounts = np.array(EdgeCounts)

    FaceName = prefRoot + caseName + '/uv/Face_10.txt'
    faceArray = np.array(readFaceIndex(FaceName))

    vertRoot = prefRoot + caseName + '/10_DsUs_C/'
    normRoot = "../PatchTest/" + caseName + '/normal_ps/'
    saveRoot = "../PatchTest/" + caseName + '/geo_ps/'
    frame0 = 160
    frame1 = 160
    timeCount = 0.
    if not os.path.exists(saveRoot):
        os.makedirs(saveRoot)
    for FrameID in range(frame0, frame1+1):
        gtVertName = vertRoot + str(FrameID).zfill(7) + '.txt'
        rrVertName = vertRoot + str(FrameID).zfill(7) + '.txt'
        ccFlagName = vertRoot + str(FrameID).zfill(7) + '_f.txt'
        rrNormName = normRoot + str(FrameID).zfill(7) + '_n.txt'
        rst, rrNormArray, rrColorArray, tt = geo_opt(gtVertName, rrVertName, rrNormName, ccFlagName,
                                                     vertEdges_0, vertEdges_1, EdgeCounts, numV, LapM, faceArray)
        rst = rst.cpu().numpy()
        saveName = saveRoot + str(FrameID).zfill(7)
        saveOutVerts(rst, saveName + '.obj')
        print("Frame: {}, time:{:.4f}s".format(FrameID, tt))
        writePlyV_F_N_C(pDir=saveName + '.ply', verts=rst, normals=rrNormArray, colors=rrColorArray, faces=faceArray)
        timeCount = timeCount + tt

    print('Final Time -->', timeCount)
