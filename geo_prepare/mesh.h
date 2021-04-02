#pragma once

#include <opencv2/opencv.hpp>
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>

using namespace std;

struct R_Mesh
{
	std::vector<cv::Vec3f> verts;
	std::vector<cv::Vec3i> faceInds;

	cv::Vec3f bbmin;
	cv::Vec3f bbmax;

	int numV() { return verts.size(); }
	int numF() { return faceInds.size(); }
	void OrthProjMesh(int H, int W)
	{
		for (int v = 0; v < numV(); v++)
		{
			cv::Vec3f pos = verts[v];
			pos = cv::Vec3f(pos[0] * W, pos[1] * H, 0.);
			verts[v] = pos;
		}
	}
	void cameraProjMesh(float fx, float fy, float cx, float cy, cv::Matx33f rot, cv::Vec3f trans, bool leftH)
	{
		for (int v = 0; v < numV(); v++)
		{
			cv::Vec3f pos = verts[v];
			if (leftH)
				pos = cv::Vec3f(pos[0], -pos[2], pos[1]);
			pos = rot * pos + trans;
			pos = cv::Vec3f(fx * pos[0]/pos[2]  + cx, fy * pos[1]/pos[2] + cy, pos[2]);
			verts[v] = pos;
		}
	}

	std::vector<cv::Vec3f> calcVertNorm()
	{
		std::vector<cv::Vec3f> normArray(numV(), cv::Vec3f(0., 0., 0.));
		std::vector<double> sumWei(numV(), 0.);
		for (int f = 0; f < numF(); f++)
		{
			cv::Vec3i find = faceInds[f];
			cv::Vec3f d0 = verts[find[1]] - verts[find[0]];
			cv::Vec3f d1 = verts[find[2]] - verts[find[1]];
			cv::Vec3f fN = normalize(d0.cross(d1));
			float area = norm(fN)*0.5;
			for (int d = 0; d < 3; d++)
			{
				normArray[find[d]] += area * fN;
				sumWei[find[d]] += area;
			}
		}
		for (int v = 0; v < numV(); v++)
			if (sumWei[v] < 1.e-8)
				continue;
			else
				normArray[v] = normalize(normArray[v] / sumWei[v]);

		return normArray;
	}

	cv::Vec3f calcFaceNormal(int fID)
	{
		cv::Vec3i find = faceInds[fID];
		cv::Vec3f d0 = verts[find[1]] - verts[find[0]];
		cv::Vec3f d1 = verts[find[2]] - verts[find[1]];
		cv::Vec3f fN = normalize(d0.cross(d1));
		return fN;
	}

	std::vector<cv::Vec3f> calcFaceCenter()
	{
		std::vector<cv::Vec3f> fCenters(numF(), cv::Vec3f(0., 0., 0.));
		for (int fi = 0; fi < numF(); fi++)
		{
			cv::Vec3i fInd = faceInds[fi];
			for (int d = 0; d < 3; d++)
				fCenters[fi] += verts[fInd[d]];
			fCenters[fi] = fCenters[fi] / 3.;
		} // end for fi
		return fCenters;
	}

	std::vector<std::vector<int>> calcRingNeighbor()
	{
		std::vector<std::vector<int>> RingNei(numV(), std::vector<int>());
		for (int f = 0; f < numF(); f++)
		{
			cv::Vec3i fInd = faceInds[f];
			for (int fv = 0; fv < 3; fv++)
			{
				int fvi = fInd[fv];
				int f0 = fInd[(fv + 1) % 3];
				int f1 = fInd[(fv + 2) % 3];
				std::vector<int>::iterator it0, it1;
				it0 = std::find(RingNei[fvi].begin(), RingNei[fvi].end(), f0);
				it1 = std::find(RingNei[fvi].begin(), RingNei[fvi].end(), f1);
				if (it0 == RingNei[fvi].end())
					RingNei[fvi].push_back(f0);
				if (it1 == RingNei[fvi].end())
					RingNei[fvi].push_back(f1);
			} //end for fv
		} // end for f
		return RingNei;
	}

	float calAverageEdgeLen()
	{
		float avlen = 0.;
		int countEdge = 0;
		for (int f = 0; f < numF(); f++)
		{
			cv::Vec3i fInd = faceInds[f];
			for (int fv = 0; fv < 3; fv++)
			{
				int f0 = fInd[fv];
				int f1 = fInd[(fv + 1) % 3];
				cv::Vec3f v0 = verts[f0];
				cv::Vec3f v1 = verts[f1];
				float l_01 = norm(v0 - v1);
				avlen += l_01;
				countEdge += 1;
			}
		}
		return avlen / float(countEdge);
	}

	float calcAverageEdegeLen(std::vector<int> faceFlags)
	{
		float avlen = 0.;
		int countEdge = 0;
		for (int f = 0; f < numF(); f++)
		{
			if (faceFlags[f] < 1)
				continue;

			cv::Vec3i fInd = faceInds[f];
			for (int fv = 0; fv < 3; fv++)
			{
				int f0 = fInd[fv];
				int f1 = fInd[(fv + 1) % 3];
				cv::Vec3f v0 = verts[f0];
				cv::Vec3f v1 = verts[f1];
				float l_01 = norm(v0 - v1);
				avlen += l_01;
				countEdge += 1;
			}
		}
		return avlen / float(countEdge);
	}
};