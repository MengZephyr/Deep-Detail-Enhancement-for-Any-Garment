#pragma once

#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include "mesh.h"

using namespace std;

void readPly(std::string fName, std::vector<cv::Vec3f>& verts, std::vector<cv::Vec3i>& faces)
{
	ifstream plyStream(fName);
	if (!plyStream.is_open())
	{
		printf("Fail to read Ply file.\n");
		return;
	}
	int numV = 0;
	int numF = 0;
	string line;
	while (getline(plyStream, line))
	{
		stringstream ss;
		string cmd;
		ss << line;
		ss >> cmd;
		if (cmd == "end_header")
			break;
		if (cmd == "element")
		{
			ss >> cmd;
			if (cmd == "vertex")
				ss >> numV;
			if (cmd == "face")
				ss >> numF;
		}
	}
	verts.clear();
	faces.clear();
	int count = 0;
	while (getline(plyStream, line))
	{
		stringstream ss;
		ss << line;
		cv::Vec3f v;
		ss >> v[0] >> v[1] >> v[2];
		verts.push_back(v);
		count++;
		if (count == numV)
			break;
	}
	count = 0;
	while (getline(plyStream, line))
	{
		stringstream ss;
		ss << line;
		int n = 0;
		ss >> n;
		if (n == 3)
		{
			cv::Vec3i f;
			ss >> f[0] >> f[1] >> f[2];
			faces.push_back(f);
		}
		count++;
		if (count == numF)
			break;
	}
	plyStream.close();
}

void savePlyFile(std::string fName, ::vector<cv::Vec3f> verts, std::vector<cv::Vec3i> colors, std::vector<cv::Vec3i> faceID)
{
	int numV = verts.size();
	int numF = faceID.size();
	ofstream plyStream(fName);
	plyStream << "ply" << endl 
		<< "format ascii 1.0" << endl;
	plyStream << "element vertex " << numV << endl;
	plyStream << "property float x" << endl
		<< "property float y" << endl
		<< "property float z" << endl
		<< "property uchar red" << endl
		<< "property uchar green" << endl
		<< "property uchar blue" << endl
	    << "property uchar alpha" << endl;
	plyStream << "element face " << numF << endl;
	plyStream << "property list uchar int vertex_indices" << endl;
	plyStream << "end_header" << endl;
	for (int v = 0; v < numV; v++)
		plyStream << verts[v][0] << " " << verts[v][1] << " " << verts[v][2] << " "
		<< colors[v][0] << " " << colors[v][1] << " " << colors[v][2] << " " << "255" << endl;
	for (int f = 0; f < numF; f++)
		plyStream << "3 " << faceID[f][0] << " " << faceID[f][1] << " " << faceID[f][2] << endl;
	plyStream.close();
	plyStream.clear();
}

void saveVertTxt(std::string fName, std::vector<cv::Vec3f> vArray)
{
	int numV = vArray.size();
	ofstream txtStream(fName);
	txtStream << numV << endl;
	for (int v = 0; v < numV; v++)
		txtStream << vArray[v][0] << " " << vArray[v][1] << " " << vArray[v][2] << endl;
	txtStream.close();
	txtStream.clear();
}

void saveIFlagTxt(std::string fName, std::vector<int> vFs)
{
	int numV = vFs.size();
	ofstream txtStream(fName);
	txtStream << numV << endl;
	for (int v = 0; v < numV; v++)
		txtStream << vFs[v] << endl;
	txtStream.close();
	txtStream.clear();
}

void readFaceFile(std::string fName, std::vector<cv::Vec3i>& fInds)
{
	ifstream faceStream(fName);
	int numF = 0;
	string line;
	getline(faceStream, line);
	stringstream ss;
	string cmd;
	ss << line;
	ss >> numF;
	fInds.clear();
	while (getline(faceStream, line))
	{
		stringstream ss;
		ss << line;
		cv::Vec3i idx;
		ss >> idx[0]>> idx[1] >> idx[2];
		fInds.push_back(cv::Vec3i(idx[0]-1, idx[1]-1, idx[2]-1));
	}
	assert(fInds.size() == numF);
	faceStream.close();
}

void readTxtFile(std::string fName, std::vector<cv::Vec3f>& Iarray)
{
	ifstream InfoStream(fName);
	if (!InfoStream.is_open())
	{
		printf("Fail to read Txt file.\n");
		return;
	}
	int numV = 0;
	string line;
	getline(InfoStream, line);
	stringstream ss;
	string cmd;
	ss << line;
	ss >> numV;
	Iarray.clear();
	while (getline(InfoStream, line))
	{
		stringstream ss;
		ss << line;
		cv::Vec3f vec;
		ss >> vec[0] >> vec[1] >> vec[2];
		Iarray.push_back(vec);
	}
	assert(Iarray.size() == numV);
	InfoStream.close();
}

std::vector<cv::Vec3i> readB2OMap(std::string fName)
{
	ifstream InfoStream(fName);
	if (!InfoStream.is_open())
	{
		printf("Fail to read Obj file.\n");
		return std::vector<cv::Vec3i>();
	}

	std::vector<cv::Vec3i> mapArray;
	string line;
	getline(InfoStream, line);
	stringstream ss;
	string cmd;
	ss << line;
	ss >> cmd;
	int numV;
	ss >> numV;
	mapArray = std::vector<cv::Vec3i>(numV, cv::Vec3i(-1, -1, -1));
	int vcc = 0;
	while (getline(InfoStream, line))
	{
		stringstream ss;
		int nm;
		ss << line;
		ss >> nm;
		for (int n = 0; n < nm; n++)
			ss >> mapArray[vcc][n];
		vcc++;
	}
	return mapArray;
}

void readObjVertArray(std::string fName, std::vector<cv::Vec3f>& vArray)
{
	ifstream InfoStream(fName);
	if (!InfoStream.is_open())
	{
		printf("Fail to read Obj file.\n");
		return;
	}
	vArray.clear();
	string line;
	while (getline(InfoStream, line))
	{
		stringstream ss;
		string cmd;
		ss << line;
		ss >> cmd;
		if (cmd == "v")
		{
			cv::Vec3f v(0., 0., 0.);
			ss >> v[0] >> v[1] >> v[2];
			//printf("(%f, %f, %f)\n", v[0], v[1], v[2]);
			vArray.push_back(v);
		}
	}
}

void readObjNormArray(std::string fName, std::vector<cv::Vec3f>& nArray)
{
	ifstream InfoStream(fName);
	if (!InfoStream.is_open())
	{
		printf("Fail to read Obj file.\n");
		return;
	}
	nArray.clear();
	string line;
	while (getline(InfoStream, line))
	{
		stringstream ss;
		string cmd;
		ss << line;
		ss >> cmd;
		if (cmd == "vn")
		{
			cv::Vec3f v(0., 0., 0.);
			ss >> v[0] >> v[1] >> v[2];
			v = normalize(v);
			//printf("(%f, %f, %f)\n", v[0], v[1], v[2]);
			nArray.push_back(v);
		}
	}
}

void readO2BFile(std::string fName, std::vector<int>& fMap)
{
	ifstream MapStream(fName);
	int numF = 0;
	string line;
	getline(MapStream, line);
	stringstream ss;
	string cmd;
	ss << line;
	ss >> cmd;
	if (cmd == "#")
		ss >> numF;
	fMap.clear();
	while (getline(MapStream, line))
	{
		stringstream ss;
		ss << line;
		int id = 0;
		ss >> id;
		fMap.push_back(id);
	}
	assert(fMap.size() == numF);
	MapStream.close();
}

R_Mesh loadInfoMesh(std::string faceName, std::string vertName)
{
	R_Mesh model;
	readFaceFile(faceName, model.faceInds);
	readTxtFile(vertName, model.verts);
	printf("numV: %d, numF: %d\n", model.numV(), model.numF());
	return model;
}

void reloadVerts(R_Mesh* pMesh, string vertName)
{
	printf("numV: %d ", pMesh->numV());
	readTxtFile(vertName, pMesh->verts);
	printf("--> %d\n", pMesh->numV());
}

R_Mesh loadTextMeshes(std::string fName)
{
	R_Mesh model;
	readPly(fName, model.verts, model.faceInds);
	model.bbmin = cv::Vec3f(FLT_MAX, FLT_MAX, FLT_MAX);
	model.bbmax = cv::Vec3f(-FLT_MAX, -FLT_MAX, -FLT_MAX);
	for(int i = 0; i < model.numV(); i++)
	{
		cv::Vec3f v = model.verts[i];
		for(int d = 0; d < 3; d++)
		{
			model.bbmin[d] = MIN(model.bbmin[d], v[d]);
			model.bbmax[d] = MAX(model.bbmax[d], v[d]);
		}
	}
	printf("numV: %d, numF: %d\n", model.numV(), model.numF());
	return model;
}