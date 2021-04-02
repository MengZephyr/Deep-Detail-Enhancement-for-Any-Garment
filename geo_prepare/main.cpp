#include "rayTracer.h"
#include "DataIO.h"
#include "KDTree.h"

void old_test()
{
	std::string faceName = "./test/Face.txt";
	std::string vertName = "./test/0000195_n.txt";
	//std::string vertName = "D:/models/MD/GetInfoFbF/DataBase/baseDress/AAA/Chama_10_n/0000175_n.txt";
	R_Mesh info_model = loadInfoMesh(faceName, vertName);

	std::string texFName = "./test/uv_mesh.ply";
	R_Mesh text_model = loadTextMeshes(texFName);
	int H = 400, W = 400;
	text_model.OrthProjMesh(H, W);
	RayIntersection myTracer;
	myTracer.addObj(&text_model);

	//std::string prefix = "D:/models/MD/GetInfoFbF/DataBase/work/20191204/rst_n_n/";
	std::string prefix = "D:/models/MD/GetInfoFbF/DataBase/baseDress/AAA/Denim_Lightwei_10_n/";
	std::string savefix = "D:/models/MD/GetInfoFbF/DataBase/baseDress/texture/Denim_Lightwei_10_n/";
	for (int fID = 3; fID < 203; fID++)
	{
		char _buffer[8];
		std::snprintf(_buffer, sizeof(_buffer), "%07d", fID);
		vertName = prefix + string(_buffer) + "_n.txt";
		reloadVerts(&info_model, vertName);

		/*vertName = prefix + string(_buffer) + ".obj";
		std::vector<cv::Vec3f> normInfo;
		readObjVertArray(vertName, normInfo);*/

		cv::Mat textMap = cv::Mat::zeros(H, W, CV_32FC3);
		/*std::vector<cv::Vec2i> pixelArray;
		std::vector<cv::Vec3i> indexArray;
		std::vector<cv::Vec2f> uvArray;*/
		//cv::Mat uvMap = cv::Mat::zeros(H, W, CV_32FC2);
		//cv::Mat indMap = cv::Mat::zeros(H, W, CV_32SC3);
		const std::vector<cv::Vec3f>& normInfo = info_model.verts;
		const std::vector<cv::Vec3i>& FaceInfo = info_model.faceInds;
		for (int y = 0; y < H; y++)
		{
			for (int x = 0; x < W; x++)
			{
				cv::Vec3f ori(x, H - y, -10.);
				cv::Vec3f dir(0., 0., 1.);
				RTCHit h = myTracer.rayIntersection(ori, dir);
				int fID = h.primID;
				if (fID < 0)
					continue;
				else
				{
					cv::Vec3f color(0., 0., 0.);
					cv::Vec3i face = FaceInfo[fID];
					cv::Vec3f n0 = normalize(normInfo[face[0]]);
					cv::Vec3f n1 = normalize(normInfo[face[1]]);
					cv::Vec3f n2 = normalize(normInfo[face[2]]);
					color = (1. - h.u - h.v) * n0 + h.u * n1 + h.v * n2;
					color = (color + cv::Vec3f(1., 1., 1.)) * 0.5;
					textMap.at<cv::Vec3f>(y, x) = cv::Vec3f(color[2], color[1], color[0]);
					//uvMap.at<cv::Vec2f>(y, x) = cv::Vec2f(h.u, h.v);
					//indMap.at<cv::Vec3i>(y, x) = cv::Vec3i(face[0], face[1], face[2]);

					/*pixelArray.push_back(cv::Vec2i(x, y));
					indexArray.push_back(face);
					uvArray.push_back(cv::Vec2f(h.u, h.v));*/
				}

			} // end for x
		} // end for y

		std::string savename = savefix + string(_buffer) + ".png";
		//std::string savename = savefix + "mask.png";
		cv::imwrite(savename, textMap * 255.);
		//FILE* fp1 = fopen(std::string(savefix + "uvMap").c_str(), "wb");
		//fwrite(uvMap.data, 400 * 400, sizeof(cv::Vec2f), fp1);
		//fclose(fp1);
		//FILE* fp2 = fopen(std::string(savefix + "FaceMap").c_str(), "wb");
		//fwrite(indMap.data, 400 * 400, sizeof(cv::Vec3i), fp2);

		/*ofstream infoStream(savefix + "renderInfo.txt");
		for (int i = 0; i < pixelArray.size(); i++)
		{
			infoStream << pixelArray[i][0] << " " << pixelArray[i][1] << " "
				<< indexArray[i][0] << " " << indexArray[i][1] << " " << indexArray[i][2] << " "
				<< uvArray[i][0] << " " << uvArray[i][1] << endl;
		}
		infoStream.close();
		infoStream.clear();*/
	}
}

void today_test()
{
	std::string faceName = "./test/Face.txt";
	std::string normalName = "./test/0000001_n.txt";
	R_Mesh normal_model = loadInfoMesh(faceName, normalName);

	std::string vertName = "./test/0000001.txt";
	R_Mesh vert_model = loadInfoMesh(faceName, vertName);



	float fx = -1388.89, fy = 1388.89;
	int rx = 1000, ry = 800;
	float cx = rx / 2., cy = ry / 2.;
	cv::Matx33f rot = cv::Matx33f::eye();
	rot(0, 0) = 0.9999507069587708; rot(0, 1) = 0.009928660467267036; rot(0, 2) = 7.136259227991104e-08;
	rot(1, 0) = -0.000758613517973572; rot(1, 1) = 0.07639551162719727; rot(1, 2) = 0.9970772862434387;
	rot(2, 0) = 0.009899635799229145; rot(2, 1) = -0.997028112411499; rot(2, 2) = 0.07639927417039871;
	cv::Vec3f trans(0., 0., 0.);
	trans[0] = -0.2366400510072708; trans[1] = -0.6399520635604858; trans[2] = -3.0259854793548584;

	std::string prefix = "D:/models/MD/GetInfoFbF/DataBase/baseDress/AAA/Chama_10_n/";
	std::string savefix = "D:/models/MD/GetInfoFbF/DataBase/work/20191202/";
	for (int fID = 3; fID < 203; fID++)
	{
		char _buffer[8];
		std::snprintf(_buffer, sizeof(_buffer), "%07d", fID);
		vertName = prefix + string(_buffer) + ".txt";
		reloadVerts(&vert_model, vertName);
		normalName = prefix + string(_buffer) + "_n.txt";
		reloadVerts(&normal_model, normalName);

		//--- proj and render frames
		vert_model.cameraProjMesh(fx, fy, cx, cy, rot, trans, true);
		cv::Mat textMap = cv::Mat::zeros(ry, rx, CV_32FC3);
		RayIntersection myTracer;
		myTracer.addObj(&vert_model);
		const std::vector<cv::Vec3f>& normInfo = normal_model.verts;
		const std::vector<cv::Vec3i>& FaceInfo = normal_model.faceInds;

		for (int y = 0; y < ry; y++)
		{
			for (int x = 0; x < rx; x++)
			{
				cv::Vec3f ori(x, y, 10.);
				cv::Vec3f dir(0., 0., -1.);
				RTCHit h = myTracer.rayIntersection(ori, dir);
				int fID = h.primID;
				if (fID < 0)
					continue;
				else
				{

					cv::Vec3f color(0., 0., 0.);
					cv::Vec3i face = FaceInfo[fID];
					color = (1. - h.u - h.v) * normInfo[face[0]] + h.u * normInfo[face[1]] + h.v * normInfo[face[2]];
					color = (color + cv::Vec3f(1., 1., 1.)) * 0.5;
					textMap.at<cv::Vec3f>(y, x) = cv::Vec3f(color[2], color[1], color[0]);
					//textMap.at<cv::Vec3f>(y, x) = cv::Vec3f(1., 1., 0.);
				}

			} // end for x
		} // end for y

		std::string savename = savefix + string(_buffer) + ".png";
		cv::imwrite(savename, textMap * 255.);
	}
	
}

void dis_test()
{
	std::string faceName = "./test/Face.txt";
	std::string vertName = "./test/0000001.txt";
	R_Mesh vert_model = loadInfoMesh(faceName, vertName);

	float fx = -1388.89, fy = 1388.89;
	int rx = 1000, ry = 800;
	float cx = rx / 2., cy = ry / 2.;
	cv::Matx33f rot = cv::Matx33f::eye();
	rot(0, 0) = 0.9999507069587708; rot(0, 1) = 0.009928660467267036; rot(0, 2) = 7.136259227991104e-08;
	rot(1, 0) = -0.000758613517973572; rot(1, 1) = 0.07639551162719727; rot(1, 2) = 0.9970772862434387;
	rot(2, 0) = 0.009899635799229145; rot(2, 1) = -0.997028112411499; rot(2, 2) = 0.07639927417039871;
	cv::Vec3f trans(0., 0., 0.);
	trans[0] = -0.2366400510072708; trans[1] = -0.6399520635604858; trans[2] = -3.0259854793548584;

	std::string prefix = "D:/models/MD/GetInfoFbF/DataBase/baseDress/AAA/Chama_10_n/";
	std::string rstfix = "D:/models/MD/GetInfoFbF/DataBase/work/20191204/rst_3R/";
	std::string savefix = "D:/models/MD/GetInfoFbF/DataBase/work/20191204/render_Dist/";

	double d_min = 0.001;
	cv::Vec3d color_min(1., 1., 0.);
	double d_max = 0.1;
	cv::Vec3d color_max(1., 0., 1.);

	for (int fID = 3; fID < 203; fID++)
	{
		char _buffer[8];
		std::snprintf(_buffer, sizeof(_buffer), "%07d", fID);
		vertName = prefix + string(_buffer) + ".txt";
		reloadVerts(&vert_model, vertName);
		std::string rstName = rstfix + string(_buffer) + ".obj";

		std::vector<cv::Vec3f> rstArray;
		std::vector<cv::Vec3f> gtArray;
		readObjVertArray(rstName, rstArray);
		readTxtFile(vertName, gtArray);

		//--- proj and render frames
		vert_model.cameraProjMesh(fx, fy, cx, cy, rot, trans, true);
		cv::Mat textMap = cv::Mat::zeros(ry, rx, CV_32FC3);
		RayIntersection myTracer;
		myTracer.addObj(&vert_model);
		const std::vector<cv::Vec3i>& FaceInfo = vert_model.faceInds;

		for (int y = 0; y < ry; y++)
		{
			for (int x = 0; x < rx; x++)
			{
				cv::Vec3f ori(x, y, 10.);
				cv::Vec3f dir(0., 0., -1.);
				RTCHit h = myTracer.rayIntersection(ori, dir);
				int fID = h.primID;
				if (fID < 0)
					continue;
				else
				{
					cv::Vec3f color(0., 0., 0.);
					cv::Vec3i face = FaceInfo[fID];
					double D0 = norm(gtArray[face[0]] - rstArray[face[0]]);
					double D1 = norm(gtArray[face[1]] - rstArray[face[1]]);
					double D2 = norm(gtArray[face[2]] - rstArray[face[2]]);

					double dist = (1. - h.u - h.v)*D0 + h.u*D1 + h.v*D2;
					dist = (dist - d_min) / (d_max - d_min);

					//printf("D0: %f, D1: %f, D2: %f, dist: %f \n", D0, D1, D2, dist);
					dist = MAX(MIN(dist, 1.), 0.);
					//printf("dist: %f \n", dist);
					//return;

					color = color_min * (1. - dist) + color_max * dist;
					textMap.at<cv::Vec3f>(y, x) = cv::Vec3f(color[0], color[1], color[2]);
					//textMap.at<cv::Vec3f>(y, x) = cv::Vec3f(1., 1., 0.);
				}

			} // end for x
		} // end for y

		std::string savename = savefix + string(_buffer) + ".png";
		cv::imwrite(savename, textMap * 255.);
	}
}

cv::Mat blurGapTexture(cv::Mat oriImg, cv::Mat mask, int numIter=3)
{
	int iter = numIter; // enlarge iter pixels
	cv::Mat mm = mask.clone();
	cv::Mat rstImg = oriImg.clone();
	while ((iter--) > 0)
	{
		cv::Mat iterImg = rstImg.clone();
		cv::Mat iterMask = mm.clone();
		for (int y = 0; y < oriImg.rows; y++)
		{
			for (int x = 0; x < oriImg.cols; x++)
			{
				cv::Vec3f color(0., 0., 0.);
				int cc = 0;
				if (mask.at<int>(y, x) > 0)
					color = oriImg.at<cv::Vec3f>(y, x);
				else
				{
					for (int hy = -1; hy <= 1; hy++)
					{
						for (int hx = -1; hx <= 1; hx++)
						{
							int px = MAX(MIN(x + hx, oriImg.cols - 1), 0);
							int py = MAX(MIN(y + hy, oriImg.rows - 1), 0);
							if (iterMask.at<int>(py, px) > 0)
							{
								color += iterImg.at<cv::Vec3f>(py, px);
								cc += 1;
							}
						} // end for hx
					} // end for hy
					if (cc > 0)
					{
						color = color / float(cc);
						mm.at<int>(y, x) = 255;
					}
				}
				rstImg.at<cv::Vec3f>(y, x) = color;
			} // end for x
		} //end for y
	}
	return rstImg;
}

# define NORMAL_RES 512 

void GrabNormals()
{
	std::string caseName = "Chamuse_tango/skirt/";
	std::string F_prefix = "D:/models/MD/DataModel/DressOri/case_3/";

	std::string maskName = F_prefix + caseName + "Mask.png";
	std::string faceName = F_prefix + caseName + "uv/Face_10.txt";
	std::string vertPrif = F_prefix + caseName + "10_DsUs_C/";
	//std::string mapPrif = F_prefix + caseName + "img_gan_c1/";
	//std::string savePrif = F_prefix + caseName + "normal_gan_c1/";
	std::string mapPrif = "D:/models/MD/DetailTask/patchTest/" + caseName+ "img_ps/";
	std::string savePrif ="D:/models/MD/DetailTask/patchTest/" + caseName + "normal_ps/";
	int frame0 = 160, frame1 = 160;
	R_Mesh vModel;
	readFaceFile(faceName, vModel.faceInds);

	std::string texFName = F_prefix + caseName + "uv/10_uvMesh.ply";
	R_Mesh text_model = loadTextMeshes(texFName);

	std::string scalFName = F_prefix + caseName + "uv/txt_0.txt";
	int txtScale = 0;
	ifstream saleStream(scalFName);
	string line;
	getline(saleStream, line);
	stringstream ss;
	ss << line;
	ss >> txtScale;
	//txtScale = 1154;
	int H = txtScale, W = txtScale;

	text_model.OrthProjMesh(H, W);
	RayIntersection myTracer;
	myTracer.addObj(&text_model);
	
	const std::vector<cv::Vec3i>& GeoFaceInfo = vModel.faceInds;
	const std::vector<cv::Vec3i>& TexFaceInfo = text_model.faceInds;
	int numF = GeoFaceInfo.size();
	cv::Mat maskImg = cv::imread(maskName, cv::IMREAD_GRAYSCALE);
	maskImg.convertTo(maskImg, CV_32SC1);

	for (int fID = frame0; fID < frame1+1; fID++)
	{
		char _buffer1[8];
		std::snprintf(_buffer1, sizeof(_buffer1), "%07d", fID);

		string vertName = vertPrif + string(_buffer1) + ".txt";
		std::vector<cv::Vec3f> vertArray;
		readTxtFile(vertName, vertArray);
		int numV = vertArray.size();
		std::vector<cv::Vec3f> normArray(numV, cv::Vec3f(0., 0., 0.));

		std::vector<float> ccArray(numV, 0.);
		
		string mapName = mapPrif + string(_buffer1) + ".png";
		//cout << mapName;
		cv::Mat mapImg = cv::imread(mapName, cv::IMREAD_COLOR);
		mapImg.convertTo(mapImg, CV_32FC3);
		//-- use blur to handle gaps
		cv::Mat txtMap = blurGapTexture(mapImg, maskImg);
		//cv::imwrite(savePrif + "bt.png", txtMap);

		for (int faceID = 0; faceID < numF; faceID++)
		{
			cv::Vec3i txtF = TexFaceInfo[faceID];
			cv::Vec3i geoF = GeoFaceInfo[faceID];
			for (int fd = 0; fd < 3; fd++)
			{
				cv::Vec3f texPos = text_model.verts[txtF[fd]];
				cv::Vec3f normV = getInfoFromMat_3f(cv::Vec2f(texPos[0], H - texPos[1]), txtMap, maskImg) / 255.;
				normV = cv::Vec3f(normV[2], normV[1], normV[0]) * 2. - cv::Vec3f(1., 1., 1.);
				if (norm(normV) < 1.e-3)
					continue;
				normArray[geoF[fd]] += normV;
				ccArray[geoF[fd]] += 1.;
			}
		}
		std::vector<cv::Vec3i> color(numV);
		for (int vertID = 0; vertID < numV; vertID++)
		{
			cv::Vec3f n = normalize(normArray[vertID]/ccArray[vertID]);
			normArray[vertID] = n;
			color[vertID] = cv::Vec3i((n + cv::Vec3f(1., 1., 1.)) * 0.5 * 255.);
		}
		//printf("kk_2...");
		savePlyFile(savePrif + "grabTest.ply", vertArray, color, GeoFaceInfo);
		//while (1);

		std::string normSName = savePrif + string(_buffer1) + "_n.txt";
		saveVertTxt(normSName, normArray);
		
	} //end for fID
}

void calcMeshNormals()
{
	std::string caseName = "Chamuse_NoHem/";
	std::string F_prefix = "D:/models/MD/DataModel/DressOri/case_1/";

	int frame_0 = 2, frame_1 = 202;
	std::string faceName = F_prefix + caseName + "uv/Face_10.txt";
	std::string vertPrif = "D:/models/MD/DetailTask/patchTest/" + caseName + "geo_d10/";
	std::string savePrif = "D:/models/MD/DetailTask/patchTest/" + caseName + "geoToTxt/10_deform/";
	R_Mesh vModel;
	readFaceFile(faceName, vModel.faceInds);

	std::string scalFName = F_prefix + caseName + "uv/txt_0.txt";
	int txtScale = 0;
	ifstream saleStream(scalFName);
	string line;
	getline(saleStream, line);
	stringstream ss;
	ss << line;
	ss >> txtScale;
	int H = txtScale, W = txtScale;

	std::string texFName = F_prefix + caseName + "uv/10_uvMesh.ply";
	R_Mesh text_model = loadTextMeshes(texFName);
	text_model.OrthProjMesh(H, W);
	RayIntersection myTracer;
	myTracer.addObj(&text_model);

	const std::vector<cv::Vec3i>& FaceInfo = vModel.faceInds;
	for (int fID = frame_0; fID < frame_1+1; fID++)
	{
		char _buffer[8];
		char _buffer1[8];
		std::snprintf(_buffer1, sizeof(_buffer1), "%07d", fID);

		std::string vertName = vertPrif + string(_buffer1) + ".obj";
		readObjVertArray(vertName, vModel.verts);
		std::vector<cv::Vec3f> vNormal = vModel.calcVertNorm();

		/*std::string gtNormName = vertPrif + string(_buffer1) + "_n.txt";
		std::vector<cv::Vec3f> vNormal;
		readTxtFile(gtNormName, vNormal);*/

		cv::Mat rstNMap = cv::Mat::zeros(H, W, CV_32FC3);
		//cv::Mat gtNMap = cv::Mat::zeros(400, 400, CV_32FC3);

		for (int y = 0; y < H; y++)
		{
			for (int x = 0; x < W; x++)
			{
				cv::Vec3f ori(x, H - y, -10.);
				cv::Vec3f dir(0., 0., 1.);
				RTCHit h = myTracer.rayIntersection(ori, dir);
				int fID = h.primID;
				if (fID < 0)
					continue;
				else
				{
					cv::Vec3f color(0., 0., 0.);
					cv::Vec3i face = FaceInfo[fID];
					cv::Vec3f n0 = normalize(vNormal[face[0]]);
					cv::Vec3f n1 = normalize(vNormal[face[1]]);
					cv::Vec3f n2 = normalize(vNormal[face[2]]);
					color = (1. - h.u - h.v) * n0 + h.u * n1 + h.v * n2;
					color = (color + cv::Vec3f(1., 1., 1.)) * 0.5;
					rstNMap.at<cv::Vec3f>(y, x) = cv::Vec3f(color[2], color[1], color[0]);
				}
			}
		}
		std::string saveName = savePrif + string(_buffer1) + ".png";
		cv::imwrite(saveName, rstNMap * 255.);
	}
}

# define TXT_ALPHA 296.437866

void calcTexSize()
{
	std::string caseName = "Larkspur/pants_knitTerry/";
	std::string F_prefix = "D:/models/MD/DataModel/DressOri/case_7/";
	std::string geoName = F_prefix + caseName + "uv/geo.ply";
	R_Mesh geo_model = loadTextMeshes(geoName);
	float geo_avEdgeL = geo_model.calAverageEdgeLen();
	printf("geoLeng: %f\n", geo_avEdgeL);
	std::string uvName = F_prefix + caseName + "uv/uv.ply";
	R_Mesh txt_model = loadTextMeshes(uvName);
	float txt_avEdgeL = txt_model.calAverageEdgeLen();
	printf("geo avarage len: %f, txt avarage len: %f\n", geo_avEdgeL, txt_avEdgeL);
	int ImgS = int(TXT_ALPHA * geo_avEdgeL / txt_avEdgeL + 0.5);
	printf("Txt_Size: %d\n", ImgS);
	std::string saveName = F_prefix + caseName + "uv/txt_0.txt";
	ofstream txtStream(saveName);
	txtStream << ImgS << endl;
	txtStream.close();
	txtStream.clear();
}

void renderOrigTex()
{
	std::string caseName = "complex_evenCoarse/";
	std::string F_prefix = "D:/models/MD/DataModel/DressOri/case_1/";
	int frameID0 = 1, frameID1 = 850;
	int pD = 40;
	char _pbuff[8];
	std::snprintf(_pbuff, sizeof(_pbuff), "%d", pD);
	std::string pDS = std::string(_pbuff);

	//--load texture uv coord, face ID
	std::string texFName = F_prefix + caseName + "uv/" + pDS + "_uvMesh.ply";
	R_Mesh text_model = loadTextMeshes(texFName);

	std::string scalFName = F_prefix + caseName + "uv/txt_0.txt";
	int txtScale = 0;
	ifstream saleStream(scalFName);
	string line;
	getline(saleStream, line);
	stringstream ss;
	ss << line;
	ss >> txtScale;

	int H = txtScale, W = txtScale;
	//int H = 900, W = 900;
	text_model.OrthProjMesh(H, W);
	RayIntersection myTracer;
	myTracer.addObj(&text_model);

	const std::vector<cv::Vec3i>& FaceInfo = text_model.faceInds;
	std::string normFRoot = F_prefix + caseName + pDS + "_L/";
	std::string savePrif = F_prefix + caseName + "t_" + pDS + "_L/";

	R_Mesh vModel;
	vModel.faceInds = text_model.faceInds;
	for (int fID = frameID0; fID < frameID1+1; fID++)
	{
		char _buffer[8];
		std::snprintf(_buffer, sizeof(_buffer), "%07d", fID);
		std::string nfName = normFRoot + "PD" +  pDS + "_" + std::string(_buffer) + ".obj";

		readObjVertArray(nfName, vModel.verts);
		std::vector<cv::Vec3f> vNormal = vModel.calcVertNorm();

		/*std::vector<cv::Vec3f> vNormal;
		readObjNormArray(nfName, vNormal);*/

		cv::Mat rstNMap = cv::Mat::zeros(H, W, CV_32FC3);

		for (int y = 0; y < H; y++)
		{
			for (int x = 0; x < W; x++)
			{
				cv::Vec3f ori(x, H - y, -10.);
				cv::Vec3f dir(0., 0., 1.);
				RTCHit h = myTracer.rayIntersection(ori, dir);
				int fID = h.primID;
				if (fID < 0)
					continue;
				else
				{
					cv::Vec3f color(0., 0., 0.);
					cv::Vec3i face = FaceInfo[fID];
					cv::Vec3f n0 = normalize(vNormal[face[0]]);
					cv::Vec3f n1 = normalize(vNormal[face[1]]);
					cv::Vec3f n2 = normalize(vNormal[face[2]]);
					color = (1. - h.u - h.v) * n0 + h.u * n1 + h.v * n2;
					color = (color + cv::Vec3f(1., 1., 1.)) * 0.5;
					rstNMap.at<cv::Vec3f>(y, x) = cv::Vec3f(color[2], color[1], color[0]);
					//rstNMap.at<cv::Vec3f>(y, x) = cv::Vec3f(1., 1., 1.);
				}
			}
		}

		std::string saveName = savePrif + string(_buffer) + ".png";
		//std::string saveName = F_prefix + caseName + "Mask.png";
		cv::imwrite(saveName, rstNMap * 255.);

	} // end for fID
}

void UpSampleBaseTxt()
{
	std::string caseName = "Chamuse_tango/skirt/";
	std::string F_prefix = "D:/models/MD/DataModel/DressOri/case_3/";

	std::string Txt30File = F_prefix + caseName + "uv/30_uvMesh.ply";
	std::string Txt10File = F_prefix + caseName + "uv/10_uvMesh.ply";
	R_Mesh text30_model = loadTextMeshes(Txt30File);
	R_Mesh text10_model = loadTextMeshes(Txt10File);


	std::string scalFName = F_prefix + caseName + "uv/txt_0.txt";
	int txtScale = 0;
	ifstream saleStream(scalFName);
	string line;
	getline(saleStream, line);
	stringstream ss;
	ss << line;
	ss >> txtScale;

	int H = txtScale, W = txtScale;

	text30_model.OrthProjMesh(H, W);
	text10_model.OrthProjMesh(H, W);

	int numV10 = text10_model.numV();
	std::vector<cv::Vec4f> mapUV10(numV10, cv::Vec4f(-1, -1, -1, -1));

	//-- trayTrace Map [1, u, v, faceID]
	RayIntersection myTracer;
	myTracer.addObj(&text30_model);

	std::vector<cv::Vec3f>& vertA10 = text10_model.verts;
	std::vector<cv::Vec3i>& faceI30 = text30_model.faceInds;
	for (int vi = 0; vi < numV10; vi++)
	{
		cv::Vec3f vPos = vertA10[vi];
		cv::Vec3f ori(vPos[0], vPos[1], -10.);
		cv::Vec3f dir(0., 0., 1.);
		RTCHit h = myTracer.rayIntersection(ori, dir);
		int fID = h.primID;
		if (fID < 0)
			continue;
		else
			mapUV10[vi] = cv::Vec4f(1, h.u, h.v, fID);
	} // end for vi


	//--kdtree Map [-1, vID0, vID1, a]
	KDTree vertTree(text30_model.verts);
	std::vector<cv::Vec3f> fCenters = text30_model.calcFaceCenter();
	KDTree FaceTree(fCenters);
	for (int vi = 0; vi < numV10; vi++)
	{
		if (mapUV10[vi][0] < 0)
		{
			cv::Vec3f vPos = vertA10[vi];
			std::vector<KDTreeLeaf> leaf = vertTree.searchKNN(vPos, 1);
			cv::Vec3f nPos = leaf[0].pos;
			if (norm(vPos - nPos) < 1.e-5)
			{
				int id0 = leaf[0].id;
				mapUV10[vi] = cv::Vec4f(-1, id0, id0, 1.);
			}
			else
			{
				std::vector<KDTreeLeaf> NeiFaceL = FaceTree.searchKNN(vPos, 1);
				int fID = NeiFaceL[0].id;
				float minProj = FLT_MAX;
				cv::Vec2i id01(-1, -1);
				for (int d = 0; d < 3; d++)
				{
					int sid = faceI30[fID][d];
					int eid = faceI30[fID][(d + 1) % 3];
					float dd = norm((vPos - text30_model.verts[sid]).cross(text30_model.verts[eid] - text30_model.verts[sid]));
					if (dd < minProj)
					{
						minProj = dd;
						id01 = cv::Vec2i(sid, eid);
					}
				}
				float aa = norm(vPos - text30_model.verts[id01[0]]);
				float bb = norm(text30_model.verts[id01[1]] - text30_model.verts[id01[0]]);
				mapUV10[vi] = cv::Vec4f(-1, id01[0], id01[1], aa/bb);
			}
		}
	} // end for vi


	//-- Grab 3D position
	std::string BORoot = F_prefix + caseName + "uv/ind_B2O_10.txt";
	std::vector<cv::Vec3i> BOMap = readB2OMap(BORoot);
	std::string GeoFRoot = F_prefix + caseName + "10_Ds_L/";
	std::string faceName = F_prefix + caseName + "uv/Face_10.txt";
	std::string saveRoot = F_prefix + caseName + "10_DsUs_C/";
	int frame0 = 100;
	int frame1 = 200;
	std::vector<cv::Vec3i> rstFaceInd;
	readFaceFile(faceName, rstFaceInd);

	for (int fID = frame0; fID < frame1 + 1; fID++)
	{
		char _buffer[8];
		std::snprintf(_buffer, sizeof(_buffer), "%07d", fID);

		/*std::string vfName = GeoFRoot + "PD40_" + std::string(_buffer) + ".obj";
		std::vector<cv::Vec3f> gVertsArray;
		readObjVertArray(vfName, gVertsArray);*/

		std::string vfName = GeoFRoot + std::string(_buffer) + ".txt";
		std::vector<cv::Vec3f> gVertsArray;
		readTxtFile(vfName, gVertsArray);

		R_Mesh DS_Mesh;
		DS_Mesh.faceInds = text10_model.faceInds;
		//DS_Mesh.faceInds = rstFaceInd;
		DS_Mesh.verts = std::vector<cv::Vec3f>(numV10, cv::Vec3f(0., 0., 0.));
		for (int vi = 0; vi < numV10; vi++)
		{
			cv::Vec4f vM = mapUV10[vi];
			if (vM[0] > 0)
			{
				int fID = vM[3];
				cv::Vec3i fface = faceI30[fID];
				cv::Vec3f v0 = gVertsArray[fface[0]];
				cv::Vec3f v1 = gVertsArray[fface[1]];
				cv::Vec3f v2 = gVertsArray[fface[2]];
				DS_Mesh.verts[vi] = (1. - vM[1] - vM[2]) * v0 + vM[1] * v1 + vM[2] * v2;
			}
			else
			{
				cv::Vec3f v0 = gVertsArray[vM[1]];
				cv::Vec3f v1 = gVertsArray[vM[2]];
				DS_Mesh.verts[vi] = (1. - vM[3]) * v0 + vM[3] * v1;
				//printf("%d, [%f, %f, %f, %f]\n", vi, vM[0], vM[1], vM[2], vM[3]);
				//printf("[%f, %f, %f]\n", DS_Mesh.verts[vi][0], DS_Mesh.verts[vi][1], DS_Mesh.verts[vi][2]);
			}
		} // end for vi

		//--combine vertex pieces
		std::vector<cv::Vec3f> rstVerts(BOMap.size(), cv::Vec3f(0., 0., 0.));
		for (int vi = 0; vi < BOMap.size(); vi++)
		{
			cv::Vec3i mID = BOMap[vi];
			cv::Vec3f vpos(0., 0., 0.);
			int c = 0;
			for (int d = 0; d < 3; d++)
			{
				if (mID[d] >= 0)
				{
					vpos += DS_Mesh.verts[mID[d]];
					c++;
				}
			}
			rstVerts[vi] = vpos / float(c);
		} // end for vi

		saveVertTxt(saveRoot + std::string(_buffer) + ".txt", rstVerts);
		savePlyFile(saveRoot + std::string(_buffer) + ".ply",
			rstVerts, std::vector<cv::Vec3i>(numV10, cv::Vec3i(192, 192, 192)), rstFaceInd);
		//while (1);

		KDTree rst_geoTree(rstVerts);
		std::vector<int> cInd(rstVerts.size(), 0);
		for (int v = 0; v < gVertsArray.size(); v++)
		{
			cv::Vec3f vPos = gVertsArray[v];
			std::vector<KDTreeLeaf> leaf = rst_geoTree.searchKNN(vPos, 1);
			cInd[leaf[0].id] = 1;
		}
		saveIFlagTxt(saveRoot + std::string(_buffer) + "_f.txt", cInd);
	} // end for fID
}

void DownSmapleBaseTxt(bool ifGeoSave, bool ifRenderTxt)
{
	std::string caseName = "case_4/ShortDress/Chamuse/";
	std::string F_prefix = "D:/models/MD/DataModel/DressOri/";
	int frameID0 = 1, frameID1 = 743;
	//--sample map from texture 30 --> 10 PD
	std::string Txt30File = F_prefix + caseName + "uv/30_uvMesh.ply";
	std::string Txt10File = F_prefix + caseName + "uv/10_uvMesh.ply";
	R_Mesh text30_model = loadTextMeshes(Txt30File);
	R_Mesh text10_model = loadTextMeshes(Txt10File);

	std::string scalFName = F_prefix + caseName + "uv/txt_0.txt";
	int txtScale = 0;
	ifstream saleStream(scalFName);
	string line;
	getline(saleStream, line);
	stringstream ss;
	ss << line;
	ss >> txtScale;

	int H = txtScale, W = txtScale;

	text30_model.OrthProjMesh(H, W);
	text10_model.OrthProjMesh(H, W);
	
	int numV30 = text30_model.numV();
	std::vector<cv::Vec3f> mapUV30(numV30, cv::Vec3f(-1, -1, -1));
	//--trayTrance Map [u, v, faceID]
	RayIntersection myTracer;
	myTracer.addObj(&text10_model);

	std::vector<cv::Vec3f>& vertA30 = text30_model.verts;
	std::vector<cv::Vec3i>& faceI10 = text10_model.faceInds;
	for (int vi = 0; vi < numV30; vi++)
	{
		cv::Vec3f vPos = vertA30[vi];
		cv::Vec3f ori(vPos[0], vPos[1], -10.);
		cv::Vec3f dir(0., 0., 1.);
		RTCHit h = myTracer.rayIntersection(ori, dir);
		int fID = h.primID;
		if (fID < 0)
			continue;
		else
			mapUV30[vi] = cv::Vec3f(h.u, h.v, fID);
	} // end for vi
	
	//--kdtree Map [-1, -1, vID]
	KDTree vertTree(text10_model.verts);
	for (int vi = 0; vi < numV30; vi++)
	{
		if (mapUV30[vi][0] < 0.)
		{
			cv::Vec3f vPos = vertA30[vi];
			KDTreeLeaf nLeaf = vertTree.search(vPos);
			mapUV30[vi] = cv::Vec3f(-1., -1., nLeaf.id);
		}
	}

	//-- Grab 3D position
	std::string GeoFRoot = F_prefix + caseName + "10_L/";
	std::string saveRoot = F_prefix + caseName;
	RayIntersection RenderTracer;
	RenderTracer.addObj(&text30_model);

	for (int fID = frameID0; fID < frameID1+1; fID++)
	{
		char _buffer[8];
		std::snprintf(_buffer, sizeof(_buffer), "%07d", fID);
		std::string vfName = GeoFRoot + "PD10_" + std::string(_buffer) + ".obj";
		std::vector<cv::Vec3f> gVertsArray;
		readObjVertArray(vfName, gVertsArray);

		R_Mesh DS_Mesh;
		DS_Mesh.faceInds = text30_model.faceInds;
		DS_Mesh.verts = std::vector<cv::Vec3f>(numV30, cv::Vec3f(0., 0., 0.));
		for (int vi = 0; vi < numV30; vi++)
		{
			cv::Vec3f vM = mapUV30[vi];
			if (vM[0] < 0.)
			{
				int vID = vM[2];
				DS_Mesh.verts[vi] = gVertsArray[vID];
			}
			else
			{
				int fID = vM[2];
				cv::Vec3i fface = faceI10[fID];
				cv::Vec3f v0 = gVertsArray[fface[0]];
				cv::Vec3f v1 = gVertsArray[fface[1]];
				cv::Vec3f v2 = gVertsArray[fface[2]];
				DS_Mesh.verts[vi] = (1. - vM[0] - vM[1]) * v0 + vM[0] * v1 + vM[1] * v2;
			}
		}
		if (ifGeoSave)
		{
			std::string saveName = saveRoot + "10_Ds_L/" + string(_buffer) + ".txt";
			saveVertTxt(saveName, DS_Mesh.verts);
		}

		if (ifRenderTxt)
		{
			//-- render normal
			const std::vector<cv::Vec3i>& FaceInfo = text30_model.faceInds;
			std::vector<cv::Vec3f> vNormal = DS_Mesh.calcVertNorm();
			cv::Mat rstNMap = cv::Mat::zeros(H, W, CV_32FC3);
			for (int y = 0; y < H; y++)
			{
				for (int x = 0; x < W; x++)
				{
					cv::Vec3f ori(x, H - y, -10.);
					cv::Vec3f dir(0., 0., 1.);
					RTCHit h = RenderTracer.rayIntersection(ori, dir);
					int fID = h.primID;
					if (fID < 0)
						continue;
					else
					{
						cv::Vec3f color(0., 0., 0.);
						cv::Vec3i face = FaceInfo[fID];
						cv::Vec3f n0 = normalize(vNormal[face[0]]);
						cv::Vec3f n1 = normalize(vNormal[face[1]]);
						cv::Vec3f n2 = normalize(vNormal[face[2]]);
						color = (1. - h.u - h.v) * n0 + h.u * n1 + h.v * n2;
						color = (color + cv::Vec3f(1., 1., 1.)) * 0.5;
						rstNMap.at<cv::Vec3f>(y, x) = cv::Vec3f(color[2], color[1], color[0]);
					}
				}
			}
			std::string saveName = saveRoot + "t_Ds10_L/" + string(_buffer) + ".png";
			cv::imwrite(saveName, rstNMap * 255.);
		}
	}
	return;
}

#define M_PI 3.14159265358979

void calcDist_NormalMap()
{
	std::string mapFile_1 = "D:/models/MD/DetailTask/test/Geo/t_rst_Ds/Chamuse/";
	std::string mapFile_2 = "D:/models/MD/DetailTask/test/rst/case_1/Chamuse/";
	std::string saveRoot = "D:/models/MD/DetailTask/test/Geo/Dist_Rt_rsDs/Chamuse/";
	std::string maskName = "D:/models/MD/DataModel/DressOri/case_1/Mask.png";
	cv::Mat maskImg = cv::imread(maskName, cv::IMREAD_GRAYSCALE);
	maskImg.convertTo(maskImg, CV_32SC1);
	int H = maskImg.rows, W = maskImg.cols;
	int frame_0 = 400, frame_1 = 600;
	cv::Vec3f mxC(1., 0., 1.), mnC(1., 1., 0.);
	float thr = 0.85;
	cout << acos(thr) * 180. / M_PI << endl;
	while (1);
	for (int fID = frame_0; fID < frame_1 + 1; fID++)
	{
		char _buffer[8];
		std::snprintf(_buffer, sizeof(_buffer), "%07d", fID);
		std::string imageName_1 = mapFile_1 + std::string(_buffer) + ".png";
		std::string imageName_2 = mapFile_2 + std::string(_buffer) + ".png";
		cv::Mat img_1 = cv::imread(imageName_1, cv::IMREAD_COLOR);
		img_1.convertTo(img_1, CV_32FC3);
		cv::Mat img_2 = cv::imread(imageName_2, cv::IMREAD_COLOR);
		img_2.convertTo(img_2, CV_32FC3);
		cv::Mat distMap = cv::Mat::zeros(H, W, CV_32FC3);
		for (int yi = 0; yi < H; yi++)
		{
			for (int xi = 0; xi < W; xi++)
			{
				if (maskImg.at<int>(yi, xi) > 0)
				{
					cv::Vec3f normV_1 = img_1.at<cv::Vec3f>(yi, xi)/255.;
					//cout << normV_1;
					normV_1 = cv::Vec3f(normV_1[2], normV_1[1], normV_1[0]) * 2. - cv::Vec3f(1., 1., 1.);
					//cout << normV_1;
					normV_1 = normalize(normV_1);
					//cout << normV_1;

					cv::Vec3f normV_2 = img_2.at<cv::Vec3f>(yi, xi)/255.;
					//cout << normV_2;
					normV_2 = cv::Vec3f(normV_2[2], normV_2[1], normV_2[0]) * 2. - cv::Vec3f(1., 1., 1.);
					//cout << normV_2;
					normV_2 = normalize(normV_2);
					//cout << normV_2;

					float dist = (normV_1.dot(normV_2) - thr) / (1. - thr);
					//cout << dist << endl;
					dist = 1. - dist;
					//cout << dist << endl;
					//cout << cos(M_PI/7.);
					dist = MAX(MIN(dist, 1.), 0.);
					cv::Vec3f cc = (1. - dist) * mnC + dist * mxC;
					//cout << dist << endl;
					//while (1);
					//cv::Vec3f cc = mnC;
					distMap.at<cv::Vec3f>(yi, xi) = cc;
				}
			} // end for xi
		} // end for yi
		std::string saveName = saveRoot + std::string(_buffer) + ".png";
		cv::imwrite(saveName, distMap * 255);
	} // end for fID
}

int JJID[8] = { 3, 4, 13, 14, 15, 16, 17, 18 }; // {spine_1, spine, leftUpLeg, leftLeg, leftFoot, rightUpLeg, rightLeg, rightFoot}
#define BODY_DIV 3
void GenerateJJMask()
{
	std::string caseName = "case_1/Chamuse/";
	std::string F_prefix = "D:/models/MD/DataModel/DressOri/";
	int TPOSFID = 2;
	int frameID0 = TPOSFID, frameID1 = 850;

	char _Tbuffer[8];
	std::snprintf(_Tbuffer, sizeof(_Tbuffer), "%07d", TPOSFID);
	std::string TPosBoneFName = "D:/models/MD/DataModel/Motions/JJ/case_1/" + std::string(_Tbuffer) + ".txt";
	std::vector<cv::Vec3f> TVerts;
	readTxtFile(TPosBoneFName, TVerts);
	float minTZ = FLT_MAX, maxTZ = 0;
	for (int v = 0; v < TVerts.size(); v++)
	{
		minTZ = MIN(minTZ, TVerts[v][2]);
		maxTZ = MAX(maxTZ, TVerts[v][2]);
	} // end for v
	printf("minZ: %f, maxZ: %f\n", minTZ, maxTZ);
	float DDRef = (maxTZ - minTZ) / float(BODY_DIV);
	printf("DDR: %f\n", DDRef);

	std::string scalFName = F_prefix + caseName + "uv/txt_0.txt";
	int txtScale = 0;
	ifstream saleStream(scalFName);
	string line;
	getline(saleStream, line);
	stringstream ss;
	ss << line;
	ss >> txtScale;
	printf("ImgSize: %d\n", txtScale);

	int H = txtScale, W = txtScale;

	//--load texture uv coord, face ID
	std::string texFName = F_prefix + caseName + "uv/30_uvMesh.ply";
	R_Mesh text_model = loadTextMeshes(texFName);
	text_model.OrthProjMesh(H, W);
	RayIntersection myTracer;
	myTracer.addObj(&text_model);
	const std::vector<cv::Vec3i>& FaceInfo = text_model.faceInds;


	for (int fID = frameID0; fID < frameID1 + 1; fID++)
	{
		char _buffer[8];
		std::snprintf(_buffer, sizeof(_buffer), "%07d", fID);

		/*std::string vfName = F_prefix + caseName + "10_Ds_L/" + std::string(_buffer) + ".txt";
		std::vector<cv::Vec3f> vertArray;
		readTxtFile(vfName, vertArray);*/

		std::string vfName = F_prefix + caseName + "30_L/PD30_" + std::string(_buffer) + ".obj";
		std::vector<cv::Vec3f> vertArray;
		readObjVertArray(vfName, vertArray);

		std::string bfName = "D:/models/MD/DataModel/Motions/JJ/case_1/" + std::string(_buffer) + ".txt";
		std::vector<cv::Vec3f> boneArray;
		readTxtFile(bfName, boneArray);

		for (int mID = 0; mID < 8; mID++)
		{
			cv::Vec3f anchorPnt = boneArray[JJID[mID]];
			std::vector<float> weiArray(vertArray.size());
			float maxW = 0;
			for (int v = 0; v < vertArray.size(); v++)
			{
				cv::Vec3f vPos = vertArray[v];
				vPos = cv::Vec3f(vPos[0], -vPos[2], vPos[1]);
				float dist = norm(vPos - anchorPnt);
				weiArray[v] = exp(-(dist * dist) / (0.25 * DDRef * DDRef));
				maxW = MAX(maxW, weiArray[v]);
			} // end for vv

			cv::Mat rstDMap = cv::Mat::zeros(H, W, CV_32FC1);

			for (int y = 0; y < H; y++)
			{
				for (int x = 0; x < W; x++)
				{
					cv::Vec3f ori(x, H - y, -10.);
					cv::Vec3f dir(0., 0., 1.);
					RTCHit h = myTracer.rayIntersection(ori, dir);
					int fID = h.primID;
					if (fID < 0)
						continue;
					else
					{
						float color = 0;
						cv::Vec3i face = FaceInfo[fID];
						float w0 = weiArray[face[0]];
						float w1 = weiArray[face[1]];
						float w2 = weiArray[face[2]];
						color = (1. - h.u - h.v) * w0 + h.u * w1 + h.v * w2;
						rstDMap.at<float>(y, x) = color;
						//rstNMap.at<cv::Vec3f>(y, x) = cv::Vec3f(1., 1., 1.);
					}
				}
			}

			char _BIDBuffer[8];
			std::snprintf(_BIDBuffer, sizeof(_BIDBuffer), "%d", mID);
			std::string saveName = F_prefix + caseName + "/uv/JJWei_30/" + std::string(_BIDBuffer) + "_" + std::string(_buffer) + ".png";
			cv::imwrite(saveName, rstDMap * 255.);
		} // end for mID
		printf("Frame_%d...Done\n", fID);
	}

}

void GenVolFromLowDriver()
{
	std::string caseName = "case_1/Chamuse_NoHem/";
	std::string F_prefix = "D:/models/MD/DataModel/DressOri/";
	int PrevFrame = 1;
	int frameID0 = PrevFrame+1, frameID1 = 850;

	std::string scalFName = F_prefix + caseName + "uv/txt_0.txt";
	int txtScale = 0;
	ifstream saleStream(scalFName);
	string line;
	getline(saleStream, line);
	stringstream ss;
	ss << line;
	ss >> txtScale;
	printf("ImgSize: %d\n", txtScale);

	int H = txtScale, W = txtScale;

	//-- PD 30 perframe information
	//--load texture uv coord, face ID
	std::string texFName = F_prefix + caseName + "uv/30_uvMesh.ply";
	R_Mesh text_model = loadTextMeshes(texFName);
	text_model.OrthProjMesh(H, W);
	RayIntersection myTracer;
	myTracer.addObj(&text_model);
	const std::vector<cv::Vec3i>& FaceInfo = text_model.faceInds;

	std::string L_MShape_Root = F_prefix + caseName + "10_Ds_L/";
	char P_buffer[8];
	std::snprintf(P_buffer, sizeof(P_buffer), "%07d", PrevFrame);

	std::string Prev_MName = L_MShape_Root + std::string(P_buffer) + ".txt";
	std::vector<cv::Vec3f> Prev_verts;
	readTxtFile(Prev_MName, Prev_verts);

	/*std::string Prev_MName = L_MShape_Root + std::string(P_buffer) + ".obj";
	std::vector<cv::Vec3f> Prev_verts;
	readObjVertArray(Prev_MName, Prev_verts);*/

	for (int nextFrame = frameID0; nextFrame < frameID1 + 1; nextFrame++)
	{
		char N_buffer[8];
		std::snprintf(N_buffer, sizeof(N_buffer), "%07d", nextFrame);

		std::string Next_MName = L_MShape_Root + std::string(N_buffer) + ".txt";
		std::vector<cv::Vec3f> Next_verts;
		readTxtFile(Next_MName, Next_verts);

		/*std::string Next_MName = L_MShape_Root + std::string(N_buffer) + ".obj";
		std::vector<cv::Vec3f> Next_verts;
		readObjVertArray(Next_MName, Next_verts);*/

		std::vector<cv::Vec3f> DeltaVArray(Next_verts.size(), cv::Vec3f(0., 0., 0.));
		for (int v = 0; v < Next_verts.size(); v++)
			DeltaVArray[v] = Next_verts[v] - Prev_verts[v];

		std::string DSaveN = F_prefix + caseName + "10v_Ds_L/" + std::string(N_buffer) + ".txt";
		saveVertTxt(DSaveN, DeltaVArray);

		Prev_verts.swap(Next_verts);

		////--proj to txt map
		//cv::Mat currM = cv::Mat::zeros(H, W, CV_32FC3);
		//for (int y = 0; y < H; y++)
		//{
		//	for (int x = 0; x < W; x++)
		//	{
		//		cv::Vec3f ori(x, H - y, -10.);
		//		cv::Vec3f dir(0., 0., 1.);
		//		RTCHit h = myTracer.rayIntersection(ori, dir);
		//		int fID = h.primID;
		//		if (fID < 0)
		//			continue;
		//		else
		//		{
		//			cv::Vec3f vol(0., 0., 0.);
		//			cv::Vec3i face = FaceInfo[fID];
		//			cv::Vec3f v0 = DeltaVArray[face[0]];
		//			cv::Vec3f v1 = DeltaVArray[face[1]];
		//			cv::Vec3f v2 = DeltaVArray[face[2]];
		//			vol = (1. - h.u - h.v) * v0 + h.u * v1 + h.v * v2;
		//			currM.at<cv::Vec3f>(y, x) = vol;
		//		}
		//	} // end for x
		//} // end for y


	} // end for nextFrame

	// Please use Function: UpSampleBaseTxt() to upsample Volocity for PD10 cases
}

void CMaskToGMask(std::string pref_Root)
{
	std::string maskName = pref_Root + "textures/ori_mask.jpg";
	cv::Mat bgrMask = cv::imread(maskName, cv::IMREAD_COLOR);
	bgrMask.convertTo(bgrMask, CV_32SC3);
	cv::Mat grayMask = cv::Mat::zeros(bgrMask.rows, bgrMask.cols, CV_32SC1);
	for (int y = 0; y < bgrMask.rows; y++)
	{
		for (int x = 0; x < bgrMask.cols; x++)
		{
			cv::Vec3i ccv = bgrMask.at<cv::Vec3i>(y, x);
			if (ccv[0] + ccv[1] + ccv[2] < 500)
				grayMask.at<int>(y, x) = 255;
		}
	}
	cv::imwrite(pref_Root + "textures/gmask.png", grayMask);
}

inline void calcTSpace(cv::Vec3f p, cv::Vec3f p1, cv::Vec3f p2,
	cv::Vec2f uv, cv::Vec2f uv1, cv::Vec2f uv2,
	cv::Vec3f& TVector)
{
	cv::Vec3f E1 = p1 - p;
	cv::Vec3f E2 = p2 - p;
	float Du1 = uv1[0] - uv[0];
	float Dv1 = uv1[1] - uv[1];
	float Du2 = uv2[0] - uv[0];
	float Dv2 = uv2[1] - uv[1];

	TVector = cv::Vec3f(0., 0., 0.);
	//BVector = cv::Vec3f(0., 0., 0.);
	float f = 1. / (Du1 * Dv2 - Du2 * Dv1);
	TVector[0] = f * (Dv2 * E1[0] - Dv1 * E2[0]);
	TVector[1] = f * (Dv2 * E1[1] - Dv1 * E2[1]);
	TVector[2] = f * (Dv2 * E1[2] - Dv1 * E2[2]);
	/*BVector[0] = f * (-Du2 * E1[0] + Du1 * E2[0]);
	BVector[1] = f * (-Du2 * E1[1] + Du1 * E2[1]);
	BVector[2] = f * (-Du2 * E1[2] + Du1 * E2[2]);*/
	TVector = normalize(TVector);
	//BVector = normalize(BVector);
}

void ConvertTanS_To_WorldS(std::string pref_Root, int BS)
{
	int frame0 = 1, frame1 = 191;
	std::string caseName = "/";
	std::string casePrefR = pref_Root + caseName;
	std::string maskName = casePrefR + "textures/gmask.png";
	cv::Mat Mask = cv::imread(maskName, cv::IMREAD_GRAYSCALE);
	Mask.convertTo(Mask, CV_32SC1);
	int H = Mask.rows, W = Mask.cols;

	std::string normalName = casePrefR + "textures/constN.png";
	//std::string normalName = casePrefR + "textures/f201_body_normal.jpg";
	cv::Mat normMap = cv::imread(normalName, cv::IMREAD_COLOR);
	normMap.convertTo(normMap, CV_32FC3);

	R_Mesh uvMesh;
	readPly(casePrefR + "uv/uv.ply", uvMesh.verts, uvMesh.faceInds);
	uvMesh.OrthProjMesh(H, W);
	RayIntersection myTracer;
	myTracer.addObj(&uvMesh);

	for (int f = frame0; f < frame1 + 1; f++)
	{
		char P_buffer[8];
		std::snprintf(P_buffer, sizeof(P_buffer), "%06d", f);
		//std::string geoName = casePrefR + "Meshes/" + std::string(P_buffer) + ".ply";

		R_Mesh geoMesh;
		readPly(casePrefR + "Meshes/" + std::string(P_buffer) + ".ply", geoMesh.verts, geoMesh.faceInds);
		std::vector<cv::Vec3f> vertNorm = geoMesh.calcVertNorm();
		std::vector<int> FaceF(geoMesh.numF(), -1);

		cv::Mat baseNormalMap = cv::Mat::zeros(H, W, CV_32FC3);
		cv::Mat baseMask = cv::Mat::zeros(H, W, CV_32SC1);

		for (int y = 0; y < H; y++)
		{
			for (int x = 0; x < W; x++)
			{
				if (Mask.at<int>(y, x) > 100)
				{
					cv::Vec3f ori(x, H - y, -10.);
					cv::Vec3f dir(0., 0., 1.);
					RTCHit h = myTracer.rayIntersection(ori, dir);
					int fID = h.primID;
					if (fID < 0)
						continue;
					else
					{
						FaceF[fID] = 2;
						cv::Vec3f N(0., 0., 0.);
						cv::Vec3f p(0., 0., 0.);
						cv::Vec3f uv(0., 0., 0.);
						cv::Vec3i face = geoMesh.faceInds[fID];
						cv::Vec3i faceUV = uvMesh.faceInds[fID];
						cv::Vec3f fNorm = geoMesh.calcFaceNormal(fID);
						cv::Vec3f nw((1. - h.u - h.v), h.u, h.v);
						cv::Vec3f n0 = normalize(vertNorm[face[0]]);
						cv::Vec3f n1 = normalize(vertNorm[face[1]]);
						cv::Vec3f n2 = normalize(vertNorm[face[2]]);
						cv::Vec3f p0 = geoMesh.verts[face[0]];
						cv::Vec3f p1 = geoMesh.verts[face[1]];
						cv::Vec3f p2 = geoMesh.verts[face[2]];
						cv::Vec3f uv0 = uvMesh.verts[faceUV[0]];
						cv::Vec3f uv1 = uvMesh.verts[faceUV[1]];
						cv::Vec3f uv2 = uvMesh.verts[faceUV[2]];

						N = (1. - h.u - h.v) * n0 + h.u * n1 + h.v * n2;
						p = (1. - h.u - h.v) * p0 + h.u * p1 + h.v * p2;
						uv = (1. - h.u - h.v) * uv0 + h.u * uv1 + h.v * uv2;

						cv::Vec3f T(0., 0., 0.), B(0., 0., 0.);
						calcTSpace(p, p1, p2,
							cv::Vec2f(uv[0] / float(W), uv[1] / float(H)),
							cv::Vec2f(uv1[0] / float(W), uv1[1] / float(H)),
							cv::Vec2f(uv2[0] / float(W), uv2[1] / float(H)), T);

						T = normalize(T - T.dot(N) * N);
						B = N.cross(T);

						cv::Vec3f normV = normMap.at<cv::Vec3f>(y, x) / 255.;
						normV = cv::Vec3f(normV[2], normV[1], normV[0]) * 2. - cv::Vec3f(1., 1., 1.);
						cv::Vec3f N0 = normV[0] * T + normV[1] * B + normV[2] * N;

						cv::Vec3f color = (N0 + cv::Vec3f(1., 1., 1.)) * 0.5;
						baseNormalMap.at<cv::Vec3f>(y, x) = cv::Vec3f(color[2], color[1], color[0]);
						baseMask.at<int>(y, x) = 255;
					}
				} // end for if(mask)
			} // end for x
		} // end for y

		char _buffer[8];
		std::snprintf(_buffer, sizeof(_buffer), "%07d", f);
		cv::imwrite(casePrefR + "/t_30_L/" + std::string(_buffer) + ".png", baseNormalMap * 255);
		//cv::imwrite(casePrefR + "/uv/Mask.png", baseMask);
		//exit(1);
		//----------------------------------------------------
		// PD30: refLen = 0.0245
		// PD10: redLen = 0.0083
		//----------------------------------------------------

		/*float geoAveL = geoMesh.calcAverageEdegeLen(FaceF);
		float uvAveL = uvMesh.calcAverageEdegeLen(FaceF) / float(H);
		printf("geoALen: %f, uvALen: %f\n", geoAveL, uvAveL);
		int rsH = int(float(TXT_ALPHA) * geoAveL / uvAveL + 0.5);
		printf("rsH: %d\n", rsH);*/
	} // end for f
}

void ConvertWorldS_To_TanS(std::string pref_Root, int BS)
{
	std::string caseName = "/case_1/complex_evenCoarse/";
	std::string casePrefR = pref_Root + caseName;
	int frame0 = 1, frame1 = 200;

	/*std::string bImgName = casePrefR + "/textures/constN.jpg";
	cv::Mat bImg = cv::imread(bImgName, cv::IMREAD_COLOR);
	bImg.convertTo(bImg, CV_32FC3);*/

	std::string maskName = casePrefR + "/Mask.png";
	cv::Mat Mask = cv::imread(maskName, cv::IMREAD_GRAYSCALE);
	Mask.convertTo(Mask, CV_32SC1);

	int H = Mask.rows, W = Mask.cols;
	R_Mesh uvMesh;
	readPly(casePrefR + "uv/uv.ply", uvMesh.verts, uvMesh.faceInds);
	uvMesh.OrthProjMesh(H, W);
	RayIntersection myTracer;
	myTracer.addObj(&uvMesh);

	for (int fID = frame0; fID < frame1+1; fID++)
	{
		char _buffer[8];
		std::snprintf(_buffer, sizeof(_buffer), "%07d", fID);
		std::string wnormMapName = casePrefR + "/img/" + std::string(_buffer) + ".png";
		cv::Mat W_NMap = cv::imread(wnormMapName, cv::IMREAD_COLOR);
		W_NMap.convertTo(W_NMap, CV_32FC3);

		/*R_Mesh geoMesh;
		char _buffer6[8];
		std::snprintf(_buffer6, sizeof(_buffer6), "%06d", fID);
		readPly(casePrefR + "Meshes/" + std::string(_buffer6) + ".ply", geoMesh.verts, geoMesh.faceInds);*/

		R_Mesh geoMesh;
		geoMesh.faceInds = uvMesh.faceInds;
		readObjVertArray(casePrefR + "40_L/PD40_" + std::string(_buffer) + ".obj", geoMesh.verts);
		
		std::vector<cv::Vec3f> vertNorm = geoMesh.calcVertNorm();

		cv::Mat rstImg = cv::Mat::zeros(H, W, CV_32FC3);
		for (int y = 0; y < H; y++)
		{
			for (int x = 0; x < W; x++)
			{
				//rstImg.at<cv::Vec3f>(y, x) = bImg.at<cv::Vec3f>(y, x);
				if (Mask.at<int>(y, x) > 100)
				{
					cv::Vec3f ori(x, H - y, -10.);
					cv::Vec3f dir(0., 0., 1.);
					RTCHit h = myTracer.rayIntersection(ori, dir);
					int fID = h.primID;
					if (fID < 0)
						continue;
					else
					{
						cv::Vec3f N(0., 0., 0.);
						cv::Vec3f p(0., 0., 0.);
						cv::Vec3f uv(0., 0., 0.);
						cv::Vec3i face = geoMesh.faceInds[fID];
						cv::Vec3i faceUV = uvMesh.faceInds[fID];
						cv::Vec3f fNorm = geoMesh.calcFaceNormal(fID);
						cv::Vec3f nw((1. - h.u - h.v), h.u, h.v);
						cv::Vec3f n0 = normalize(vertNorm[face[0]]);
						cv::Vec3f n1 = normalize(vertNorm[face[1]]);
						cv::Vec3f n2 = normalize(vertNorm[face[2]]);
						cv::Vec3f p0 = geoMesh.verts[face[0]];
						cv::Vec3f p1 = geoMesh.verts[face[1]];
						cv::Vec3f p2 = geoMesh.verts[face[2]];
						cv::Vec3f uv0 = uvMesh.verts[faceUV[0]];
						cv::Vec3f uv1 = uvMesh.verts[faceUV[1]];
						cv::Vec3f uv2 = uvMesh.verts[faceUV[2]];

						N = (1. - h.u - h.v) * n0 + h.u * n1 + h.v * n2;
						p = (1. - h.u - h.v) * p0 + h.u * p1 + h.v * p2;
						uv = (1. - h.u - h.v) * uv0 + h.u * uv1 + h.v * uv2;

						cv::Vec3f T(0., 0., 0.), B(0., 0., 0.);
						calcTSpace(p, p1, p2,
							cv::Vec2f(uv[0] / float(W), uv[1] / float(H)),
							cv::Vec2f(uv1[0] / float(W), uv1[1] / float(H)),
							cv::Vec2f(uv2[0] / float(W), uv2[1] / float(H)), T);

						T = normalize(T - T.dot(N) * N);
						B = N.cross(T);

						cv::Vec3f DNorm = W_NMap.at<cv::Vec3f>(y, x) / 255.;
						DNorm = cv::Vec3f(DNorm[2], DNorm[1], DNorm[0]) * 2. - cv::Vec3f(1., 1., 1.);
						DNorm = normalize(DNorm);
						cv::Vec3f SDNorm = cv::Vec3f(DNorm.dot(T), DNorm.dot(B), DNorm.dot(N));
						cv::Vec3f Color = (SDNorm + cv::Vec3f(1., 1., 1.)) * 0.5;
						rstImg.at<cv::Vec3f>(y, x) = cv::Vec3f(Color[2], Color[1], Color[0]) * 255;
					}
				} // end if mask
			} // end for xi
		} // end for yi

		cv::imwrite(casePrefR + "/DE/" + std::string(_buffer) + ".png", rstImg);
	} // end for fID
}

void reTxtnormalMap(std::string pref_Root, int BS)
{
	std::string refName = pref_Root + "/textures/CNorm_1.png";
	cv::Mat refNMap = cv::imread(refName, cv::IMREAD_COLOR);
	refNMap.convertTo(refNMap, CV_32FC3);
	std::string maskName = pref_Root + "/textures/CMask_1.png";
	cv::Mat refMask = cv::imread(maskName, cv::IMREAD_GRAYSCALE);
	refMask.convertTo(refMask, CV_32SC1);
	refNMap = blurGapTexture(refNMap, refMask, 10);
	cv::imwrite(pref_Root + "/t.png", refNMap);

	int refH = refNMap.rows, refW = refNMap.cols;
	std::string uvMName = pref_Root + "/uv/simuv_30.ply";
	R_Mesh uvMesh;
	readPly(uvMName, uvMesh.verts, uvMesh.faceInds);
	std::vector<cv::Vec3f> vertNorm(uvMesh.numV(), cv::Vec3f(0., 0., 0.));
	std::vector<int> vFlag(uvMesh.numV(), 0);
	for (int vi = 0; vi < uvMesh.numV(); vi++)
	{
		cv::Vec3f p = uvMesh.verts[vi];
		cv::Vec2f tPos = cv::Vec2f(p[0] * refW, refH - p[1] * refH);

		cv::Vec3f normV = getInfoFromMat_3f(tPos, refNMap, refMask) / 255.;
		normV= cv::Vec3f(normV[2], normV[1], normV[0]) * 2. - cv::Vec3f(1., 1., 1.);
		vertNorm[vi] = normalize(normV);
	} //end for vi

	std::string scalFName = pref_Root + "uv/txt_0.txt";
	int txtScale = 0;
	ifstream saleStream(scalFName);
	string line;
	getline(saleStream, line);
	stringstream ss;
	ss << line;
	ss >> txtScale;
	printf("ImgSize: %d\n", txtScale);

	int H = txtScale, W = txtScale;
	uvMesh.OrthProjMesh(H, W);
	RayIntersection myTracer;
	myTracer.addObj(&uvMesh);

	cv::Mat rstNMap = cv::Mat::zeros(H, W, CV_32FC3);
	cv::Mat rstMask = cv::imread(pref_Root + "/textures/CMask_562.png", cv::IMREAD_GRAYSCALE);
	cv::Mat rstNNMask = cv::Mat::zeros(H, W, CV_32SC1);
	for (int y = 0; y < H; y++)
	{
		for (int x = 0; x < W; x++)
		{
			if (!(rstMask.at<uchar>(y, x) > 0))
				continue;

			cv::Vec3f ori(x, H - y, -10.);
			cv::Vec3f dir(0., 0., 1.);
			RTCHit h = myTracer.rayIntersection(ori, dir);
			int fID = h.primID;
			if (fID < 0)
				continue;
			else
			{
				cv::Vec3i face = uvMesh.faceInds[fID];
				cv::Vec3f n0 = normalize(vertNorm[face[0]]);
				cv::Vec3f n1 = normalize(vertNorm[face[1]]);
				cv::Vec3f n2 = normalize(vertNorm[face[2]]);
				cv::Vec3f N = (1. - h.u - h.v) * n0 + h.u * n1 + h.v * n2;
				cv::Vec3f color = (N + cv::Vec3f(1., 1., 1.)) * 0.5;
				rstNMap.at<cv::Vec3f>(y, x) = cv::Vec3f(color[2], color[1], color[0]);
				rstNNMask.at<int>(y, x) = 255;
			}
		} //end for x
	} // end for y
	cv::imwrite(pref_Root + "/t_30.png", rstNMap * 255.);
	cv::imwrite(pref_Root + "/30_mask.png", rstNNMask);
}

int main()
{
	//----------------------------------------------------
	//calcTexSize();
	//renderOrigTex();
	//DownSmapleBaseTxt(true, true); // PD10--> PD10_ds: Geo from 10_L to 10_Ds_L, txt grab from 10_L to t_Ds10_L
	//calcMeshNormals(); // compute normals from geo[geo-->txt]

	//GenerateJJMask();

	//GenVolFromLowDriver();
	//UpSampleBaseTxt(); // PD10_ds/PD30 --> PD10_DsUs/PD_30_Us: Geo from 10_Ds_L to 10_DsUs_C, or from 30_L to 30_Us_C, or from 30v_L to 30v_Us_C
	GrabNormals(); // Grab normal from texture for PD10_DsUs_C/PD30_Us_C
	//calcDist_NormalMap();

	//----------------------------------------------------
	// For mixamo skin texture
	//----------------------------------------------------
	//std::string pref_Root = "D:/models/MD/DataModel/DressOri/";
	//int BS = 2048;
	//CMaskToGMask(pref_Root);
	//ConvertTanS_To_WorldS(pref_Root, BS);
	//ConvertWorldS_To_TanS(pref_Root, BS);
	//reTxtnormalMap(pref_Root, BS);
	printf("Done.\n");
	
	while (1);
}