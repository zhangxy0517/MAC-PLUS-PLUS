#include <stdio.h>
#include <memory>
#include <vector>
#include <time.h>
#include <algorithm>
#include <pcl/point_types.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/registration/transforms.h>
#define BOOST_TYPEOF_EMULATION
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/shot_omp.h>
#include <pcl/features/shot_lrf.h>
#include <pcl/keypoints/harris_3d.h>
#include <pcl/surface/mls.h>
#include <pcl/common/centroid.h>
#include <pcl/common/eigen.h>
#include <pcl/visualization/pcl_visualizer.h>
#include<pcl/keypoints/iss_3d.h>
#include "Eva.h"
/*******************************************************************************Descriptor********************************************************/
void SHOT_compute(PointCloudPtr &cloud, vector<int> &indices, float sup_radius, vector<vector<float>>&features)
{
	int i, j;
	pcl::PointIndicesPtr Idx(new pcl::PointIndices);
	Idx->indices = indices;
	pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> n;
	pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
	tree->setInputCloud(cloud);
	n.setInputCloud(cloud);
	n.setSearchMethod(tree);
	n.setKSearch(20);
	n.compute(*normals);

	//SHOT
	pcl::SHOTEstimationOMP<pcl::PointXYZ, pcl::Normal, pcl::SHOT352> shot_est;
	shot_est.setInputCloud(cloud);
	shot_est.setInputNormals(normals);
	pcl::PointCloud<pcl::SHOT352>::Ptr shots(new pcl::PointCloud<pcl::SHOT352>());
	pcl::search::KdTree<pcl::PointXYZ>::Ptr treeI(new pcl::search::KdTree<pcl::PointXYZ>);
	treeI->setInputCloud(cloud);
	shot_est.setSearchMethod(tree);
	shot_est.setIndices(Idx);
	shot_est.setRadiusSearch(sup_radius);
	shot_est.compute(*shots);

	features.resize(shots->points.size());
	for (i = 0; i<features.size(); i++)
	{
		features[i].resize(352);
		for (j = 0; j<352; j++)
		{
			features[i][j] = shots->points[i].descriptor[j];
		}
	}
}

pcl::PointCloud<pcl::PointXYZ>::Ptr getHarris3D_detector(PointCloudPtr cloud, float NMS_radius, vector<int>&key_indices)
{
	key_indices.clear();
	pcl::HarrisKeypoint3D<pcl::PointXYZ, pcl::PointXYZI> detector;
	detector.setNonMaxSupression(true);
	detector.setRadius(NMS_radius);
	detector.setInputCloud(cloud);
	detector.setRefine(false);
	pcl::PointCloud<pcl::PointXYZI>::Ptr keypoints(new pcl::PointCloud<pcl::PointXYZI>());
	detector.compute(*keypoints);
	//
	pcl::PointCloud<pcl::PointXYZ>::Ptr _keypoints(new pcl::PointCloud<pcl::PointXYZ>());
	for (int i = 0; i<keypoints->points.size(); i++)
	{
		pcl::PointXYZ p;
		p.x = keypoints->points[i].x;
		p.y = keypoints->points[i].y;
		p.z = keypoints->points[i].z;
		_keypoints->points.push_back(p);
	}
	pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
	vector<int> Idx;
	vector<float> Dist;
	kdtree.setInputCloud(cloud);
	for (int i = 0; i<_keypoints->size(); i++)
	{
		kdtree.nearestKSearch(_keypoints->points[i], 1, Idx, Dist);
		key_indices.push_back(Idx[0]);
	}
	return _keypoints;
}

void FPFH_descriptor(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, float sup_radius, std::vector<std::vector<float>>& features){
    int i, j;
    pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> n;
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(cloud);
    n.setInputCloud(cloud);
    n.setSearchMethod(tree);
    n.setRadiusSearch(sup_radius / 4);
    n.compute(*normals);
    // Create the FPFH estimation class, and pass the input dataset+normals to it
    pcl::FPFHEstimationOMP<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> fpfh;
    fpfh.setInputCloud (cloud);
    fpfh.setInputNormals (normals);
    // alternatively, if cloud is of tpe PointNormal, do fpfh.setInputNormals (cloud);

    // Create an empty kdtree representation, and pass it to the FPFH estimation object.
    // Its content will be filled inside the object, based on the given input dataset (as no other search surface is given).
    fpfh.setSearchMethod (tree);
    // Output datasets
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfhs (new pcl::PointCloud<pcl::FPFHSignature33> ());

    // Use all neighbors in a sphere of radius 5cm
    // IMPORTANT: the radius used here has to be larger than the radius used to estimate the surface normals!!!
    fpfh.setRadiusSearch (sup_radius);

    // Compute the features
    fpfh.compute (*fpfhs);
    features.resize(fpfhs->points.size());
    for(i = 0; i < features.size(); i++){
        features[i].resize(33);
        for(j = 0; j < 33; j++){
            features[i][j] = fpfhs->points[i].histogram[j];
        }
    }
}