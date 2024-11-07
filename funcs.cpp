#include <stdio.h>
#include <vector>
#include <time.h>
#include <algorithm>
#include <pcl/point_types.h>
#include <pcl/registration/transforms.h>
#define BOOST_TYPEOF_EMULATION
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/shot.h>
#include <pcl/registration/transformation_estimation_svd.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/surface/gp3.h>
#include <pcl/surface/mls.h>
#include <pcl/visualization/pcl_visualizer.h>
#include "Eva.h"
/*******************************************************************************dataload********************************************************/
int XYZorPly_Read(string Filename, PointCloudPtr& cloud)
{
	int i;
	int nXYZ_nums;
	vector<Vertex> vXYZ;
	FILE* fp = fopen(Filename.c_str(), "r");
	if (fp == NULL)
	{
		printf("File can't open!\n");
		return -1;
	}
	const char* FILEPATH = Filename.c_str();
	char a = FILEPATH[strlen(FILEPATH) - 1];
	//
	if (a == 'y')
	{
		char str[1024];
		fscanf(fp, "%s\n", &str);
		fscanf(fp, "%s %s %s\n", &str, &str, &str);
		fscanf(fp, "%s %s %d\n", &str, &str, &nXYZ_nums);
		fscanf(fp, "%s %s %s\n", &str, &str, &str);
		fscanf(fp, "%s %s %s\n", &str, &str, &str);
		fscanf(fp, "%s %s %s\n", &str, &str, &str);
		fscanf(fp, "%s %s %s\n", &str, &str, &str);
		fscanf(fp, "%s %s %s %s %s\n", &str, &str, &str, &str, &str);
		fscanf(fp, "%s\n", &str);
	}
	else
	{
		fscanf(fp, "%d\n", &nXYZ_nums);
	}
	vXYZ.resize(nXYZ_nums);
	for (i = 0; i < vXYZ.size(); i++)
	{
		fscanf(fp, "%f %f %f\n", &vXYZ[i].x, &vXYZ[i].y, &vXYZ[i].z);
	}
	fclose(fp);
	cloud->width = vXYZ.size();
	cloud->height = 1;
	cloud->is_dense = true;
	cloud->points.resize(cloud->width * cloud->height);
	for (i = 0; i < cloud->points.size(); i++)
	{
		cloud->points[i].x = vXYZ[i].x;
		cloud->points[i].y = vXYZ[i].y;
		cloud->points[i].z = vXYZ[i].z;
	}
	return 0;
}

float MeshResolution_mr_compute(PointCloudPtr& cloud)
{
	int i;
	//计算点云分辨率
	float mr = 0;
	pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
	vector<int>pointIdx;
	vector<float>pointDst;
	kdtree.setInputCloud(cloud);
	pcl::PointXYZ query_point;
	for (i = 0; i < cloud->points.size(); i++)
	{
		query_point = cloud->points[i];
		kdtree.nearestKSearch(query_point, 2, pointIdx, pointDst);
		float x = cloud->points[pointIdx[0]].x - cloud->points[pointIdx[1]].x;
		float y = cloud->points[pointIdx[0]].y - cloud->points[pointIdx[1]].y;
		float z = cloud->points[pointIdx[0]].z - cloud->points[pointIdx[1]].z;
		float mr_temp = sqrt(x * x + y * y + z * z);
		mr += mr_temp;
	}
	mr /= cloud->points.size();
	return mr;//approximate calculation
}

int Voxel_grid_downsample(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr& new_cloud,
                      float leaf_size) {
    pcl::VoxelGrid<pcl::PointXYZ> sor;
    sor.setInputCloud(cloud);
    sor.setLeafSize(leaf_size, leaf_size, leaf_size);
    sor.filter(*new_cloud);
    return 0;
}
/*******************************************************************************Feature match********************************************************/
void feature_matching(PointCloudPtr& cloud_source, PointCloudPtr& cloud_target,
                      vector<vector<float>>& feature_source, vector<vector<float>>& feature_target, vector<Corre_3DMatch>& Corres)
{
    int i, j;
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr Feature_source(new pcl::PointCloud<pcl::FPFHSignature33>);
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr Feature_target(new pcl::PointCloud<pcl::FPFHSignature33>);
    Feature_source->points.resize(feature_source.size());
    Feature_target->points.resize(feature_target.size());
    for (i = 0; i < feature_source.size(); i++)
    {
        for (j = 0; j < 33; j++)
        {
            if (j < feature_source[i].size()) Feature_source->points[i].histogram[j] = feature_source[i][j];
            else Feature_source->points[i].histogram[j] = 0;
        }
    }
    for (i = 0; i < feature_target.size(); i++)
    {
        for (j = 0; j < 33; j++)
        {
            if (j < feature_target[i].size()) Feature_target->points[i].histogram[j] = feature_target[i][j];
            else Feature_target->points[i].histogram[j] = 0;
        }
    }
    //
    pcl::KdTreeFLANN<pcl::FPFHSignature33> kdtree;
    vector<int>Idx;
    vector<float>Dist;
    kdtree.setInputCloud(Feature_target);
    for (i = 0; i < Feature_source->points.size(); i++)
    {
        kdtree.nearestKSearch(Feature_source->points[i], 1, Idx, Dist);
        Corre_3DMatch temp;
        temp.src_index = i;
        temp.des_index = Idx[0];
        temp.src = cloud_source->points[i];
        temp.des = cloud_target->points[Idx[0]];
        temp.score = 1 - sqrt(Dist[0]);
        Corres.push_back(temp);
    }
}

int Correct_corre_compute(PointCloudPtr &cloud_s, PointCloudPtr &cloud_t, vector<Corre_3DMatch > &Corres, float correct_thresh, Eigen::Matrix4d& GT_mat, string path)
{
	if (Corres.size() == 0) return 0;
	int i;
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_s_trans(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::transformPointCloud(*cloud_s, *cloud_s_trans, GT_mat);
	string TC_path = path + "/true_corre.txt";
	FILE* fp = fopen(TC_path.c_str(), "w");
	int Corret_num = 0;
	for (i = 0; i < Corres.size(); i++)
	{
		int Idx_s = Corres[i].src_index;
		int Idx_t = Corres[i].des_index;
		float dist = Distance(cloud_s_trans->points[Idx_s], cloud_t->points[Idx_t]);
		if (dist <= correct_thresh) {
			fprintf(fp, "1\n");
			Corret_num++;
		}
		else {
			fprintf(fp, "0\n");
		}
	}
	fclose(fp);
	return Corret_num;
}
void Correct_corre_select(PointCloudPtr cloud_s, PointCloudPtr cloud_t, vector<Corre> Corres, float correct_thresh,
	Eigen::Matrix4f& GT_mat, vector<Corre>& Corres_selected)
{
	int i;
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_s_trans(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::transformPointCloud(*cloud_s, *cloud_s_trans, GT_mat);
	//
	int Corret_num = 0;
	for (i = 0; i < Corres.size(); i++)
	{
		int Idx_s = Corres[i].source_idx;
		int Idx_t = Corres[i].target_idx;
		float dist = pow(cloud_s_trans->points[Idx_s].x - cloud_t->points[Idx_t].x, 2) + pow(cloud_s_trans->points[Idx_s].y - cloud_t->points[Idx_t].y, 2)
			+ pow(cloud_s_trans->points[Idx_s].z - cloud_t->points[Idx_t].z, 2);
		dist = sqrt(dist);
		if (dist <= correct_thresh)
			Corres_selected.push_back(Corres[i]);
	}
}

float OTSU_thresh(vector<float> all_scores)
{
	int i;
	int Quant_num = 100;
	float score_sum = 0.0;
	float fore_score_sum = 0.0;
	vector<int> score_Hist(Quant_num, 0);
	vector<float> score_sum_Hist(Quant_num, 0.0);
	float max_score_value, min_score_value;
	for (i = 0; i < all_scores.size(); i++)
	{
		score_sum += all_scores[i];
	}
	sort(all_scores.begin(), all_scores.end());
	max_score_value = all_scores[all_scores.size() - 1];
	min_score_value = all_scores[0];
	float Quant_step = (max_score_value - min_score_value) / Quant_num;
	for (i = 0; i < all_scores.size(); i++)
	{
		int ID = all_scores[i] / Quant_step;
		if (ID >= Quant_num) ID = Quant_num - 1;
		score_Hist[ID]++;
		score_sum_Hist[ID] += all_scores[i];
	}
	float fmax = -1000;
	int n1 = 0, n2;
	float m1, m2, sb;
	float thresh = (max_score_value - min_score_value) / 2;//default value
	for (i = 0; i < Quant_num; i++)
	{
		float Thresh_temp = i * (max_score_value - min_score_value) / float (Quant_num);
		n1 += score_Hist[i];
		if (n1 == 0) continue;
		n2 = all_scores.size() - n1;
		if (n2 == 0) break;
		fore_score_sum += score_sum_Hist[i];
		m1 = fore_score_sum / n1;
		m2 = (score_sum - fore_score_sum) / n2;
		sb = (float )n1 * (float )n2 * pow(m1 - m2, 2);
		if (sb > fmax)
		{
			fmax = sb;
			thresh = Thresh_temp;
		}
	}
	return thresh;
}

//
float Distance(pcl::PointXYZ& A, pcl::PointXYZ& B) {
	float distance = 0;
	float d_x = A.x - B.x;
	float d_y = A.y - B.y;
	float d_z = A.z - B.z;
	distance = sqrt(d_x * d_x + d_y * d_y + d_z * d_z);
    if(!isfinite(distance)){
        cout << distance <<"\t" <<A.x << " " <<A.y << " " << A.z << "\t" << B.x << " " <<B.y << " " << B.z <<endl;
    }
	return distance;
}

float RMSE_compute_scene(PointCloudPtr &cloud_source, PointCloudPtr &cloud_target, Eigen::Matrix4f& Mat_est, Eigen::Matrix4f& Mat_GT, float overlap_thresh)
{
    float RMSE_temp = 0.0f;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_source_trans_GT(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::transformPointCloud(*cloud_source, *cloud_source_trans_GT, Mat_GT);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_source_trans_EST(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::transformPointCloud(*cloud_source, *cloud_source_trans_EST, Mat_est);
    vector<int>overlap_idx;
    overlap_thresh = 0.0375;
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree1;
    pcl::PointXYZ query_point;
    vector<int>pointIdx;
    vector<float>pointDst;
    kdtree1.setInputCloud(cloud_target);
    for (int i = 0; i < cloud_source_trans_GT->points.size(); i++)
    {
        query_point = cloud_source_trans_GT->points[i];
        kdtree1.nearestKSearch(query_point, 1, pointIdx, pointDst);
        if (sqrt(pointDst[0]) <= overlap_thresh)
            overlap_idx.push_back(i);
    }
    //
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree2;
    kdtree2.setInputCloud(cloud_source_trans_GT);
    for (int i = 0; i < overlap_idx.size(); i++)
    {
        //query_point = cloud_source_trans_EST->points[overlap_idx[i]];
        //kdtree2.nearestKSearch(query_point,1,pointIdx,pointDst); RMSE_temp+=sqrt(pointDst[0]);
        float dist_x = pow(cloud_source_trans_EST->points[overlap_idx[i]].x - cloud_source_trans_GT->points[overlap_idx[i]].x, 2);
        float dist_y = pow(cloud_source_trans_EST->points[overlap_idx[i]].y - cloud_source_trans_GT->points[overlap_idx[i]].y, 2);
        float dist_z = pow(cloud_source_trans_EST->points[overlap_idx[i]].z - cloud_source_trans_GT->points[overlap_idx[i]].z, 2);
        float dist = dist_x + dist_y + dist_z;
        RMSE_temp += dist;
    }
    RMSE_temp /= overlap_idx.size();
    return sqrt(RMSE_temp);
}