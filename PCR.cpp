//MKL boost
//#define EIGEN_USE_MKL_ALL
//#define EIGEN_VACTORIZE_SSE4_2
#include <cstdio>
#include <vector>
#include <time.h>
#include <algorithm>
#include <pcl/point_types.h>
#include <pcl/registration/transforms.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/registration/transformation_estimation_svd.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/filter.h>
#include <pcl/common/centroid.h>
#include <pcl/common/eigen.h>
#include <pcl/io/pcd_io.h>
#include <cmath>
#include "Eva.h"
#include "omp.h"
//#include<mkl.h>
#include <unsupported/Eigen/MatrixFunctions>
extern bool no_logs;
using namespace Eigen;

bool compare_vote_score(const Vote& v1, const Vote& v2) {
	return v1.score > v2.score;
}

bool compare_local_score(const local &l1, const local &l2){
    return l1.score > l2.score;
}

bool compare_corres_ind(const Corre_3DMatch& c1, const Corre_3DMatch& c2){
    return c1.des_index < c2.des_index;
}


Eigen::MatrixXf Graph_construction(vector<Corre_3DMatch>& correspondence, float resolution, bool sc2, const string &name, const string &descriptor, float inlier_thresh) {
	int size = correspondence.size();
	Eigen::MatrixXf cmp_score = Eigen::MatrixXf::Zero(size, size);
	Corre_3DMatch c1, c2;
	float score = 0, src_dis = 0, des_dis = 0, dis = 0, alpha_dis = 10 * resolution;;
	if (name == "KITTI")
	{
        float thresh = descriptor == "fpfh" ? 0.9 : 0.999;
		for (int i = 0; i < size; i++)
		{
			c1 = correspondence[i];
			for (int j = i + 1; j < size; j++)
			{
				c2 = correspondence[j];
				src_dis = Distance(c1.src, c2.src);
				des_dis = Distance(c1.des, c2.des); //(c1, c2) (c3, c2) ?
				dis = abs(src_dis - des_dis);
				score = 1.0 - (dis * dis) / (inlier_thresh * inlier_thresh); //10-20 1.2, 20-30 1.8 ?
				//score = exp(-dis * dis);
				score = (score < thresh) ? 0 : score;//fcgf 0.9999 fpfh 0.9(10) 0.85(20) 0.8(30)
				cmp_score(i, j) = score;
				cmp_score(j, i) = score;

			}
		}
	}
	else if (name == "3dmatch" || name == "3dlomatch")
	{
		for (int i = 0; i < size; i++)
		{
			c1 = correspondence[i];
			for (int j = i + 1; j < size; j++)
			{
				c2 = correspondence[j];
				src_dis = Distance(c1.src, c2.src);
				des_dis = Distance(c1.des, c2.des);
				dis = abs(src_dis - des_dis);
                //if(c1.des_index != c2.des_index && src_dis > 0.05 && des_dis > 0.05){
                    if (descriptor == "predator" || low_inlieratio)
                    {
                        score = 1.0 - (dis * dis) / (inlier_thresh * inlier_thresh);
                        if (add_overlap || low_inlieratio) // 0228
                        {
                            score = (score < 0.99) ? 0 : score; //fpfh/fcgf overlap 0.99
// mac-op 250 500 1000 2500 5000
//        0.9 0.95 0.99 0.995 0.999
                        }
                        else {
                            score = (score < 0.999) ? 0 : score;
                        }
                    }
                    else {
                        
                        score = exp(-dis * dis / (2 * alpha_dis * alpha_dis));
                        if (name == "3dmatch" && descriptor == "fcgf")
                        {
                            score = (score < 0.995) ? 0 : score; //0.995
                        }
                        else if (name == "3dmatch" && descriptor == "fpfh") {
                            score = (score < 0.99) ? 0 : score;
                        }
                        else if (descriptor == "spinnet" || descriptor == "d3feat") {
                            score = (score < 0.85) ? 0 : score;
                            // spinnet 5000 2500 1000 500 250
                            //         0.99 0.99 0.95 0.9 0.85
                        }
                        else {
                            score = (score < 0.99) ? 0 : score; //3dlomatch 0.99, 3dmatch fcgf 0.999 fpfh 0.995
                            // >=1000 0.99 <1000 0.9
                        }
                    }
               // }
				cmp_score(i, j) = score;
				cmp_score(j, i) = score;
			}
		}
	}
	else if (name == "U3M") {
		for (int i = 0; i < size; i++)
		{
			c1 = correspondence[i];
			for (int j = i + 1; j < size; j++)
			{
				c2 = correspondence[j];
				//计算兼容性分数
				src_dis = Distance(c1.src, c2.src);
				des_dis = Distance(c1.des, c2.des);
				dis = abs(src_dis - des_dis);
				alpha_dis = 10 * resolution;
				score = exp(-dis * dis / (2 * alpha_dis * alpha_dis));
				score = (score < 0.95) ? 0 : score;
				cmp_score(i, j) = score;
				cmp_score(j, i) = score;
			}
		}
	}
	if (sc2)
	{
		//Eigen::setNbThreads(6);
		cmp_score = cmp_score.cwiseProduct(cmp_score * cmp_score);
	}
	return cmp_score;
}

vector<int> vectors_intersection(const vector<int>& v1, const vector<int>& v2) {
	vector<int> v;
	set_intersection(v1.begin(), v1.end(), v2.begin(), v2.end(), back_inserter(v));
	return v;
}

vector<int> vectors_union(const vector<int>& v1, const vector<int>& v2){
    vector<int> v;
    set_union(v1.begin(), v1.end(), v2.begin(), v2.end(), back_inserter(v));
    return v;
}

float calculate_rotation_error(Eigen::Matrix3f& est, Eigen::Matrix3f& gt) {
	float tr = (est.transpose() * gt).trace();
	return acos(min(max((tr - 1.0) / 2.0, -1.0), 1.0)) * 180.0 / M_PI;
}

float calculate_translation_error(Eigen::Vector3f& est, Eigen::Vector3f& gt) {
	Eigen::Vector3f t = est - gt;
	return sqrt(t.dot(t)) * 100;
}

bool evaluation_est(Eigen::Matrix4f &est, Eigen::Matrix4f &gt, float re_thresh, float te_thresh, float& RE, float& TE) {
	Eigen::Matrix3f rotation_est, rotation_gt;
	Eigen::Vector3f translation_est, translation_gt;
	rotation_est = est.topLeftCorner(3, 3);
	rotation_gt = gt.topLeftCorner(3, 3);
	translation_est = est.block(0, 3, 3, 1);
	translation_gt = gt.block(0, 3, 3, 1);

	RE = calculate_rotation_error(rotation_est, rotation_gt);
	TE = calculate_translation_error(translation_est, translation_gt);
	if (0 <= RE && RE <= re_thresh && 0 <= TE && TE <= te_thresh)
	{
		return true;
	}
	return false;
}

void weight_SVD(PointCloudPtr& src_pts, PointCloudPtr& des_pts, Eigen::VectorXf& weights, float weight_threshold, Eigen::Matrix4f& trans_Mat) {
	for (int i = 0; i < weights.size(); i++)
	{
		weights(i) = (weights(i) < weight_threshold) ? 0 : weights(i);
	}
	//weights升维度
	Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> weight;
	Eigen::VectorXf ones = weights;
	ones.setOnes();
	weight = (weights * ones.transpose());
	Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> Identity = weight;
	//构建对角阵
	Identity.setIdentity();
	weight = (weights * ones.transpose()).cwiseProduct(Identity);
	pcl::ConstCloudIterator<pcl::PointXYZ> src_it(*src_pts);
	pcl::ConstCloudIterator<pcl::PointXYZ> des_it(*des_pts);
	//获取点云质心
	src_it.reset(); des_it.reset();
	Eigen::Matrix<float, 4, 1> centroid_src, centroid_des;
	pcl::compute3DCentroid(src_it, centroid_src);
	pcl::compute3DCentroid(des_it, centroid_des);

	//去除点云质心
	src_it.reset(); des_it.reset();
	Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> src_demean, des_demean;
	pcl::demeanPointCloud(src_it, centroid_src, src_demean);
	pcl::demeanPointCloud(des_it, centroid_des, des_demean);

	//计算加权协方差矩阵
	Eigen::Matrix<float, 3, 3> H = (src_demean * weight * des_demean.transpose()).topLeftCorner(3, 3);
	//cout << H << endl;

	// Compute the Singular Value Decomposition
	Eigen::JacobiSVD<Eigen::Matrix<float, 3, 3> > svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
	Eigen::Matrix<float, 3, 3> u = svd.matrixU();
	Eigen::Matrix<float, 3, 3> v = svd.matrixV();

	// Compute R = V * U'
	if (u.determinant() * v.determinant() < 0)
	{
		for (int x = 0; x < 3; ++x)
			v(x, 2) *= -1;
	}

	Eigen::Matrix<float, 3, 3> R = v * u.transpose(); //正交矩阵的乘积还是正交矩阵，因此R的逆等于R的转置

	// Return the correct transformation
	Eigen::Matrix<float, 4, 4> Trans;
	Trans.setIdentity();
	Trans.topLeftCorner(3, 3) = R;
	const Eigen::Matrix<float, 3, 1> Rc(R * centroid_src.head(3));
	Trans.block(0, 3, 3, 1) = centroid_des.head(3) - Rc;
	trans_Mat = Trans;
}

void post_refinement(vector<Corre_3DMatch>&correspondence, PointCloudPtr& src_corr_pts, PointCloudPtr& des_corr_pts, Eigen::Matrix4f& initial/* 由最大团生成的变换 */, float& best_score, float inlier_thresh, int iterations, const string &metric) {
    int pointNum = src_corr_pts->points.size();
	float pre_score = best_score;
	for (int i = 0; i < iterations; i++)
	{
		float score = 0;
		Eigen::VectorXf weights, weight_pred;
		weights.resize(pointNum);
		weights.setZero();
		vector<int> pred_inlier_index;
		PointCloudPtr trans(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::transformPointCloud(*src_corr_pts, *trans, initial);
        //remove nan points
        trans->is_dense = false;
        vector<int>mapping;
        pcl::removeNaNFromPointCloud(*trans, *trans, mapping);
        if(!trans->size()) return;
		for (int j = 0; j < pointNum; j++)
		{
			float dist = Distance(trans->points[j], des_corr_pts->points[j]);
			float w = 1;
			if (add_overlap)
			{
				w = correspondence[j].score;
			}
			if (dist < inlier_thresh)
			{
				pred_inlier_index.push_back(j);
				weights[j] = 1 / (1 + pow(dist / inlier_thresh, 2));
				if (metric == "inlier")
				{
					score+=1*w;
				}
				else if (metric == "MAE")
				{
					score += (inlier_thresh - dist)*w / inlier_thresh;
				}
				else if (metric == "MSE")
				{
					score += pow((inlier_thresh - dist), 2)*w / pow(inlier_thresh, 2);
				}
			}
		}
		if (score < pre_score) {
			break;
		}
		else {
			pre_score = score;
			//估计pred_inlier
			PointCloudPtr pred_src_pts(new pcl::PointCloud<pcl::PointXYZ>);
			PointCloudPtr pred_des_pts(new pcl::PointCloud<pcl::PointXYZ>);
			pcl::copyPointCloud(*src_corr_pts, pred_inlier_index, *pred_src_pts);
			pcl::copyPointCloud(*des_corr_pts, pred_inlier_index, *pred_des_pts);
			weight_pred.resize(pred_inlier_index.size());
			for (int k = 0; k < pred_inlier_index.size(); k++)
			{
				weight_pred[k] = weights[pred_inlier_index[k]];
			}
			//weighted_svd
			weight_SVD(pred_src_pts, pred_des_pts, weight_pred, 0, initial);
			pred_src_pts.reset(new pcl::PointCloud<pcl::PointXYZ>);
			pred_des_pts.reset(new pcl::PointCloud<pcl::PointXYZ>);
		}
		pred_inlier_index.clear();
		trans.reset(new pcl::PointCloud<pcl::PointXYZ>);
	}
	best_score = pre_score;
}

float evaluation_trans(vector<Corre_3DMatch>& correspondnece, PointCloudPtr& src_corr_pts, PointCloudPtr& des_corr_pts, Eigen::Matrix4f& trans, float metric_thresh, const string &metric, float resolution) {
	PointCloudPtr src_trans(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::transformPointCloud(*src_corr_pts, *src_trans, trans);
    src_trans->is_dense = false;
    vector<int>mapping;
    pcl::removeNaNFromPointCloud(*src_trans, *src_trans, mapping);
    if(!src_trans->size()) return 0;
	float score = 0.0;
	int inlier = 0;
	int corr_num = src_corr_pts->points.size();
	for (int i = 0; i < corr_num; i++)
	{
		float dist = Distance(src_trans->points[i], des_corr_pts->points[i]);
		float w = 1;
		if (add_overlap)
		{
			w = correspondnece[i].score;
		}
		if (dist < metric_thresh)
		{
			inlier++;
			if (metric == "inlier")
			{
				score += 1*w;//correspondence[i].inlier_weight;
			}
			else if (metric == "MAE")
			{
				score += (metric_thresh - dist)*w / metric_thresh;
			}
			else if (metric == "MSE")
			{
				score += pow((metric_thresh - dist), 2)*w / pow(metric_thresh, 2);
			}
		}
	}
	src_trans.reset(new pcl::PointCloud<pcl::PointXYZ>);
	return score;
}

void find_clique_of_node2(Eigen::MatrixXf& Graph, igraph_vector_int_list_t* cliques, vector<Corre_3DMatch>& correspondence,vector<int>& sampled_ind, vector<int>&remain){
    remain.clear();
    sampled_ind.clear();
    int n = correspondence.size();
    int m = igraph_vector_int_list_size(cliques);
    unordered_set<int> vis;
    vector<local> result(n);
#pragma omp parallel for
    for(int i = 0; i < n; i++){
        result[i].corre_ind = i;
    }
    vector<Vote> c_w;
    for(int i = 0; i < m; i++){
        igraph_vector_int_t* v = igraph_vector_int_list_get_ptr(cliques, i);
        float weight = 0;
        int length = igraph_vector_int_size(v);

        for (int j = 0; j < length; j++)
        {
            int a = (int)VECTOR(*v)[j];
            for (int k = j + 1; k < length; k++)
            {
                int b = (int)VECTOR(*v)[k];
                weight += Graph(a, b);
            }
        }
        Vote t;
        t.index = i;
        t.score = weight;
        c_w.push_back(t);
        for (int j = 0; j < length; j++)
        {
            int k = (int)VECTOR(*v)[j];
            result[k].clique_ind_score.push_back(t);
        }
    }
    float avg_score = 0;
#pragma omp parallel for
    for(int i = 0; i < n; i++){
        result[i].score = 0;
        for(int j = 0; j < result[i].clique_ind_score.size(); j ++){
            result[i].score += result[i].clique_ind_score[j].score;
        }
#pragma omp critical
        {
            avg_score += result[i].score;
        }
    }
    sort(result.begin(), result.end(), compare_local_score); //所有节点从大到小排序

    if( m <= n ){
        for(int i = 0; i < m; i++){
            remain.push_back(i);
        }
        for(int i = 0; i < n; i++){
            if(!result[i].score){
                continue;
            }
            sampled_ind.push_back(result[i].corre_ind);
        }
        return;
    }

    avg_score /= n;
    int max_cnt = 10;  //default 10
    for(int i = 0; i < n; i++){
        if(result[i].score < avg_score) break;
        sort(result[i].clique_ind_score.begin(), result[i].clique_ind_score.end(), compare_vote_score); //局部从大到小排序
        sampled_ind.push_back(result[i].corre_ind);
        int seleted_cnt = 1;
        for(int j = 0; j < result[i].clique_ind_score.size(); j++){
            if(seleted_cnt > max_cnt) break;
            int ind = result[i].clique_ind_score[j].index;
            if(vis.find(ind) == vis.end()){
                vis.insert(ind);
            }
            else{
                continue;
            }
            seleted_cnt ++;
        }
    }
    remain.assign(vis.begin(), vis.end());
    return;
}

//保存数据,需要与寻找法向量部分组合
void savetxt(vector<Corre_3DMatch>corr, const string& save_path) {
	ofstream outFile;
	outFile.open(save_path.c_str(), ios::out);
	for (auto & i : corr)
	{
		outFile << i.src_index << " " << i.des_index << endl;
	}
	outFile.close();
}

//######################################################################################################################
float g_angleThreshold = 5.0 * M_PI / 180;//5 degree
float g_distanceThreshold = 0.1;

bool EnforceSimilarity1(const pcl::PointXYZINormal &point_a, const pcl::PointXYZINormal &point_b, float squared_distance){
    if(point_a.normal_x == 666 || point_b.normal_x == 666 || point_a.normal_y == 666 || point_b.normal_y == 666 || point_a.normal_z == 666 || point_b.normal_z == 666){
        return false;
    }
    Eigen::VectorXf temp(3);
    temp[0] = point_a.normal_x - point_b.normal_x;
    temp[1] = point_a.normal_y - point_b.normal_y;
    temp[2] = point_a.normal_z - point_b.normal_z;
    if(temp.norm() < g_distanceThreshold){
        return true;
    }
    return false;
}

bool checkEulerAngles(float angle){
    if(isfinite(angle) && angle >= -M_PIf32 && angle <= M_PIf32){
        return true;
    }
    return false;
}

int clusterTransformationByRotation(vector<Eigen::Matrix3f> &Rs, vector<Eigen::Vector3f> &Ts, float angle_thresh,float dis_thresh,  pcl::IndicesClusters &clusters, pcl::PointCloud<pcl::PointXYZINormal>::Ptr &trans){
    if(Rs.empty() || Ts.empty() || Rs.size() != Ts.size()){
        return -1;
    }
    int num = Rs.size();
    g_distanceThreshold = dis_thresh;
    trans->resize(num);
    for (size_t i = 0; i < num; i++) {
        Eigen::Transform<float, 3, Eigen::Affine> R(Rs[i]);
        pcl::getEulerAngles<float>(R, (*trans)[i].x, (*trans)[i].y, (*trans)[i].z);
        // 去除无效解
        if(!checkEulerAngles((*trans)[i].x) || !checkEulerAngles((*trans)[i].y) || !checkEulerAngles((*trans)[i].z)){
            cout << "INVALID POINT" << endl;
            (*trans)[i].x = 666;
            (*trans)[i].y = 666;
            (*trans)[i].z = 666;
            (*trans)[i].normal_x = 666;
            (*trans)[i].normal_y = 666;
            (*trans)[i].normal_z = 666;
        }
        else{ // 需要解决同一个角度的正负问题 6.14   平面 y=PI 右侧的解（需要验证） 6.20
            (*trans)[i].x = ((*trans)[i].x < 0 && (*trans)[i].x >= -M_PIf32) ? (*trans)[i].x + 2*M_PIf32 : (*trans)[i].x;
            (*trans)[i].y = ((*trans)[i].y < 0 && (*trans)[i].y >= -M_PIf32) ? (*trans)[i].y + 2*M_PIf32 : (*trans)[i].y;
            (*trans)[i].z = ((*trans)[i].z < 0 && (*trans)[i].z >= -M_PIf32) ? (*trans)[i].z + 2*M_PIf32 : (*trans)[i].z;
            (*trans)[i].normal_x = (float)Ts[i][0];
            (*trans)[i].normal_y = (float)Ts[i][1];
            (*trans)[i].normal_z = (float)Ts[i][2];
        }
    }

    pcl::ConditionalEuclideanClustering<pcl::PointXYZINormal> cec(true);
    cec.setInputCloud(trans);
    cec.setConditionFunction(&EnforceSimilarity1);
    cec.setClusterTolerance(angle_thresh);
    cec.setMinClusterSize(2);
    cec.setMaxClusterSize(static_cast<int>(num));
    cec.segment(clusters);
    for (int i = 0; i < clusters.size (); ++i)
    {
        for (int j = 0; j < clusters[i].indices.size (); ++j) {
            trans->points[clusters[i].indices[j]].intensity = i;
        }
    }
    return 0;
}

float OAMAE_1tok(PointCloudPtr& raw_src, PointCloudPtr& raw_des, Eigen::Matrix4f &est, vector<pair<int, vector<int>>> &src_des, float thresh){
    float score = 0.0;
    PointCloudPtr src_trans(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::transformPointCloud(*raw_src, *src_trans, est);
    for(auto & i : src_des){
        int src_ind = i.first;
        vector<int> des_ind = i.second;
        float num = 0.0;
        float dis = 0.0;
        if(!pcl::isFinite(src_trans->points[src_ind])) continue;
        for(auto & e : des_ind){
            //计算距离
            float distance = Distance(src_trans->points[src_ind], raw_des->points[e]);
            if (distance < thresh)
            {
                num++;
                dis += (thresh - distance) / thresh;
            }
        }
        score += num > 0 ? (dis / num) : 0;
    }
    src_trans.reset(new pcl::PointCloud<pcl::PointXYZ>);
    return score;
}

float OAMAE(PointCloudPtr& raw_src, PointCloudPtr& raw_des, Eigen::Matrix4f &est, vector<pair<int, vector<int>>> &des_src, float thresh){
    float score = 0.0;
    PointCloudPtr src_trans(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::transformPointCloud(*raw_src, *src_trans, est);
    for(auto & i : des_src){
        int des_ind = i.first;
        vector<int> src_ind = i.second;
        float num = 0.0;
        float dis = 0.0;
        for(auto & e : src_ind){
            if(!pcl::isFinite(src_trans->points[e])) continue;
            //计算距离
            float distance = Distance(src_trans->points[e], raw_des->points[des_ind]);
            if (distance < thresh)
            {
                num++;
                dis += (thresh - distance) / thresh;
            }
        }
        score += num > 0 ? (dis / num) : 0;
    }
    src_trans.reset(new pcl::PointCloud<pcl::PointXYZ>);
    return score;
}

void getCorrPatch(vector<Corre_3DMatch>&sampled_corr, PointCloudPtr &src, PointCloudPtr &des, PointCloudPtr &src_batch, PointCloudPtr &des_batch, float radius){
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree_src, kdtree_des;
    kdtree_src.setInputCloud(src);
    kdtree_des.setInputCloud(des);
    vector<int>src_ind, des_ind;
    vector<float>src_dis, des_dis;
    vector<int>src_batch_ind, des_batch_ind;
    for(int i = 0; i < sampled_corr.size(); i++){
        kdtree_src.radiusSearch(sampled_corr[i].src_index, radius, src_ind, src_dis);
        kdtree_des.radiusSearch(sampled_corr[i].des_index, radius, des_ind, des_dis);
        sort(src_ind.begin(), src_ind.end());
        sort(des_ind.begin(), des_ind.end());
        src_batch_ind = vectors_union(src_ind, src_batch_ind);
        des_batch_ind = vectors_union(des_ind, des_batch_ind);
    }
    pcl::copyPointCloud(*src, src_batch_ind, *src_batch);
    pcl::copyPointCloud(*des, des_batch_ind, *des_batch);
    return;
}

float trancatedChamferDistance(PointCloudPtr& src, PointCloudPtr& des, Eigen::Matrix4f &est, float thresh){
    PointCloudPtr src_trans(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::transformPointCloud(*src, *src_trans, est);
    //remove nan points
    src_trans->is_dense = false;
    vector<int>mapping;
    pcl::removeNaNFromPointCloud(*src_trans, *src_trans, mapping);
    if(!src_trans->size()) return 0;
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree_src_trans, kdtree_des;
    kdtree_src_trans.setInputCloud(src_trans);
    kdtree_des.setInputCloud(des);
    vector<int>src_ind(1), des_ind(1);
    vector<float>src_dis(1), des_dis(1);
    float score1 = 0, score2 = 0;
    int cnt1 = 0, cnt2 = 0;
    for(int i = 0; i < src_trans->size(); i++){
        pcl::PointXYZ src_trans_query = (*src_trans)[i];
        if(!pcl::isFinite(src_trans_query)) continue;
        kdtree_des.nearestKSearch(src_trans_query, 1, des_ind, des_dis);
        if(des_dis[0] > pow(thresh, 2)){
            continue;
        }
        score1 += (thresh - sqrt(des_dis[0])) / thresh;
        cnt1 ++;
    }
    score1 /= cnt1;
    for(int i = 0; i < des->size(); i++){
        pcl::PointXYZ des_query = (*des)[i];
        if(!pcl::isFinite(des_query)) continue;
        kdtree_src_trans.nearestKSearch(des_query, 1, src_ind, src_dis);
        if(src_dis[0] > pow(thresh, 2)){
            continue;
        }
        score2 += (thresh - sqrt(src_dis[0])) / thresh;
        cnt2++;
    }
    score2 /= cnt2;
    return (score1 + score2) / 2;
}

Eigen::Matrix4f clusterInternalTransEva(pcl::IndicesClusters &clusterTrans, int best_index, Eigen::Matrix4f &initial, vector<Eigen::Matrix3f> &Rs, vector<Eigen::Vector3f> &Ts,
                                        PointCloudPtr& src_kpts, PointCloudPtr& des_kpts, vector<pair<int, vector<int>>> &des_src, float thresh, Eigen::Matrix4f& GTmat, string folderpath){

    //string cluster_eva = folderpath + "/cluster_eva.txt";
    //ofstream outfile(cluster_eva, ios::trunc);
    //outfile.setf(ios::fixed, ios::floatfield);

    float RE, TE;
    bool suc = evaluation_est(initial, GTmat, 15, 30, RE, TE);


    Eigen::Matrix3f R_initial = initial.topLeftCorner(3,3);
    Eigen::Vector3f T_initial = initial.block(0, 3, 3, 1);
    float max_score = OAMAE(src_kpts, des_kpts, initial, des_src, thresh);
    cout << "Center est: " << suc << ", RE = " << RE << ", TE = " << TE << ", score = " << max_score << endl;
    //outfile << setprecision(4) << RE << " " << TE << " " << max_score << " "<< suc <<  endl;
    Eigen::Matrix4f est = initial;

    //统计类内R T差异情况
    vector<pair<float, float>> RTdifference;
    float avg_Rdiff =0, avg_Tdiff =0;
    int n = 0;
    for(int i = 0; i < clusterTrans[best_index].indices.size(); i++){
        int ind = clusterTrans[best_index].indices[i];
        Eigen::Matrix3f R = Rs[ind];
        Eigen::Vector3f T = Ts[ind];
        float R_diff = calculate_rotation_error(R, R_initial);
        float T_diff = calculate_translation_error(T, T_initial);
        if(isfinite(R_diff) && isfinite(T_diff)){
            avg_Rdiff += R_diff;
            avg_Tdiff += T_diff;
            n++;
        }
        RTdifference.emplace_back(R_diff, T_diff);
    }
    avg_Tdiff /= n;
    avg_Rdiff /= n;

    for(int i = 0; i < clusterTrans[best_index].indices.size(); i++){
        //继续缩小解空间
        if(!isfinite(RTdifference[i].first) || !isfinite(RTdifference[i].second) || RTdifference[i].first > avg_Rdiff || RTdifference[i].second > avg_Tdiff) continue;
        //if(RTdifference[i].first > 5 || RTdifference[i].second > 10) continue;
        int ind = clusterTrans[best_index].indices[i];
        Eigen::Matrix4f mat;
        mat.setIdentity();
        mat.block(0, 3, 3, 1) = Ts[ind];
        mat.topLeftCorner(3,3) = Rs[ind];
        suc = evaluation_est(mat, GTmat, 15, 30, RE, TE);
        float score = OAMAE(src_kpts, des_kpts, mat, des_src, thresh);
        //outfile << setprecision(4) << RE << " " << TE << " " << score << " "<< suc <<endl;
        if(score > max_score){
            max_score = score;
            est = mat;
            cout << "Est in cluster: " << suc << ", RE = " << RE << ", TE = " << TE  << ", score = " << score  <<endl;
        }
    }
    //outfile.close();
    return est;
}

void make_des_src_pair(const vector<Corre_3DMatch>& correspondence, vector<pair<int, vector<int>>>& des_src){ //需要读取保存的kpts, 匹配数据按照索引形式保存
    assert(correspondence.size() > 1);
    des_src.clear();
    vector<Corre_3DMatch> corr;
    corr.assign(correspondence.begin(), correspondence.end());
    sort(corr.begin(), corr.end(), compare_corres_ind);
    int des = corr[0].des_index;
    vector<int>src;
    src.push_back(corr[0].src_index);
    for(int i = 1; i < corr.size(); i++){
        if(corr[i].des_index != des){
            des_src.emplace_back(des, src);
            src.clear();
            des = corr[i].des_index;
        }
        src.push_back(corr[i].src_index);
    }
    corr.clear();
    corr.shrink_to_fit();
}

bool checkValue(float value){
    if(isfinite(value) && value >= 0 && value <= 2*M_PIf32){
        return true;
    }
    return false;
}