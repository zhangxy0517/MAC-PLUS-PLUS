#ifndef _EVA_H_ 
#define _EVA_H_
#define constE 2.718282
#define NULL_POINTID -1
#define NULL_Saliency -1000
#define Random(x) (rand()%x)
#define Corres_view_gap -200
#define Align_precision_threshold 0.1
#define tR 116//30
#define tG 205//144
#define tB 211//255
#define sR 253//209//220
#define sG 224//26//20
#define sB 2//32//60
#define L2_thresh 0.5
#define Ratio_thresh 0.2
#define GC_dist_thresh 3
#define Hough_bin_num 15
#define SI_GC_thresh 0.8
#define RANSAC_Iter_Num 5000
#define GTM_Iter_Num 100
#define CV_voting_size 20
#define EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET
using namespace std;

extern bool add_overlap;
extern bool low_inlieratio;
extern bool no_logs;
//
#include <pcl/surface/gp3.h>
#include <pcl/surface/mls.h>
#include <unordered_set>
#include <Eigen/Eigen>
#include <igraph/igraph.h>
#include <sys/stat.h>
#include <unistd.h>
#include <pcl/segmentation/impl/conditional_euclidean_clustering.hpp>
//
typedef pcl::PointCloud<pcl::PointXYZ>::Ptr PointCloudPtr;
struct hashFunction
{
    size_t operator()(const vector<int>
                      &myVector) const
    {
        std::hash<int> hasher;
        size_t answer = 0;

        for (int i : myVector)
        {
            answer ^= hasher(i) + 0x9e3779b9 +
                      (answer << 6) + (answer >> 2);
        }
        return answer;
    }
};
typedef struct {
	float x;
	float y;
	float z;
}Vertex;
typedef struct {
	int pointID;
	Vertex x_axis;
	Vertex y_axis;
	Vertex z_axis;
}LRF;
typedef struct {
	int source_idx;
	int target_idx;
	LRF source_LRF;
	LRF target_LRF;
	float score;
}Corre;
typedef struct {
	int src_index;
	int des_index;
	pcl::PointXYZ src;
	pcl::PointXYZ des;
	Eigen::Vector3f src_norm;
	Eigen::Vector3f des_norm;
	float score;
	int inlier_weight;
}Corre_3DMatch;
typedef struct
{
	int index;
	float score;
    bool flag;
}Vote;
typedef struct
{
	int index;
	int degree;
	float score;
	vector<int> corre_index;
	int true_num;
}Vote_exp;
typedef struct {
    int corre_ind;
    vector<Vote>clique_ind_score;
    float score;
}local;
typedef struct
{
    int clique_index;
    int clique_size;
    float clique_weight;
    int clique_num;
}node_cliques;
/**********************************************funcs***************************************/
//dataload
int XYZorPly_Read(string Filename, PointCloudPtr& cloud);
float MeshResolution_mr_compute(PointCloudPtr& cloud);
void feature_matching(PointCloudPtr& cloud_source, PointCloudPtr& cloud_target,
                      vector<vector<float>>& feature_source, vector<vector<float>>& feature_target, vector<Corre_3DMatch>& Corres);
int Correct_corre_compute(PointCloudPtr &cloud_s, PointCloudPtr &cloud_t, vector<Corre_3DMatch> &Corres, float correct_thresh, Eigen::Matrix4d& GT_mat, string path);
void Correct_corre_select(PointCloudPtr cloud_s, PointCloudPtr cloud_t, vector<Corre> Corres, float correct_thresh,
	Eigen::Matrix4f& GT_mat, vector<Corre>& Corres_selected);
float OTSU_thresh(vector<float> values);
float Distance(pcl::PointXYZ& A, pcl::PointXYZ& B);
Eigen::MatrixXf Graph_construction(vector<Corre_3DMatch>& correspondence, float resolution, bool sc2, const string &name,const string &descriptor, float inlier_thresh);
/**********************************************3DCorres_methods***************************************/
//descriptor
void SHOT_compute(PointCloudPtr &cloud, vector<int> &indices, float sup_radius, vector<vector<float>>& features);
void FPFH_descriptor(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, float sup_radius, std::vector<std::vector<float>>& features);
int Voxel_grid_downsample(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr& new_cloud,
                      float leaf_size);
pcl::PointCloud<pcl::PointXYZ>::Ptr getHarris3D_detector(PointCloudPtr cloud, float NMS_radius, vector<int>& key_indices);
/**********************************************Visualization***************************************/
void visualization(const PointCloudPtr& cloud_src, const PointCloudPtr& cloud_tar, /*PointCloudPtr keyPoint_src, PointCloudPtr keyPoint_tar,*/ Eigen::Matrix4f &Mat, float resolution);
void visualization(const PointCloudPtr &cloud_src, const PointCloudPtr &cloud_tar, vector<Corre_3DMatch>&match, Eigen::Matrix4d &Mat, float &resolution);
void visualization(const PointCloudPtr& ov_src,  const PointCloudPtr& cloud_src, const PointCloudPtr& cloud_tar, vector<Corre_3DMatch>& match, Eigen::Matrix4d& Mat, Eigen::Matrix4d& GTmat, float& resolution);
void RMSE_visualization(const PointCloudPtr& cloud_source, const PointCloudPtr& cloud_target, Eigen::Matrix4f& Mat_est, Eigen::Matrix4f& Mat_GT, float mr);
float RMSE_compute(const PointCloudPtr& cloud_source, const PointCloudPtr& cloud_target, Eigen::Matrix4f& Mat_est, Eigen::Matrix4f& Mat_GT, float mr);
float RMSE_compute_scene(PointCloudPtr &cloud_source, PointCloudPtr &cloud_target, Eigen::Matrix4f& Mat_est, Eigen::Matrix4f& Mat_GT, float overlap_thresh);
void cloud_viewer_RGB(const PointCloudPtr& cloud, vector<Vertex>colors, int mode);
void cloud_viewer(const PointCloudPtr& cloud, const char* name);
void cloud_viewer_src_des(const PointCloudPtr& cloud_src, const PointCloudPtr& cloud_des);
void Corres_Viewer_Scorecolor(const PointCloudPtr& cloud_s, const PointCloudPtr& cloud_t, vector<Corre>& Hist_match, float& mr, int k);
void Corres_initial_visual(const PointCloudPtr& cloud_s, const PointCloudPtr& cloud_t, vector<Corre>& Hist_match, float& mr, Eigen::Matrix4d& GT_Mat);
void Corres_selected_visual(const PointCloudPtr& cloud_s, const PointCloudPtr& cloud_t, vector<Corre_3DMatch>& Hist_match, float& mr, float GT_thresh, Eigen::Matrix4d& GT_Mat);
void Corres_Viewer_Score(const PointCloudPtr& cloud_s, const PointCloudPtr& cloud_t, vector<Corre_3DMatch>& Hist_match, float& mr, int k);
bool compare_vote_score(const Vote& v1, const Vote& v2);
vector<int> vectors_intersection(const vector<int>& v1, const vector<int>& v2);
vector<int> vectors_union(const vector<int>& v1, const vector<int>& v2);
float calculate_rotation_error(Eigen::Matrix3f& est, Eigen::Matrix3f& gt);
float calculate_translation_error(Eigen::Vector3f& est, Eigen::Vector3f& gt);
void weight_SVD(PointCloudPtr& src_pts, PointCloudPtr& des_pts, Eigen::VectorXf& weights, float weight_threshold, Eigen::Matrix4f& trans_Mat);
float evaluation_trans(vector<Corre_3DMatch>& correspondence, PointCloudPtr& src_corr_pts, PointCloudPtr& des_corr_pts,  Eigen::Matrix4f& trans, float metric_thresh, const string &metric, float resolution);
bool evaluation_est(Eigen::Matrix4f &est, Eigen::Matrix4f &gt, float re_thresh, float te_thresh, float& RE, float& TE);
void post_refinement(vector<Corre_3DMatch>& correspondence, PointCloudPtr& src_corr_pts, PointCloudPtr& des_corr_pts, Eigen::Matrix4f& initial, float & best_score, float inlier_thresh, int iterations, const string &metric);
void GUO_ICP(PointCloudPtr& cloud_source, PointCloudPtr& cloud_target, float mr, int Max_iter_Num, Eigen::Matrix4f& Mat_ICP);
void savetxt(vector<Corre_3DMatch>corr, const string& save_path);
int clusterTransformationByRotation(vector<Eigen::Matrix3f> &Rs, vector<Eigen::Vector3f> &Ts, float angle_thresh,float dis_thresh,  pcl::IndicesClusters &clusters, pcl::PointCloud<pcl::PointXYZINormal>::Ptr &trans);
float OAMAE(PointCloudPtr& raw_src, PointCloudPtr& raw_des, Eigen::Matrix4f &est, vector<pair<int, vector<int>>> &des_src, float thresh);
float trancatedChamferDistance(PointCloudPtr& src, PointCloudPtr& des, Eigen::Matrix4f &est, float thresh);
void make_des_src_pair(const vector<Corre_3DMatch>& correspondence, vector<pair<int, vector<int>>>& des_src);
void find_clique_of_node2(Eigen::MatrixXf& Graph, igraph_vector_int_list_t* cliques, vector<Corre_3DMatch>& correspondence,vector<int>& sampled_ind, vector<int>&remain);
void getCorrPatch(vector<Corre_3DMatch>&sampled_corr, PointCloudPtr &src, PointCloudPtr &des, PointCloudPtr &src_batch, PointCloudPtr &des_batch, float radius);
Eigen::Matrix4f clusterInternalTransEva(pcl::IndicesClusters &clusterTrans, int best_index, Eigen::Matrix4f &initial, vector<Eigen::Matrix3f> &Rs, vector<Eigen::Vector3f> &Ts,
                                         PointCloudPtr& src_kpts, PointCloudPtr& des_kpts, vector<pair<int, vector<int>>> &des_src, float thresh, Eigen::Matrix4f& GTmat, string folderpath);
Eigen::Matrix4f clusterInternalTransEva1(pcl::IndicesClusters &clusterTrans, int best_index, Eigen::Matrix4f &initial, vector<Eigen::Matrix3f> &Rs, vector<Eigen::Vector3f> &Ts,
                                         PointCloudPtr& src_kpts, PointCloudPtr& des_kpts, vector<pair<int, vector<int>>> &des_src, float thresh, Eigen::Matrix4f& GTmat, bool _1tok ,string folderpath);
#endif