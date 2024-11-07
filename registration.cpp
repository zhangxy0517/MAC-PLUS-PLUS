#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/shot.h>
#include <pcl/registration/transformation_estimation_svd.h>
#include <pcl/registration/icp.h>
#include <pcl/visualization/pcl_visualizer.h>
//#include <direct.h>
#include <iostream>
#include <string>
#include <algorithm>
//#include <io.h>
#include "omp.h"
#include "Eva.h"
#include <stdarg.h>
#include <chrono>
#include <unistd.h>
#include <sys/types.h>
//#include <windows.h>
//#include <io.h>
using namespace Eigen;
using namespace std;
// igraph 0.10.6
 bool add_overlap;
 bool low_inlieratio;
 bool no_logs;

#define VMRSS_LINE 22 //VmRSS: 所在行数
#define PROCESS_ITEM 14
double getPidMemory(unsigned int pid){

    char file_name[64]={0};
    FILE *fd;
    char line_buff[512]={0};
    sprintf(file_name,"/proc/%d/status",pid);

    fd =fopen(file_name,"r");
    if(nullptr == fd){
        return 0;
    }

    char name[64];
    double vmrss;
    for (int i=0; i<VMRSS_LINE-1;i++){
        fgets(line_buff,sizeof(line_buff),fd);
    }

    fgets(line_buff,sizeof(line_buff),fd);
    sscanf(line_buff,"%s %lf",name,&vmrss);
    fclose(fd);

    return vmrss;
}


void calculate_gt_overlap(vector<Corre_3DMatch>&corre, PointCloudPtr &src, PointCloudPtr &tgt, Eigen::Matrix4f &GTmat, bool ind, float GT_thresh, float &max_corr_weight){
    PointCloudPtr src_trans(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::transformPointCloud(*src, *src_trans, GTmat);
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree_src_trans, kdtree_des;
    kdtree_src_trans.setInputCloud(src_trans);
    kdtree_des.setInputCloud(tgt);
    vector<int>src_ind(1), des_ind(1);
    vector<float>src_dis(1), des_dis(1);
    PointCloudPtr src_corr(new pcl::PointCloud<pcl::PointXYZ>);
    PointCloudPtr src_corr_trans(new pcl::PointCloud<pcl::PointXYZ>);
    if(!ind){
        for(auto & i :corre){
            src_corr->points.push_back(i.src);
        }
        pcl::transformPointCloud(*src_corr, *src_corr_trans, GTmat);
        src_corr.reset(new pcl::PointCloud<pcl::PointXYZ>);
    }
    for(int i  = 0; i < (int )corre.size(); i++){
        pcl::PointXYZ src_query, des_query;
        if(!ind){
            src_query = src_corr_trans->points[i];
            des_query = corre[i].des;
        }
        else{
            src_query = src->points[corre[i].src_index];
            des_query = tgt->points[corre[i].des_index];
        }
        kdtree_des.nearestKSearch(src_query, 1, des_ind, src_dis);
        kdtree_src_trans.nearestKSearch(des_query, 1, src_ind, des_dis);
        int src_ov_score = src_dis[0] > pow(GT_thresh,2) ? 0 : 1; //square dist  <= GT_thresh
        int des_ov_score = des_dis[0] > pow(GT_thresh,2) ? 0 : 1;
        if(src_ov_score && des_ov_score){
            corre[i].score = 1;
            max_corr_weight = 1;
        }
        else{
            corre[i].score = 0;
        }
    }
    src_corr_trans.reset(new pcl::PointCloud<pcl::PointXYZ>);
    src_trans.reset(new pcl::PointCloud<pcl::PointXYZ>);
}


void find_index_for_corr(PointCloudPtr &src, PointCloudPtr &des, vector<Corre_3DMatch > &corr){
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree_src, kdtree_des;
    kdtree_src.setInputCloud(src);
    kdtree_des.setInputCloud(des);
    vector<int>src_ind(1), des_ind(1);
    vector<float>src_dis(1), des_dis(1);
    for(int i = 0; i < (int)corr.size(); i++){
        pcl::PointXYZ src_pt, des_pt;
        src_pt = corr[i].src;
        des_pt = corr[i].des;
        kdtree_src.nearestKSearch(src_pt, 1, src_ind, src_dis);
        kdtree_des.nearestKSearch(des_pt, 1, des_ind, des_dis);
        corr[i].src_index = src_ind[0];
        corr[i].des_index = des_ind[0];
    }
    return;
}

bool registration(const string &name, const string &src_pointcloud, const string &des_pointcloud, const string &corr_path, const string &label_path, const string &ov_label, const string &gt_mat, const string &folderPath, const string &descriptor, double& time_epoch, double& mem_epoch, vector<double>& time_number, float &RE, float &TE, int &correct_est_num, int &inlier_num, int &total_num, vector<double>&pred_inlier) {
    bool sc2 = true;
    bool use_icp = false;
    bool instance_equal = true;
    bool cluster_internal_eva = true;
    int max_est_num = INT_MAX;
    low_inlieratio = false;
    add_overlap = false;
    no_logs = false;
    string metric = "MAE";
    omp_set_num_threads(12);
    int success_num = 0;

    cout << folderPath << endl;
    // 获取数据文件目录
    string dataPath = corr_path.substr(0, corr_path.rfind("/"));
    // 获取当前项目名称
    string item_name = folderPath.substr(folderPath.rfind("/") + 1, folderPath.length());

    vector<pair<int, vector<int>>> one2k_match;

    FILE* corr, * gt;
    corr = fopen(corr_path.c_str(), "r");
    gt = fopen(label_path.c_str(), "r");
    if (corr == NULL) {
        std::cout << " error in loading correspondence data. " << std::endl;
        cout << corr_path << endl;
        exit(-1);
    }
    if (gt == NULL) {
        std::cout << " error in loading ground truth label data. " << std::endl;
        cout << label_path << endl;
        exit(-1);
    }

    FILE* ov;
    vector<float>ov_corr_label;
    float max_corr_weight = 0;
    if (add_overlap && ov_label != "NULL")
    {
        ov = fopen(ov_label.c_str(), "r");
        if (ov == NULL) {
            std::cout << " error in loading overlap data. " << std::endl;
            exit(-1);
        }
        cout << ov_label << endl;
        while (!feof(ov))
        {
            float value;
            fscanf(ov, "%f\n", &value);
            if(value > max_corr_weight){
                max_corr_weight = value;
            }
            ov_corr_label.push_back(value);
        }
        fclose(ov);
        cout << "load overlap data finished." << endl;
    }

    //PointCloudPtr Overlap_src(new pcl::PointCloud<pcl::PointXYZ>);
    PointCloudPtr Raw_src(new pcl::PointCloud<pcl::PointXYZ>);
    PointCloudPtr Raw_des(new pcl::PointCloud<pcl::PointXYZ>);
    float raw_des_resolution = 0;
    float raw_src_resolution = 0;
    //pcl::KdTreeFLANN<pcl::PointXYZ>kdtree_Overlap_des, kdtree_Overlap_src;

    PointCloudPtr cloud_src(new pcl::PointCloud<pcl::PointXYZ>);
    PointCloudPtr cloud_des(new pcl::PointCloud<pcl::PointXYZ>);
    PointCloudPtr cloud_src_kpts(new pcl::PointCloud<pcl::PointXYZ>);
    PointCloudPtr cloud_des_kpts(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::Normal>::Ptr normal_src(new pcl::PointCloud<pcl::Normal>);//法向量计算结果
    pcl::PointCloud<pcl::Normal>::Ptr normal_des(new pcl::PointCloud<pcl::Normal>);
    vector<Corre_3DMatch>correspondence;
    vector<int>true_corre;
    inlier_num = 0;
    float resolution = 0;
    bool kitti = false;
    Eigen::Matrix4f GTmat;


    FILE* fp = fopen(gt_mat.c_str(), "r");
    if (fp == NULL)
    {
        printf("Mat File can't open!\n");
        return -1;
    }
    fscanf(fp, "%f %f %f %f\n", &GTmat(0, 0), &GTmat(0, 1), &GTmat(0, 2), &GTmat(0, 3));
    fscanf(fp, "%f %f %f %f\n", &GTmat(1, 0), &GTmat(1, 1), &GTmat(1, 2), &GTmat(1, 3));
    fscanf(fp, "%f %f %f %f\n", &GTmat(2, 0), &GTmat(2, 1), &GTmat(2, 2), &GTmat(2, 3));
    fscanf(fp, "%f %f %f %f\n", &GTmat(3, 0), &GTmat(3, 1), &GTmat(3, 2), &GTmat(3, 3));
    fclose(fp);
    if (low_inlieratio)
    {
        if (pcl::io::loadPCDFile(src_pointcloud.c_str(), *cloud_src) < 0) {
            std::cout << " error in loading source pointcloud. " << std::endl;
            exit(-1);
        }

        if (pcl::io::loadPCDFile(des_pointcloud.c_str(), *cloud_des) < 0) {
            std::cout << " error in loading target pointcloud. " << std::endl;
            exit(-1);
        }
        while (!feof(corr)) {
            Corre_3DMatch t;
            pcl::PointXYZ src, des;
            fscanf(corr, "%f %f %f %f %f %f\n", &src.x, &src.y, &src.z, &des.x, &des.y, &des.z);
            t.src = src;
            t.des = des;
            correspondence.push_back(t);
        }
        if(add_overlap && ov_label == "NULL") { // GT overlap
            cout << "load gt overlap" << endl;
            calculate_gt_overlap(correspondence, cloud_src, cloud_des, GTmat, false, 0.0375, max_corr_weight);
        }
        else if (add_overlap && ov_label != "NULL"){
            for(int i  = 0; i < (int )correspondence.size(); i++){
                correspondence[i].score = ov_corr_label[i];
                if(ov_corr_label[i] > max_corr_weight){
                    max_corr_weight = ov_corr_label[i];
                }
            }
        }
        fclose(corr);
    }
    else {
        if (name == "KITTI")//KITTI
        {
            int idx = 0;
            kitti = true;
            string src_kpts = dataPath + "/src_kpts.pcd";
            string des_kpts = dataPath + "/tgt_kpts.pcd";
            if (pcl::io::loadPCDFile(src_kpts.c_str(), *cloud_src_kpts) < 0) {
                std::cout << " error in loading source pointcloud. " << std::endl;
                exit(-1);
            }

            if (pcl::io::loadPCDFile(des_kpts.c_str(), *cloud_des_kpts) < 0) {
                std::cout << " error in loading target pointcloud. " << std::endl;
                exit(-1);
            }
            while (!feof(corr))
            {
                Corre_3DMatch t;
//                int src_ind, tgt_ind;
//                fscanf(corr, "%d %d\n", &src_ind, &tgt_ind);
//                t.src_index = src_ind;
//                t.des_index = tgt_ind;
//                t.src = cloud_src_kpts->points[src_ind];
//                t.des = cloud_des_kpts->points[tgt_ind];
                pcl::PointXYZ src, des;
                fscanf(corr, "%f %f %f %f %f %f\n", &src.x, &src.y, &src.z, &des.x, &des.y, &des.z);
                t.src = src;
                t.des = des;
                t.inlier_weight = 0;
                if (add_overlap)
                {
                    t.score = ov_corr_label[idx];
                }
                else
                {
                    t.score = 0;
                }
                correspondence.push_back(t);
                idx++;
            }
            fclose(corr);
            find_index_for_corr(cloud_src_kpts, cloud_des_kpts, correspondence);
        }
        else if (name == "U3M") {
            XYZorPly_Read(src_pointcloud.c_str(), cloud_src);
            XYZorPly_Read(des_pointcloud.c_str(), cloud_des);
            float resolution_src = MeshResolution_mr_compute(cloud_src);
            float resolution_des = MeshResolution_mr_compute(cloud_des);
            resolution = (resolution_des + resolution_src) / 2;
            cout << resolution << endl;
            string src_kpts = folderPath + "/src_kpts.pcd";
            string des_kpts = folderPath + "/tgt_kpts.pcd";
            if (pcl::io::loadPCDFile(src_kpts.c_str(), *cloud_src_kpts) < 0) {
                std::cout << " error in loading source pointcloud. " << std::endl;
                exit(-1);
            }

            if (pcl::io::loadPCDFile(des_kpts.c_str(), *cloud_des_kpts) < 0) {
                std::cout << " error in loading target pointcloud. " << std::endl;
                exit(-1);
            }
            int idx = 0;
            while (!feof(corr))
            {
                Corre_3DMatch t;
//                int src_ind, tgt_ind;
//                fscanf(corr, "%d %d\n", &src_ind, &tgt_ind);
//                t.src_index = src_ind;
//                t.des_index = tgt_ind;
//                t.src = cloud_src_kpts->points[src_ind];
//                t.des = cloud_des_kpts->points[tgt_ind];
                pcl::PointXYZ src, des;
                fscanf(corr, "%f %f %f %f %f %f\n", &src.x, &src.y, &src.z, &des.x, &des.y, &des.z);
                t.src = src;
                t.des = des;
                t.inlier_weight = 0;
                if (add_overlap)
                {
                    t.score = ov_corr_label[idx];
                }
                else
                {
                    t.score = 0;
                }
                correspondence.push_back(t);
                idx++;
            }
            fclose(corr);
            find_index_for_corr(cloud_src_kpts, cloud_des_kpts, correspondence);
        }
        else if (name == "3dmatch" || name == "3dlomatch") {

            if (!(src_pointcloud == "NULL" && des_pointcloud == "NULL"))
            {
                if (pcl::io::loadPLYFile(src_pointcloud.c_str(), *cloud_src) < 0) {
                    std::cout << " error in loading source pointcloud. " << std::endl;
                    exit(-1);
                }

                if (pcl::io::loadPLYFile(des_pointcloud.c_str(), *cloud_des) < 0) {
                    std::cout << " error in loading target pointcloud. " << std::endl;
                    exit(-1);
                }
                float resolution_src = MeshResolution_mr_compute(cloud_src);
                float resolution_des = MeshResolution_mr_compute(cloud_des);
                resolution = (resolution_des + resolution_src) / 2;

                string src_kpts = src_pointcloud.substr(0, src_pointcloud.rfind('.')) + ".pcd";
                string des_kpts = des_pointcloud.substr(0, des_pointcloud.rfind('.')) + ".pcd";
                //string srcname = src_pointcloud.substr(src_pointcloud.rfind('/') + 1, src_pointcloud.rfind('.') - src_pointcloud.rfind('/') - 1) + ".pcd";
                //string desname = des_pointcloud.substr(des_pointcloud.rfind('/') + 1, des_pointcloud.rfind('.') - des_pointcloud.rfind('/') - 1) + ".pcd";
                //string src_kpts = dataPath +'/' + descriptor + '/' + srcname;
                //string des_kpts = dataPath +'/' + descriptor + '/' + desname;


                if (pcl::io::loadPCDFile(src_kpts.c_str(), *cloud_src_kpts) < 0) {
                    std::cout << " error in loading source keypoints. " << std::endl;
                    exit(-1);
                }

                if (pcl::io::loadPCDFile(des_kpts.c_str(), *cloud_des_kpts) < 0) {
                    std::cout << " error in loading target keypoints. " << std::endl;
                    exit(-1);
                }

                int idx = 0;
                while (!feof(corr))
                {
                    Corre_3DMatch t;
                    pcl::PointXYZ src, des;
                    fscanf(corr, "%f %f %f %f %f %f\n", &src.x, &src.y, &src.z, &des.x, &des.y, &des.z);
                    t.src = src;
                    t.des = des;

//                    int src_ind, tgt_ind;
//                    fscanf(corr, "%d %d\n", &src_ind, &tgt_ind);
//                    t.src_index = src_ind;
//                    t.des_index = tgt_ind;
//                    t.src = cloud_src_kpts->points[src_ind];
//                    t.des = cloud_des_kpts->points[tgt_ind];
                    if (add_overlap && ov_label != "NULL")
                    {
                        t.score = ov_corr_label[idx];
                    }
                    else{
                        t.score = 0;
                    }
                    t.inlier_weight = 0;
                    correspondence.push_back(t);
                    idx ++;
                }
                fclose(corr);
                find_index_for_corr(cloud_src_kpts, cloud_des_kpts, correspondence);
                if(add_overlap && ov_label == "NULL"){
                    cout << "load gt overlap" << endl;
                    calculate_gt_overlap(correspondence, cloud_src, cloud_des, GTmat, false, 0.0375, max_corr_weight);
                }
            }
            else {
                int idx = 0;
                while (!feof(corr))
                {
                    Corre_3DMatch t;
                    pcl::PointXYZ src, des;
                    fscanf(corr, "%f %f %f %f %f %f\n", &src.x, &src.y, &src.z, &des.x, &des.y, &des.z);
                    t.src = src;
                    t.des = des;
                    t.inlier_weight = 0;
                    if (add_overlap)
                    {
                        t.score = ov_corr_label[idx];
                    }
                    else
                    {
                        t.score = 0;
                    }
                    correspondence.push_back(t);
                    idx++;
                }
                fclose(corr);
                string src_kpts = dataPath + '/' + item_name + "@kpt_src.pcd";
                string tgt_kpts = dataPath + '/' + item_name + "@kpt_tgt.pcd";
                string raw_src = dataPath + '/' + item_name + "@raw_src.pcd";
                string raw_tgt = dataPath + '/' + item_name + "@raw_tgt.pcd";

                if (pcl::io::loadPCDFile(raw_src.c_str(), *cloud_src) < 0)
                {
                    cout << "error in loading raw_src pcd" << endl;
                    exit(-1);
                }
                if (pcl::io::loadPCDFile(raw_tgt.c_str(), *cloud_des) < 0)
                {
                    cout << "error in loading raw_des pcd" << endl;
                    exit(-1);
                }
                if (pcl::io::loadPCDFile(src_kpts.c_str(), *cloud_src_kpts) < 0) {
                    std::cout << " error in loading source keypoints. " << std::endl;
                    exit(-1);
                }

                if (pcl::io::loadPCDFile(tgt_kpts.c_str(), *cloud_des_kpts) < 0) {
                    std::cout << " error in loading target keypoints. " << std::endl;
                    exit(-1);
                }
                find_index_for_corr(cloud_src_kpts, cloud_des_kpts, correspondence);
            }
        }
        else {
            exit(-1);
        }
    }

    total_num = correspondence.size();
    while (!feof(gt))
    {
        int value;
        fscanf(gt, "%d\n", &value);
        true_corre.push_back(value);
        if (value == 1)
        {
            inlier_num++;
        }
    }
    fclose(gt);

    float inlier_ratio = 0;
    if (inlier_num == 0)
    {
        cout << " NO INLIERS！ " << endl;
    }
    inlier_ratio = inlier_num / (total_num / 1.0);

//    string _1tok = dataPath + '/' + item_name + "@1tok.txt";
//    fp = fopen(_1tok.c_str(), "r");
//    if (fp == NULL)
//    {
//        printf("1tok File can't open!\n");
//        return -1;
//    }
//    int i = 0;
//    while (!feof(fp)) {
//        vector<int>relax_ind;
//        int value;
//        for(int j = 0; j < 99; j++){
//            fscanf(fp, "%d ", &value);
//            relax_ind.push_back(value);
//        }
//        fscanf(fp, "%d\n", &value);
//        relax_ind.push_back(value);
//        one2k_match.emplace_back(i, relax_ind);
//        relax_ind.clear();
//        i++;
//    }
//    fclose(fp);

/**********************************不同数据集的阈值参数设置************************************/
    float RE_thresh, TE_thresh, inlier_thresh;
    if (name == "KITTI")
    {
        RE_thresh = 5;
        TE_thresh = 180;
        inlier_thresh = 1.8;
    }
    else if (name == "3dmatch" || name == "3dlomatch")
    {
        RE_thresh = 15;
        TE_thresh = 30;
        inlier_thresh = 0.1;
    }
    else if (name == "U3M") {
        inlier_thresh = 5 * resolution;
    }
    RE = RE_thresh;
    TE = TE_thresh;
    std::chrono::time_point<std::chrono::system_clock> start, end;
    std::chrono::duration<double> elapsed_time;
    time_number.clear();
/********************************构图****************************************/
    start = std::chrono::system_clock::now();
    Eigen::MatrixXf Graph = Graph_construction(correspondence, resolution, sc2, name, descriptor, inlier_thresh);
    end = std::chrono::system_clock::now();
    elapsed_time = end - start;
    time_epoch += std::chrono::duration_cast<std::chrono::milliseconds>(elapsed_time).count();
    time_number[0] = elapsed_time.count();
    cout << " graph construction: " << elapsed_time.count() << endl;
    if (Graph.norm() == 0) {
        cout << "Graph is disconnected. You may need to check the compatibility threshold!" << endl;
        return false;
    }

    vector<int>degree(total_num, 0);
    vector<Vote_exp> pts_degree;
    for (int i = 0; i < total_num; i++)
    {
        Vote_exp t;
        t.true_num = 0;
        vector<int> corre_index;
        for (int j = 0; j < total_num; j++)
        {
            if (i != j && Graph(i, j)) {
                degree[i] ++;
                corre_index.push_back(j);
                if (true_corre[j])
                {
                    t.true_num++;
                }
            }
        }
        t.index = i;
        t.degree = degree[i];
        t.corre_index = corre_index;
        pts_degree.push_back(t);
    }

    //计算节点聚类系数，判断图是否密集，若密集则去除部分节点和边
    start = std::chrono::system_clock::now();
    vector<Vote> cluster_factor;
    float sum_fenzi = 0;
    float sum_fenmu = 0;

    for (int i = 0; i < total_num; i++)
    {
        Vote t;
        float sum_i = 0;
        float wijk = 0;
        int index_size = pts_degree[i].corre_index.size();
#pragma omp parallel
        {
#pragma omp for
            for (int j = 0; j < index_size; j++)
            {
                int a = pts_degree[i].corre_index[j];
                for (int k = j + 1; k < index_size; k++)
                {
                    int b = pts_degree[i].corre_index[k];
                    if (Graph(a, b)) {
#pragma omp critical
                        wijk += pow(Graph(i, a) * Graph(i, b) * Graph(a, b), 1.0 / 3); //wij + wik
                    }
                }
            }
        }

        if (degree[i] > 1)
        {
            float f1 = wijk;
            float f2 = degree[i] * (degree[i] - 1) * 0.5;
            sum_fenzi += f1;
            sum_fenmu += f2;
            float factor = f1 / f2;
            t.index = i;
            t.score = factor;
            cluster_factor.push_back(t);
        }
        else {
            t.index = i;
            t.score = 0;
            cluster_factor.push_back(t);
        }
    }
    end = std::chrono::system_clock::now();
    elapsed_time = end - start;
    cout << " coefficient computation: " << elapsed_time.count() << endl;
    float average_factor = 0;
    for (size_t i = 0; i < cluster_factor.size(); i++)
    {
        average_factor += cluster_factor[i].score;
    }
    average_factor /= cluster_factor.size();

    float total_factor = sum_fenzi / sum_fenmu;

    vector<Vote>cluster_factor_bac;
    cluster_factor_bac.assign(cluster_factor.begin(), cluster_factor.end());
    sort(cluster_factor.begin(), cluster_factor.end(), compare_vote_score);

    vector<float> cluster_coefficients;
    cluster_coefficients.resize(cluster_factor.size());
    for (size_t i = 0; i < cluster_factor.size(); i++)
    {
        cluster_coefficients[i] = cluster_factor[i].score;
    }

    float OTSU = 0;
    if (cluster_factor[0].score != 0)
    {
        OTSU = OTSU_thresh(cluster_coefficients);
    }
    float cluster_threshold = min(OTSU, min(average_factor, total_factor));

    cout << cluster_threshold << "->min(" << average_factor << " " << total_factor << " " << OTSU << ")" << endl;
    cout << " inliers: " << inlier_num << "\ttotal num: " << total_num << "\tinlier ratio: " << inlier_ratio*100 << "%" << endl;
    //OTSU计算权重的阈值
    float weight_thresh; //OTSU_thresh(sorted);

    if (add_overlap)
    {
        cout << "Max weight: " << max_corr_weight << endl;
        if(max_corr_weight > 0.5){
            weight_thresh = 0.5;
            //internal_selection = true;
        }
        else {
            cout << "internal selection is unused." << endl;
            weight_thresh = 0;
            if(max_corr_weight == 0){
                instance_equal = true;
            }
        }
    }
    else {
        weight_thresh = 0;
    }

    //匹配置信度评分
    if (!add_overlap || instance_equal)
    {
        for (size_t i = 0; i < total_num; i++)
        {
            correspondence[i].score = cluster_factor_bac[i].score;
        }
    }

    /*****************************************调同igraph搜索团**************************************************/
    igraph_t g;
    igraph_matrix_t g_mat;
    igraph_matrix_init(&g_mat, Graph.rows(), Graph.cols());

    //减少图规模
    if (cluster_threshold > 2.9 && correspondence.size() > 50) // default 3 kitti-lc 2
    {
        float f = 10;
        while (1)
        {
            if (f * max(OTSU, total_factor) > cluster_factor[49].score)
            {
                f -= 0.05;
            }
            else {
                break;
            }
        }
        for (int i = 0; i < Graph.rows(); i++)
        {
            if (cluster_factor_bac[i].score > f * max(OTSU, total_factor))
            {
                for (int j = i + 1; j < Graph.cols(); j++)
                {
                    if (cluster_factor_bac[j].score > f * max(OTSU, total_factor))
                    {
                        MATRIX(g_mat, i, j) = Graph(i, j);
                        MATRIX(g_mat, j, i) = MATRIX(g_mat, i, j);
                    }
                }
            }
        }
    }
    else {
        for (int i = 0; i < Graph.rows(); i++)
        {
            for (int j = i + 1; j < Graph.cols(); j++)
            {
                if (Graph(i, j))
                {
                    MATRIX(g_mat, i, j) = Graph(i, j);
                    MATRIX(g_mat, j, i) = MATRIX(g_mat, i, j);
                }
            }
        }
    }

    igraph_set_attribute_table(&igraph_cattribute_table);
    igraph_vector_t weight;
    igraph_vector_init(&weight, 0);
    igraph_weighted_adjacency(&g, &g_mat, IGRAPH_ADJ_UNDIRECTED, &weight, IGRAPH_LOOPS_ONCE);


    //找出所有最大团
    igraph_vector_int_list_t cliques;
    igraph_vector_int_list_init(&cliques, 0);
    start = std::chrono::system_clock::now();

    int min_clique_size = 3;
    if(kitti){
        min_clique_size = 4;
    }
    int max_clique_size = 0;
    bool recomputecliques = true;
    int clique_num = 0; //默认无上限
    int iter_num = 1;

    //控制搜索到的数量
    while(recomputecliques){
        igraph_maximal_cliques(&g, &cliques, min_clique_size,  max_clique_size); //3dlomatch 3 3dmatch; 3 Kitti 4 (说明)
        clique_num = igraph_vector_int_list_size(&cliques);
        if(clique_num > 10000000 && iter_num <= 5){
            max_clique_size = 15;
            min_clique_size +=iter_num;
            iter_num++;
            igraph_vector_int_list_destroy(&cliques);
            igraph_vector_int_list_init(&cliques, 0);
            cout << "clique number " << clique_num << " is too large! recomputing .." << endl;
        }
        else{
            recomputecliques = false;
        }
    }

    end = std::chrono::system_clock::now();
    elapsed_time = end - start;
    time_epoch += std::chrono::duration_cast<std::chrono::milliseconds>(elapsed_time).count();
    time_number[1] = elapsed_time.count();

    if (clique_num == 0) {
        //若搜索不到团，提示无法配准
        cout << " NO CLIQUES! " << endl;
        return false;
    }
    cout << " clique computation: " << elapsed_time.count() << endl;

    //数据清理
    igraph_destroy(&g);
    igraph_matrix_destroy(&g_mat);
    start = std::chrono::system_clock::now();

/*****************************************种子匹配生成以及团筛选**************************************************/
    vector<int>remain;
    vector<int> sampled_ind; //排过序的
    vector<Corre_3DMatch> sampled_corr;
    PointCloudPtr sampled_corr_src(new pcl::PointCloud<pcl::PointXYZ>);
    PointCloudPtr sampled_corr_des(new pcl::PointCloud<pcl::PointXYZ>);
    int num = 0;

    find_clique_of_node2(Graph, &cliques, correspondence,sampled_ind, remain);
    for(auto &ind : sampled_ind){
        sampled_corr.push_back(correspondence[ind]);
        sampled_corr_src->push_back(correspondence[ind].src);
        sampled_corr_des->push_back(correspondence[ind].des);
        if(true_corre[ind]){
            num++;
        }
    }
   ///TODO 注意这里的内点率要比初始匹配内点率高
    cout << sampled_ind.size() << " sampled correspondences have " << num << " inlies: "<< num / ((int)sampled_ind.size() / 1.0) * 100 << "%" << endl;

    string sampled_corr_txt = folderPath + "/sampled_corr.txt";
    ofstream outFile1;
    outFile1.open(sampled_corr_txt.c_str(), ios::out);
    for(int i = 0;i <(int)sampled_corr.size(); i++){
        outFile1 << sampled_corr[i].src_index << " " << sampled_corr[i].des_index <<endl;
    }
    outFile1.close();

    string sampled_corr_label = folderPath + "/sampled_corr_label.txt";
    ofstream outFile2;
    outFile2.open(sampled_corr_label.c_str(), ios::out);
    for(auto &ind : sampled_ind){
        if(true_corre[ind]){
            outFile2 << "1" << endl;
        }
        else{
            outFile2 << "0" << endl;
        }
    }
    outFile2.close();

    end = std::chrono::system_clock::now();
    elapsed_time = end - start;
    time_epoch += std::chrono::duration_cast<std::chrono::milliseconds>(elapsed_time).count();
    time_number[2] = elapsed_time.count();
    cout << " clique selection: " << elapsed_time.count() << endl;

    PointCloudPtr src_corr_pts(new pcl::PointCloud<pcl::PointXYZ>);
    PointCloudPtr des_corr_pts(new pcl::PointCloud<pcl::PointXYZ>);
    for (size_t i = 0; i < correspondence.size(); i++)
    {
        src_corr_pts->push_back(correspondence[i].src);
        des_corr_pts->push_back(correspondence[i].des);
    }

    /******************************************配准部分***************************************************/
    Eigen::Matrix4f best_est1, best_est2;

    bool found = false;
    float best_score = 0;

    start = std::chrono::system_clock::now();
    int total_estimate = remain.size();

    vector<Eigen::Matrix3f> Rs;
    vector<Eigen::Vector3f> Ts;
    vector<float> scores;
    vector<vector<int>>group_corr_ind;
    int max_size = 0;
    int min_size = 666;
    int selected_size = 0;

    vector<Vote>est_vector;
    vector<pair<int, vector<int>>> des_src;
    make_des_src_pair(correspondence, des_src); //将初始匹配形成点到点集的对应
#pragma omp parallel for
    for (int i = 0; i < (int)total_estimate; i++)
    {
        vector<Corre_3DMatch>Group, Group1;
        vector<int>selected_index;
        igraph_vector_int_t* v = igraph_vector_int_list_get_ptr(&cliques, remain[i]);
        int group_size = igraph_vector_int_size(v);
        for (int j = 0; j < group_size; j++)
        {
            Corre_3DMatch C = correspondence[VECTOR(*v)[j]];
            Group.push_back(C);
            selected_index.push_back(VECTOR(*v)[j]);
        }
        sort(selected_index.begin(), selected_index.end()); //交并前需要排序

        Eigen::Matrix4f est_trans;
        PointCloudPtr src_pts(new pcl::PointCloud<pcl::PointXYZ>);
        PointCloudPtr des_pts(new pcl::PointCloud<pcl::PointXYZ>);
        vector<float>weights;
        for (auto & k : Group)
        {
            if (k.score >= weight_thresh)
            {
                Group1.push_back(k);
                src_pts->push_back(k.src);
                des_pts->push_back(k.des);
                weights.push_back(k.score);
            }
        }
        if (weights.size() < 3)
        {
            continue;
        }
        Eigen::VectorXf weight_vec = Eigen::Map<Eigen::VectorXf, Eigen::Unaligned>(weights.data(), weights.size());
        weights.clear();
        weights.shrink_to_fit();
        weight_vec /= weight_vec.maxCoeff();
        if (!add_overlap || instance_equal) {
            weight_vec.setOnes(); // 2023.2.23
        }
        weight_SVD(src_pts, des_pts, weight_vec, 0, est_trans); //生成位姿变换假设
        Group.assign(Group1.begin(), Group1.end());
        Group1.clear();
/******************************************初步评估所有的假设***************************************************/
        float score = 0.0, score_local = 0.0;
        score = OAMAE(cloud_src_kpts, cloud_des_kpts, est_trans, des_src, inlier_thresh);
        score_local = evaluation_trans( Group, src_pts, des_pts, est_trans, inlier_thresh, metric, resolution);

        src_pts.reset(new pcl::PointCloud<pcl::PointXYZ>);
        des_pts.reset(new pcl::PointCloud<pcl::PointXYZ>);
        Group.clear();
        Group.shrink_to_fit();

        //GT未知
        if (score > 0)
        {
#pragma omp critical
            {
                Eigen::Matrix4f trans_f = est_trans;
                Eigen::Matrix3f R = trans_f.topLeftCorner(3, 3);
                Eigen::Vector3f T = trans_f.block(0, 3, 3, 1);
                Rs.push_back(R);
                Ts.push_back(T);
                scores.push_back(score_local);
                group_corr_ind.push_back(selected_index);
                selected_size = selected_index.size();
                Vote t;
                t.index = i;
                t.score = score;
                float re, te;
                t.flag = evaluation_est(est_trans, GTmat, 15, 30, re, te);
                if(t.flag){
                    success_num ++;
                }
                est_vector.push_back(t);
                if (best_score < score)
                {
                    best_score = score;
                    best_est1 = est_trans;
                    //selected = Group;
                    //corre_index = selected_index;
                }
            }
        }
        selected_index.clear();
        selected_index.shrink_to_fit();
    }

    //释放内存空间
    igraph_vector_int_list_destroy(&cliques);
    bool clique_reduce = false;
    vector<int>indices(est_vector.size());
    for (int i = 0; i < (int )est_vector.size(); ++i) {
        indices[i] = i;
    }
    sort(indices.begin(), indices.end(), [&est_vector](int a, int b){return est_vector[a].score > est_vector[b].score;});
    vector<Vote>est_vector1(est_vector.size());
    for(int i = 0; i < (int )est_vector.size(); i++){
        est_vector1[i] = est_vector[indices[i]];
    }
    est_vector.assign(est_vector1.begin(), est_vector1.end());
    est_vector1.clear();

    //先evaluate再筛选
    int max_num = min(min((int)total_num, (int)total_estimate), max_est_num);
    success_num = 0;
    vector<int>remained_est_ind;
    vector<Eigen::Matrix3f> Rs_new;
    vector<Eigen::Vector3f> Ts_new;
    if((int )est_vector.size() > max_num) { //选出排名靠前的假设
        cout << "too many cliques" << endl;
    }
    for(int i = 0; i < min(max_num, (int )est_vector.size()); i++){
        remained_est_ind.push_back(indices[i]);
        Rs_new.push_back(Rs[indices[i]]);
        Ts_new.push_back(Ts[indices[i]]);
        success_num += est_vector[i].flag ? 1 : 0;
    }
    Rs.clear();
    Ts.clear();
    Rs.assign(Rs_new.begin(), Rs_new.end());
    Ts.assign(Ts_new.begin(), Ts_new.end());
    Rs_new.clear();
    Ts_new.clear();

    if(success_num > 0){
        if(!no_logs){
            string est_info = folderPath + "/est_info.txt";
            ofstream est_info_file(est_info, ios::trunc);
            est_info_file.setf(ios::fixed, ios::floatfield);
            for(auto &i : est_vector){
                est_info_file << setprecision(10) << i.score << " " << i.flag << endl;
            }
            est_info_file.close();
        }
    }
    else{
        cout<< "NO CORRECT ESTIMATION!!!" << endl;
    }

    //cout << success_num << " : " << max_num << " : " << total_estimate << " : " << clique_num << endl;
    //cout << min_size << " : " << max_size << " : " << selected_size << endl;
    correct_est_num = success_num;
/******************************************聚类参数设置***************************************************/
    float angle_thresh;
    float dis_thresh;
    if(name == "3dmatch" || name == "3dlomatch"){
        angle_thresh = 5.0 * M_PI / 180.0;
        dis_thresh = inlier_thresh;
    }
    else if(name == "U3M"){
        angle_thresh = 3.0 * M_PI / 180.0;
        dis_thresh = 5*resolution;
    }
    else if(name == "KITTI"){
        angle_thresh = 3.0 * M_PI / 180.0;
        dis_thresh = inlier_thresh;
    }
    else{
        cout << "not implement" << endl;
        exit(-1);
    }

/******************************************假设聚类***************************************************/
    pcl::IndicesClusters clusterTrans;
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr trans(new pcl::PointCloud<pcl::PointXYZINormal>);
    float eigenSimilarityScore = numeric_limits<float>::max();
    int similar2est1_cluster; //类号
    int similar2est1_ind;//类内号
    int best_index;

    clusterTransformationByRotation(Rs, Ts, angle_thresh, dis_thresh,  clusterTrans, trans);
    cout << "Total "<<max_num <<" cliques form "<< clusterTrans.size() << " clusters." << endl;
    //如果聚类失败的特殊处理，退化为标准MAC算法
    if(clusterTrans.size() ==0){
        Eigen::MatrixXf tmp_best;
        if (name == "U3M")
        {
            RE = RMSE_compute(cloud_src, cloud_des, best_est1, GTmat, resolution);
            TE = 0;
        }
        else {
            if (!found)
            {
                found = evaluation_est(best_est1, GTmat, RE_thresh, TE_thresh, RE, TE);
            }
            tmp_best = best_est1;
            best_score = 0;
            post_refinement(sampled_corr, sampled_corr_src, sampled_corr_des, best_est1, best_score, inlier_thresh, 20, metric);
        }
        if (name == "U3M")
        {
            if (RE <= 5)
            {
                cout << RE << endl;
                cout << best_est1 << endl;
                return true;
            }
            else {
                return false;
            }
        }
        else {
//            float rmse = RMSE_compute_scene(cloud_src, cloud_des, best_est1, GTmat, 0.0375);
//            cout << "RMSE: " << rmse <<endl;
            if (found) {
                float new_re, new_te;
                evaluation_est(best_est1, GTmat, RE_thresh, TE_thresh, new_re, new_te);

                if (new_re < RE && new_te < TE) {
                    cout << "est_trans updated!!!" << endl;
                    cout << "RE=" << new_re << " " << "TE=" << new_te << endl;
                    cout << best_est1 << endl;
                } else {
                    best_est1 = tmp_best;
                    cout << "RE=" << RE << " " << "TE=" << TE << endl;
                    cout << best_est1 << endl;
                }
                RE = new_re;
                TE = new_te;
//                if(rmse > 0.2) return false;
//                else return true;
                return true;
            } else {
                float new_re, new_te;
                found = evaluation_est(best_est1, GTmat, RE_thresh, TE_thresh, new_re, new_te);
                if (found) {
                    RE = new_re;
                    TE = new_te;
                    cout << "est_trans corrected!!!" << endl;
                    cout << "RE=" << RE << " " << "TE=" << TE << endl;
                    cout << best_est1 << endl;
                    return true;
                }
                else{
                    cout << "RE=" << RE << " " << "TE=" << TE << endl;
                    return false;
                }
//                if(rmse > 0.2) return false;
//                else return true;
            }
        }
    }

    //聚类排序,大聚类排在前面
    int goodClusterNum = 0;
    vector<Vote>sortCluster(clusterTrans.size());
    for(int i = 0; i < (int )clusterTrans.size(); i++){
        sortCluster[i].index = i;
        sortCluster[i].score = clusterTrans[i].indices.size();
        if(sortCluster[i].score > 1){
            goodClusterNum++;
        }
    }
    assert(goodClusterNum>0);
    sort(sortCluster.begin(), sortCluster.end(), compare_vote_score);

    //找出best_est1对应在哪个聚类中
    vector<Eigen::Matrix4f,aligned_allocator<Matrix4f>> est_trans2; //内存对齐
    vector<int>clusterIndexOfest2;
    vector<int>globalUnionInd;
#pragma omp parallel for
    for(int i = 0; i  < (int )sortCluster.size(); i++){
        int index = sortCluster[i].index;
        for(int j = 0; j < (int )clusterTrans[index].indices.size(); j++){
            int k = clusterTrans[index].indices[j];
            Eigen::Matrix3f R = Rs[k];
            Eigen::Vector3f T = Ts[k];
            Eigen::Matrix4f mat;
            mat.setIdentity();
            mat.block(0, 3, 3, 1) = T;
            mat.topLeftCorner(3,3) = R;
            float similarity = (best_est1.inverse() * mat - Eigen::Matrix4f::Identity(4, 4)).norm();
#pragma omp critical
            {
                if(similarity < eigenSimilarityScore){
                    eigenSimilarityScore = similarity;
                    similar2est1_ind = j;
                    similar2est1_cluster = index;
                }
            }
        }
    }
    cout << "Mat " << similar2est1_ind <<" in cluster " << similar2est1_cluster << " ("<< sortCluster[similar2est1_cluster].score << ") is similar to best_est1 with score " << eigenSimilarityScore <<endl;

    //对于每个聚类生成聚类中心，类匹配
    vector<vector<int>>subclusterinds;
#pragma omp parallel for
    for(int i = 0; i < (int )sortCluster.size(); i ++){
        //考察同一聚类的匹配
        vector<Corre_3DMatch>subClusterCorr;
        PointCloudPtr cluster_src_pts(new pcl::PointCloud<pcl::PointXYZ>);
        PointCloudPtr cluster_des_pts(new pcl::PointCloud<pcl::PointXYZ>);
        vector<int>subUnionInd;
        int index = sortCluster[i].index; //clusterTrans中的序号
        int k = clusterTrans[index].indices[0]; //初始聚类中心
        float cluster_center_score = scores[remained_est_ind[k]]; //初始聚类中心分数
        subUnionInd.assign(group_corr_ind[remained_est_ind[k]].begin(), group_corr_ind[remained_est_ind[k]].end());

        for(int j = 1; j < (int )clusterTrans[index].indices.size(); j ++){
            int m = clusterTrans[index].indices[j];
            float current_score = scores[remained_est_ind[m]]; //local score
            if (current_score > cluster_center_score){ //分数最高的设为聚类中心 8.10
                k = m;
                cluster_center_score = current_score;
            }
            subUnionInd = vectors_union(subUnionInd, group_corr_ind[remained_est_ind[m]]);
        }

        for (int l = 0; l < (int )subUnionInd.size(); ++l) {
            subClusterCorr.push_back(correspondence[subUnionInd[l]]);
            cluster_src_pts->push_back(correspondence[subUnionInd[l]].src);
            cluster_des_pts->push_back(correspondence[subUnionInd[l]].des);
        }
        Eigen::Matrix4f mat;
        mat.setIdentity();
        mat.block(0, 3, 3, 1) = Ts[k];
        mat.topLeftCorner(3,3) = Rs[k];
        //cout << "Cluster " << index << ", score " << cluster_center_score;
        //post_refinement(subClusterCorr, cluster_src_pts, cluster_des_pts, mat, cluster_center_score, inlier_thresh, 20, metric);
        //cout << ", after refine " << cluster_center_score << endl;
#pragma omp critical
        {
            globalUnionInd = vectors_union(globalUnionInd,subUnionInd);
            est_trans2.push_back(mat);
            subclusterinds.push_back(subUnionInd);
            clusterIndexOfest2.push_back(index);
        }
        subClusterCorr.clear();
        subUnionInd.clear();
    }

    vector<Corre_3DMatch>globalUnionCorr;
    PointCloudPtr globalUnionCorrSrc(new pcl::PointCloud<pcl::PointXYZ>);
    PointCloudPtr globalUnionCorrDes(new pcl::PointCloud<pcl::PointXYZ>);
    for(int i = 0; i < (int )globalUnionInd.size(); i++){
        globalUnionCorr.push_back(correspondence[globalUnionInd[i]]);
    }
    vector<pair<int, vector<int>>> des_src2;
    make_des_src_pair(globalUnionCorr, des_src2);

//找出best_est2 最好的聚类中心与其对应的类
    best_score = 0;
#pragma omp parallel for
    for(int i = 0; i < (int )est_trans2.size(); i++){
        double cluster_eva_score;
        //cluster_eva_score = OAMAE_1tok(cloud_src_kpts, cloud_des_kpts, est_trans2[i], one2k_match, inlier_thresh);
        cluster_eva_score = OAMAE(cloud_src_kpts, cloud_des_kpts, est_trans2[i], des_src2, inlier_thresh);
#pragma omp critical
        {
            if (best_score < cluster_eva_score) {
                best_score = cluster_eva_score;
                best_est2 = est_trans2[i];
                best_index = clusterIndexOfest2[i];
            }
        }
    }

    //按照clusterIndexOfest2 排序 subclusterinds
    indices.clear();
    for(int i = 0; i < (int )clusterIndexOfest2.size(); i++){
        indices.push_back(i);
    }
    sort(indices.begin(), indices.end(), [&clusterIndexOfest2](int a, int b){return clusterIndexOfest2[a] < clusterIndexOfest2[b];});
    vector<vector<int>> subclusterinds1;
    for(auto &ind : indices){
        subclusterinds1.push_back(subclusterinds[ind]);
    }
    subclusterinds.clear();
    subclusterinds.assign(subclusterinds1.begin(), subclusterinds1.end());
    subclusterinds1.clear();

    //输出每个best_est分别在哪个类
    if(best_index == similar2est1_cluster){
        cout << "Both choose cluster " << best_index << endl;
    }
    else{
        cout << "best_est1: " << similar2est1_cluster << ", best_est2: " << best_index << endl;
    }
    //1、sampled corr -> overlap prior batch -> TCD 确定best_est1和best_est2中最好的
    Eigen::Matrix4f best_est;
    PointCloudPtr sampled_src(new pcl::PointCloud<pcl::PointXYZ>); // dense point cloud
    PointCloudPtr sampled_des(new pcl::PointCloud<pcl::PointXYZ>);

    getCorrPatch(sampled_corr, cloud_src_kpts, cloud_des_kpts, sampled_src, sampled_des, 2*inlier_thresh);
    //点云patch后校验两个best_est
    float score1 = trancatedChamferDistance(sampled_src, sampled_des, best_est1, inlier_thresh);
    float score2 = trancatedChamferDistance(sampled_src, sampled_des, best_est2, inlier_thresh);
    vector<Corre_3DMatch>cluster_eva_corr;
    PointCloudPtr cluster_eva_corr_src(new pcl::PointCloud<pcl::PointXYZ>);
    PointCloudPtr cluster_eva_corr_des(new pcl::PointCloud<pcl::PointXYZ>);
    cout << "best_est1: " << score1 << ", best_est2: " << score2 << endl;

    ///TODO 测试点：trancatedChamferDistance是否误检
    //score2 = score1 + 1;

    // cluster_internal_evaluation
    if(cluster_internal_eva){
        if(eigenSimilarityScore < 0.1){ //best_est1在聚类中
            if(score1 > score2) { //best_est1好的情况
                best_index = similar2est1_cluster;
                best_est = best_est1;
                cout << "prior is better" << endl;
            }
            else { //best_est2好的情况
                best_est = best_est2;
                cout << "post is better" << endl;
            }
            //取匹配交集
            vector<int>cluster_eva_corr_ind;
            cluster_eva_corr_ind.assign(subclusterinds[best_index].begin(), subclusterinds[best_index].end());
            sort(cluster_eva_corr_ind.begin(), cluster_eva_corr_ind.end());
            sort(sampled_ind.begin(), sampled_ind.end());
            cluster_eva_corr_ind = vectors_intersection(cluster_eva_corr_ind, sampled_ind);
            if(!cluster_eva_corr_ind.size()){
                exit(-1);
            }
            num = 0;

            for(auto &ind : cluster_eva_corr_ind){
                cluster_eva_corr.push_back(correspondence[ind]);
                cluster_eva_corr_src->push_back(correspondence[ind].src);
                cluster_eva_corr_des->push_back(correspondence[ind].des);
                if(true_corre[ind]){
                    num++;
                }
            }
            ///TODO 注意这里的内点率要比seed内点率高
            cout << cluster_eva_corr_ind.size() << " intersection correspondences have " << num << " inlies: "<< num / ((int)cluster_eva_corr_ind.size() / 1.0) * 100 << "%" << endl;
            vector<pair<int, vector<int>>> des_src3;
            make_des_src_pair(cluster_eva_corr, des_src3);
            best_est = clusterInternalTransEva1(clusterTrans, best_index, best_est, Rs, Ts, cloud_src_kpts, cloud_des_kpts, des_src3, inlier_thresh, GTmat, false, folderPath);
            //1tok
            //best_est = clusterInternalTransEva1(clusterTrans, best_index, best_est, Rs, Ts, cloud_src_kpts, cloud_des_kpts, one2k_match, inlier_thresh, GTmat, folderPath);
        }
        else{ //best_est1不在聚类中
            if(score2 > score1){ //best_est2好的情况
                best_est = best_est2;
                cout << "post is better" << endl;
                vector<int>cluster_eva_corr_ind;
                cluster_eva_corr_ind.assign(subclusterinds[best_index].begin(), subclusterinds[best_index].end());
                sort(cluster_eva_corr_ind.begin(), cluster_eva_corr_ind.end());
                sort(sampled_ind.begin(), sampled_ind.end());
                cluster_eva_corr_ind = vectors_intersection(cluster_eva_corr_ind, sampled_ind);
                if(!cluster_eva_corr_ind.size()){
                    exit(-1);
                }
                num = 0;

                for(auto &ind : cluster_eva_corr_ind){
                    cluster_eva_corr.push_back(correspondence[ind]);
                    cluster_eva_corr_src->push_back(correspondence[ind].src);
                    cluster_eva_corr_des->push_back(correspondence[ind].des);
                    if(true_corre[ind]){
                        num++;
                    }
                }
                cout << cluster_eva_corr_ind.size() << " intersection correspondences have " << num << " inlies: "<< num / ((int)cluster_eva_corr_ind.size() / 1.0) * 100 << "%" << endl;
                vector<pair<int, vector<int>>> des_src3;
                make_des_src_pair(cluster_eva_corr, des_src3);
                best_est = clusterInternalTransEva1(clusterTrans, best_index, best_est, Rs, Ts, cloud_src_kpts, cloud_des_kpts, des_src3, inlier_thresh, GTmat, false, folderPath);
                //1tok
                //best_est = clusterInternalTransEva1(clusterTrans, best_index, best_est, Rs, Ts, cloud_src_kpts, cloud_des_kpts, des_src3, inlier_thresh, GTmat, folderPath);
            }
            else{ //仅优化best_est1
                best_index = -1; //不存在类中
                best_est = best_est1;
                cout << "prior is better but not in cluster! Refine est1" <<endl;
            }
        }
    }
    else{
        best_est = score1 > score2 ? best_est1 : best_est2;
    }

    end = std::chrono::system_clock::now();
    elapsed_time = end - start;
    time_epoch += std::chrono::duration_cast<std::chrono::milliseconds>(elapsed_time).count();
    time_number[3] =  elapsed_time.count();
    //cout << " post evaluation: " << elapsed_time.count() << endl;

    Eigen::Matrix4f tmp_best;
    if (name == "U3M")
    {
        RE = RMSE_compute(cloud_src, cloud_des, best_est, GTmat, resolution);
        TE = 0;
    }
    else {
        if (!found)
        {
            found = evaluation_est(best_est, GTmat, RE_thresh, TE_thresh, RE, TE);
        }
        tmp_best = best_est;
        best_score = 0;
        post_refinement(sampled_corr, sampled_corr_src, sampled_corr_des, best_est, best_score, inlier_thresh, 20, metric);

        vector<int> pred_inlier_index;
        PointCloudPtr src_trans(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::transformPointCloud(*src_corr_pts, *src_trans, best_est);
        int cnt = 0;
        int t = 0;
        for (int j = 0; j < correspondence.size(); j++)
        {
            double dist = Distance(src_trans->points[j], des_corr_pts->points[j]);
            if (dist < inlier_thresh){
                cnt ++;
                if (true_corre[j]){
                    t ++;
                }
            }
        }

        double IP = 0, IR = 0, F1 = 0;
        if(cnt > 0) IP = t / (cnt / 1.0);
        if(inlier_num > 0) IR = t / (inlier_num / 1.0);
        if( IP && IR){
            F1 = 2.0 / (1.0 /IP + 1.0 / IR);
        }
        cout << IP << " " << IR << " " << F1 << endl;
        pred_inlier.push_back(IP);
        pred_inlier.push_back(IR);
        pred_inlier.push_back(F1);

        //ICP
        if(use_icp){
            pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
            icp.setInputSource(cloud_src_kpts); //稀疏一些耗时小
            icp.setInputTarget(cloud_des);
            icp.setMaxCorrespondenceDistance(0.05);
            icp.setTransformationEpsilon(1e-10);
            icp.setMaximumIterations(50);
            icp.setEuclideanFitnessEpsilon(0.2);
            PointCloudPtr final(new pcl::PointCloud<pcl::PointXYZ>);
            icp.align(*final, best_est);
            if(icp.hasConverged()){
                best_est = icp.getFinalTransformation();
                cout << "ICP fitness score: " << icp.getFitnessScore() << endl;
            }
            else{
                cout << "ICP cannot converge!!!" << endl;
            }
        }
    }

    if(!no_logs){
        //保存匹配到txt
        //savetxt(correspondence, folderPath + "/corr.txt");
        //savetxt(selected, folderPath + "/selected.txt");
        string save_est = folderPath + "/est.txt";
        //string save_gt = folderPath + "/GTmat.txt";
        ofstream outfile(save_est, ios::trunc);
        outfile.setf(ios::fixed, ios::floatfield);
        outfile << setprecision(10) << best_est;
        outfile.close();
        //CopyFile(gt_mat.c_str(), save_gt.c_str(), false);
        //string save_label = folderPath + "/label.txt";
        //CopyFile(label_path.c_str(), save_label.c_str(), false);

        //保存ply
        //string save_src_cloud = folderPath + "/source.ply";
        //string save_tgt_cloud = folderPath + "/target.ply";
        //CopyFile(src_pointcloud.c_str(), save_src_cloud.c_str(), false);
        //CopyFile(des_pointcloud.c_str(), save_tgt_cloud.c_str(), false);
    }

    int pid = getpid();
    mem_epoch = getPidMemory(pid);

    //保存聚类信息
    string analyse_csv = folderPath + "/cluster.csv";
    string correct_csv = folderPath + "/cluster_correct.csv";
    string selected_csv = folderPath + "/cluster_selected.csv";
    ofstream outFile, outFile_correct, outFile_selected;
    outFile.open(analyse_csv.c_str(), ios::out);
    outFile_correct.open(correct_csv.c_str(), ios::out);
    outFile_selected.open(selected_csv.c_str(), ios::out);
    outFile.setf(ios::fixed, ios::floatfield);
    outFile_correct.setf(ios::fixed, ios::floatfield);
    outFile_selected.setf(ios::fixed, ios::floatfield);
    outFile << "x" << ',' << "y" << ',' << "z" << ',' << "r" << ',' << "g" << ',' << "b" << endl;
    outFile_correct << "x" << ',' << "y" << ',' << "z" << ',' << "r" << ',' << "g" << ',' << "b" << endl;
    outFile_selected << "x" << ',' << "y" << ',' << "z" << ',' << "r" << ',' << "g" << ',' << "b" << endl;
    for(int i = 0;i <(int)sortCluster.size(); i++){
        int r = rand() % 255;
        int g = rand() % 255;
        int b = rand() % 255;
        int cluster_id = sortCluster[i].index;
        for(int j = 0; j < (int)clusterTrans[cluster_id].indices.size(); j++){
            int id = clusterTrans[cluster_id].indices[j];
            if(est_vector[id].flag){
                outFile_correct << setprecision(4) << trans->points[id].x << ',' << trans->points[id].y << ',' << trans->points[id].z << ',' << r << ',' << g << ',' << b <<endl;
                //cout << "Correct est in cluster " << cluster_id << " (" << sortCluster[i].score << ")" << endl;
            }
            if(cluster_id == best_index) outFile_selected << setprecision(4) << trans->points[id].x << ',' << trans->points[id].y << ',' << trans->points[id].z << ',' << r << ',' << g << ',' << b <<endl;
            outFile << setprecision(4) << trans->points[id].x << ',' << trans->points[id].y << ',' << trans->points[id].z << ',' << r << ',' << g << ',' << b <<endl;
        }
    }
    outFile.close();
    outFile_correct.close();

    correspondence.clear();
    correspondence.shrink_to_fit();
    ov_corr_label.clear();
    ov_corr_label.shrink_to_fit();
    true_corre.clear();
    true_corre.shrink_to_fit();
    degree.clear();
    degree.shrink_to_fit();
    pts_degree.clear();
    pts_degree.shrink_to_fit();
    cluster_factor.clear();
    cluster_factor.shrink_to_fit();
    cluster_factor_bac.clear();
    cluster_factor_bac.shrink_to_fit();
    remain.clear();
    remain.shrink_to_fit();
    sampled_ind.clear();
    sampled_ind.shrink_to_fit();
    Rs.clear();
    Rs.shrink_to_fit();
    Ts.clear();
    Ts.shrink_to_fit();
    src_corr_pts.reset(new pcl::PointCloud<pcl::PointXYZ>);
    des_corr_pts.reset(new pcl::PointCloud<pcl::PointXYZ>);
    cloud_src.reset(new pcl::PointCloud<pcl::PointXYZ>);
    cloud_des.reset(new pcl::PointCloud<pcl::PointXYZ>);
    cloud_src_kpts.reset(new pcl::PointCloud<pcl::PointXYZ>);
    cloud_des_kpts.reset(new pcl::PointCloud<pcl::PointXYZ>);
    normal_src.reset(new pcl::PointCloud<pcl::Normal>);
    normal_des.reset(new pcl::PointCloud<pcl::Normal>);
    Raw_src.reset(new pcl::PointCloud<pcl::PointXYZ>);
    Raw_des.reset(new pcl::PointCloud<pcl::PointXYZ>);

    if (name == "U3M")
    {
        if (RE <= 5)
        {
            cout << RE << endl;
            cout << best_est << endl;
            return true;
        }
        else {
            return false;
        }
    }
    else {
        //float rmse = RMSE_compute_scene(cloud_src, cloud_des, best_est1, GTmat, 0.0375);
        //cout << "RMSE: " << rmse <<endl;
        if (found)
        {
            float new_re, new_te;
            evaluation_est(best_est, GTmat, RE_thresh, TE_thresh, new_re, new_te);
            if (new_re < RE && new_te < TE)
            {
                cout << "est_trans updated!!!" << endl;
                cout << "RE=" << new_re << " " << "TE=" << new_te << endl;
                cout << best_est << endl;
            }
            else {
                best_est = tmp_best;
                cout << "RE=" << RE << " " << "TE=" << TE << endl;
                cout << best_est << endl;
            }
            RE = new_re;
            TE = new_te;
//            if(rmse > 0.2){
//                return false;
//            }
//            else{
//                return true;
//            }
            return true;
        }
        else {
            float new_re, new_te;
            found = evaluation_est(best_est, GTmat, RE_thresh, TE_thresh, new_re, new_te);
            if (found)
            {
                RE = new_re;
                TE = new_te;
                cout << "est_trans corrected!!!" << endl;
                cout << "RE=" << RE << " " << "TE=" << TE << endl;
                cout << best_est << endl;
                return true;
            }
            else{
                cout << "RE=" << RE << " " << "TE=" << TE << endl;
                return false;
            }
//            if(rmse > 0.2){
//                return false;
//            }
//            else{
//                return true;
//            }
            //Corres_selected_visual(Raw_src, Raw_des, correspondence, resolution, 0.1, GTmat);
            //Corres_selected_visual(Raw_src, Raw_des, selected, resolution, 0.1, GTmat);
        }
    }
}

int main(int argc, char** argv){
    string datasetName(argv[1]);
    string src_cloud(argv[2]);
    string tgt_cloud(argv[3]);
    string corr_path(argv[4]);
    string gt_label_path(argv[5]);
    string gt_mat_path(argv[6]);
    string ov_label(argv[7]);
    string folderPath(argv[8]);
    string desc(argv[9]);
    Eigen::VectorXd time_his(10);
    Eigen::VectorXd nen_his(10);
    if (access(folderPath.c_str(), 0))
    {
        if (mkdir(folderPath.c_str(),S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) != 0) {
            cout << " 创建数据项目录失败 " << endl;
            exit(-1);
        }
    }
    int iterNum = 1;
    for(int i = 0; i < iterNum; i++){
        double time_epoch = 0.0, mem_epoch = 0.0;
        vector<double> time_number(4, 0);
        vector<double> pred_inlier;
        float re, te;
        int correct_est_num = 0;
        int inlier_num=0, total_num=0;
        bool success = registration(datasetName, src_cloud, tgt_cloud, corr_path, gt_label_path, ov_label, gt_mat_path, folderPath, desc, time_epoch, mem_epoch,time_number, re, te, correct_est_num, inlier_num, total_num, pred_inlier);
        ofstream out;
        if(success){
            string eva_result = folderPath + "/eva.txt";
            out.open(eva_result.c_str(), ios::out);
            out.setf(ios::fixed, ios::floatfield);
            out << setprecision(4) << re << " " << te << " " << correct_est_num << endl;
            out.close();
        }

        string info = folderPath + "/status.txt";
        out.open(info.c_str(), ios::out);
        out.setf(ios::fixed, ios::floatfield);
        out << setprecision(4) << time_epoch << " " << mem_epoch << " " << correct_est_num <<" " << inlier_num << " " << total_num  << " " << pred_inlier[0] << " " << pred_inlier[1] << " " << pred_inlier[2] << endl;
        out.close();
    }
}