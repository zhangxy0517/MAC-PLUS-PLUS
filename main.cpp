#include <pcl/io/ply_io.h>
#include<iostream>
#include <string>
#include <sys/stat.h>  // 包含文件权限定义
#include <cstdlib>     // 包含exit()函数
#include <getopt.h>
#include <unistd.h>
#include <iomanip>  // 添加此头文件以使用 setprecision
#include <fstream>  // 添加此头文件以使用 ifstream


using namespace std;
string folderPath;
bool add_overlap;
bool low_inlieratio;
bool no_logs;

string program_name = "./MAC";

string threeDMatch[8] = {
        "7-scenes-redkitchen",
        "sun3d-home_at-home_at_scan1_2013_jan_1",
        "sun3d-home_md-home_md_scan9_2012_sep_30",
        "sun3d-hotel_uc-scan3",
        "sun3d-hotel_umd-maryland_hotel1",
        "sun3d-hotel_umd-maryland_hotel3",
        "sun3d-mit_76_studyroom-76-1studyroom2",
        "sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika",
};

string threeDlomatch[8] = {
        "7-scenes-redkitchen_3dlomatch",
        "sun3d-home_at-home_at_scan1_2013_jan_1_3dlomatch",
        "sun3d-home_md-home_md_scan9_2012_sep_30_3dlomatch",
        "sun3d-hotel_uc-scan3_3dlomatch",
        "sun3d-hotel_umd-maryland_hotel1_3dlomatch",
        "sun3d-hotel_umd-maryland_hotel3_3dlomatch",
        "sun3d-mit_76_studyroom-76-1studyroom2_3dlomatch",
        "sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika_3dlomatch",
};

string ETH[4] = {
        "gazebo_summer",
        "gazebo_winter",
        "wood_autmn",
        "wood_summer",
};


double RE, TE, success_estimate_rate;
vector<int>scene_num;
vector<string> analyse(const string& name, const string& result_scene, const string& dataset_scene, const string& descriptor, ofstream& outfile, int iters, int data_index) {
    if (!no_logs && access(result_scene.c_str(), 0))
    {
        if (mkdir(result_scene.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH)!=0) {
            cout << " Create scene folder failed! " << endl;
            exit(-1);
        }
    }
    vector<string>error_pair;

    //错误数据上测试
    string error_txt;
    //error_txt = result_scene + "/error_pair.txt";

    //所有数据上测试
    if (descriptor == "fpfh" || descriptor == "spinnet" || descriptor == "d3feat")
    {
        error_txt = dataset_scene + "/dataload.txt";
    }
    else if (descriptor == "fcgf")
    {
        error_txt = dataset_scene + "/dataload_fcgf.txt";
    }
    if (access(error_txt.c_str(), 0))
    {
        cout << " Could not find dataloader file! " << endl;
        exit(-1);
    }

    ifstream f1(error_txt);
    string line;
    while (getline(f1, line))
    {
        error_pair.push_back(line);
    }
    f1.close();
    scene_num.push_back(error_pair.size());
    vector<string>match_success_pair;
    int index = 1;
    RE = 0;
    TE = 0;
    success_estimate_rate = 0;
    vector<double>time;
    vector<int>clique_size;
    for (const auto& pair : error_pair)
    {
        time.clear();
        clique_size.clear();
        cout << "Pair " << index << ", " << "total " << error_pair.size() << " pairs." << endl;
        index++;
        string result_folder = result_scene + "/" + pair;
        string::size_type i = pair.find("+") + 1;
        string src_filename = dataset_scene + "/" + pair.substr(0, i - 1) + ".ply";
        string des_filename = dataset_scene + "/" + pair.substr(i, pair.length() - i) + ".ply";
        //cout << src_filename << " " << des_filename << endl;
        string corr_path = dataset_scene + "/" + pair + (descriptor == "fcgf" ? "@corr_fcgf.txt" : "@corr.txt");
        string gt_label = dataset_scene + "/" + pair + (descriptor == "fcgf" ? "@label_fcgf.txt" : "@label.txt");
        string gt_mat_path = dataset_scene + "/" + pair + (descriptor == "fcgf" ? "@GTmat_fcgf.txt" : "@GTmat.txt");

        //调用 源.cpp
        string ov_label = "NULL";
        //string ov_label = dataset_scene + "/" + pair + "@gt_ov.txt";

        //float re, te;
        //double inlier_num, total_num, inlier_ratio, success_estimate, total_estimate;
        //int corrected = registration(name, src_filename, des_filename, corr_path, gt_label, ov_label, gt_mat_path, result_folder, re, te, inlier_num, total_num, inlier_ratio, success_estimate, total_estimate, descriptor, time, clique_size);
        string cmd = program_name + " " + name + " " + src_filename + " " + des_filename + " " + corr_path + " " + gt_label + " " + gt_mat_path + " " + ov_label + " " + result_folder +  " " + descriptor;
        system(cmd.c_str());

        double re=0, te=0;
        string check_file = result_folder + "/eva.txt";
        if (access(check_file.c_str(), 0))
        {
            cout << pair << " Fail." << endl;
        }
        else
        {
            cout << pair << " Success." << endl;
            FILE *f = fopen(check_file.c_str(), "r");
            fscanf(f, "%lf %lf", &re, &te);
            fclose(f);
            //cout << re << " " << te <<endl;
            RE += re;
            TE += te;
            match_success_pair.push_back(pair);
        }
        cout << endl;
    }
    error_pair.clear();
    return match_success_pair;
}

void demo(){
    string datasetName = "3dmatch";
    string src_filename = "demo/src.ply";
    string des_filename = "demo/tgt.ply";
    string corr_path = "demo/corr.txt";
    string gt_label_path = "demo/label.txt";
    string descriptor = "fpfh";
    string gt_mat_path = "demo/GTmat.txt";
    string ov_label = "NULL";
    string result_folder = "demo/result";
    string cmd = program_name + " " + datasetName + " " + src_filename + " " + des_filename + " " + corr_path + " " + gt_label_path + " " + gt_mat_path + " " + ov_label + " " + result_folder +  " " + descriptor;
    cout << cmd << endl;
    system(cmd.c_str());
}

int main(int argc, char** argv) {
    demo();
    //////////////////////////////////////////////////////////////////
    add_overlap = false;
    low_inlieratio = false;
    no_logs = false;
    int id = 0;
    string resultPath(argv[1]); //程序生成文件的保存目录
    string datasetPath(argv[2]); //数据集路径
    string datasetName(argv[3]); //数据集名称
    string descriptor(argv[4]); //描述子
    sscanf(argv[5], "%d", &id);
    //////////////////////////////////////////////////////////////////
    vector<double>scene_re_sum;
    vector<double>scene_te_sum;
    int corrected = 0;
    vector<int> scene_correct_num;
    int total_num = 0;
    double total_re = 0;
    double total_te = 0;
    if (access(resultPath.c_str(), 0))
    {
        if (mkdir(resultPath.c_str(),S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) != 0) {
            cout << " Create save folder failed! " << endl;
            exit(-1);
        }
    }

    if (descriptor == "predator" && (datasetName == "3dmatch" || datasetName == "3dlomatch")) {
        vector<string>pairs;
        string loader = datasetPath + "/dataload.txt";
        cout << loader << endl;
        ifstream f1(loader);
        string line;
        while (getline(f1, line))
        {
            pairs.push_back(line);
        }
        f1.close();
        vector<double>time;
        for (int i = id; i < pairs.size(); i++)
        {
            cout << "Pair " << i + 1 << "，total" << pairs.size()/*name_list.size()*/ << endl;
            string filename = pairs[i];
            string corr_path = datasetPath + "/" + filename + "@corr.txt";
            string gt_mat_path = datasetPath + "/" + filename + "@GTmat.txt";
            string gt_label_path = datasetPath + "/" + filename + "@label.txt";
            string ov_label = "NULL";
            if(add_overlap){
                ov_label = datasetPath + "/" + filename + "@gt_ov.txt";
            }
            folderPath = resultPath + "/" + filename;
            string cmd = program_name + " " + datasetName + " " + corr_path + " " + gt_label_path + " " + gt_mat_path + " " + ov_label + " " + folderPath +  " " + descriptor;
            system(cmd.c_str());
            cout << endl;
        }
        int iter_num = 1;
        cout << "Avg Time:" << endl;
        for(int i = 0; i < iter_num; i++){
            double avg_time = 0.0;
            int n2 = 0;
            vector<double>time_history;
            for(int j = 0; j < pairs.size(); j++){
                string filename = pairs[i];
                folderPath = resultPath + "/" + filename;
                string time_info = folderPath + '/' + to_string(i) +  "@time.txt";
                FILE *in = fopen(time_info.c_str(), "r");
                if(in == NULL){
                    continue;
                }
                n2++;
                double time;
                fscanf(in, "%lf\n", &time);
                avg_time+=time;
                fclose(in);
            }
            avg_time /= (n2 / 1.0);
            cout << avg_time << endl;
        }
    }
    else if (datasetName == "3dlomatch") {
        for (size_t i = id; i < 8; i++) {
            string analyse_csv = resultPath + "/" + threeDlomatch[i] + "_" + descriptor + ".csv";
            ofstream outFile;
//            outFile.open(analyse_csv.c_str(), ios::out);
//            outFile.setf(ios::fixed, ios::floatfield);
//            outFile << "pair_name" << ',' << "corrected_or_no" << ',' << "inlier_num" << ',' << "total_num" << ','
//                    << "inlier_ratio" << ',' << "est_rr" << ',' << "RE" << ',' << "TE" << endl;
            vector<string>matched = analyse("3dlomatch", resultPath + "/" + threeDlomatch[i],datasetPath + "/" + threeDlomatch[i], descriptor, outFile, id, i);
            scene_re_sum.push_back(RE);
            scene_te_sum.push_back(TE);
            if (!matched.empty())
            {
                cout << endl;
                cout << threeDlomatch[i] << ":" << endl;
                for (auto t : matched)
                {
                    cout << "\t" << t << endl;
                }
                cout << endl;
                cout << threeDlomatch[i] << ":" << matched.size() / (scene_num[i] / 1.0) << endl;
                cout << "RE:" << RE / matched.size() << "\tTE:" << TE / matched.size() << endl;
                corrected += matched.size();
                scene_correct_num.push_back(matched.size());
            }
            outFile.close();
            matched.clear();
        }
        string detail_txt = resultPath + "/details.txt";
        ofstream outFile;
        outFile.open(detail_txt.c_str(), ios::out);
        outFile.setf(ios::fixed, ios::floatfield);
        for (size_t i = 0; i < 8; i++)
        {
            total_num += scene_num[i];
            total_re += scene_re_sum[i];
            total_te += scene_te_sum[i];
            cout << i + 1 << ":" << endl;
            outFile << i + 1 << ":" << endl;
            cout << "\tRR: " << scene_correct_num[i] << "/" << scene_num[i] << " " << scene_correct_num[i] / (scene_num[i] / 1.0) << endl;
            outFile << "\tRR: " << scene_correct_num[i] << "/" << scene_num[i] << " " << setprecision(4) << scene_correct_num[i] / (scene_num[i] / 1.0) << endl;
            cout << "\tRE: " << scene_re_sum[i] / (scene_correct_num[i] / 1.0) << endl;
            outFile << "\tRE: " << setprecision(4) << scene_re_sum[i] / (scene_correct_num[i] / 1.0) << endl;
            cout << "\tTE: " << scene_te_sum[i] / (scene_correct_num[i] / 1.0) << endl;
            outFile << "\tTE: " << setprecision(4) << scene_te_sum[i] / (scene_correct_num[i] / 1.0) << endl;
        }
        cout << "total:" << endl;
        outFile << "total:" << endl;
        cout << "\tRR: " << corrected / (total_num / 1.0) << endl;
        outFile << "\tRR: " << setprecision(4) << corrected / (total_num / 1.0) << endl;
        cout << "\tRE: " << total_re / (corrected / 1.0) << endl;
        outFile << "\tRE: " << setprecision(4) << total_re / (corrected / 1.0) << endl;
        cout << "\tTE: " << total_te / (corrected / 1.0) << endl;
        outFile << "\tTE: " << setprecision(4) << total_te / (corrected / 1.0) << endl;
        outFile.close();
    }
    else{
        exit(-1);
    }
    return 0;
}