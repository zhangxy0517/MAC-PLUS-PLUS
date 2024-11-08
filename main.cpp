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
int cnt;
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
    //scene_num.push_back(error_pair.size());
    vector<string>match_success_pair;
    int index = 1;
    RE = 0;
    TE = 0;
    success_estimate_rate = 0;
    vector<double>time;
    vector<int>clique_size;
    cnt = 0;
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

        int inlier_num = 0, total_num = 0;
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
        int corrected=0;
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
            corrected=1;
        }

        string status = result_folder + "/status.txt";
        double mem_epoch, time_epoch;
        double IP, IR, F1;
        int correct_est_num;
        FILE *f = fopen(status.c_str(), "r");
        fscanf(f, "%lf %lf %d %d %d %lf %lf %lf", &time_epoch, &mem_epoch, &correct_est_num, &inlier_num, &total_num, &IP, &IR, &F1);
        fclose(f);

        cout << endl;
        outfile << pair << ',' << corrected << ',' << inlier_num << ',' << total_num << ',';
        outfile << setprecision(4) << inlier_num / (total_num / 1.0) << ',' << re << ',' << te << ','  << time_epoch << ',' << mem_epoch << ',' << correct_est_num  << ',' << IP << ',' << IR << ',' << F1 <<endl;


        cout << endl;
        cnt ++;
    }
    scene_num.push_back(cnt);
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

void usage(){
    cout << "Usage:" << endl;
    cout << "\tHELP --help" <<endl;
    cout << "\tDEMO --demo" << endl;
    cout << "\tREQUIRED ARGS:" << endl;
    cout << "\t\t--output_path\toutput path for saving results." << endl;
    cout << "\t\t--input_path\tinput data path." << endl;
    cout << "\t\t--dataset_name\tdataset name. [3dmatch/3dlomatch/KITTI/ETH/U3M]" << endl;
    cout << "\t\t--descriptor\tdescriptor name. [fpfh/fcgf/spinnet/predator]" << endl;
    cout << "\t\t--start_index\tstart from given index. (begin from 0)" << endl;
    cout << "\tOPTIONAL ARGS:" << endl;
    cout << "\t\t--no_logs\tforbid generation of log files." << endl;
};

int main(int argc, char** argv) {
//    demo();
//    exit(0);
    //////////////////////////////////////////////////////////////////
    add_overlap = false;
    low_inlieratio = false;
    no_logs = false;
    int id = 0;
    string resultPath; //程序生成文件的保存目录
    string datasetPath; //数据集路径
    string datasetName; //数据集名称
    string descriptor; //描述子
    //////////////////////////////////////////////////////////////////
    int opt;
    int digit_opind = 0;
    int option_index = 0;
    static struct option long_options[] = {
            {"output_path", required_argument, NULL, 'o'},
            {"input_path", required_argument, NULL, 'i'},
            {"dataset_name", required_argument, NULL, 'n'},
            {"descriptor", required_argument, NULL, 'd'},
            {"start_index", required_argument, NULL, 's'},
            {"no_logs", optional_argument, NULL, 'g'},
            {"help", optional_argument, NULL, 'h'},
            {"demo", optional_argument, NULL, 'm'},
            {NULL, 0, 0, '\0'}
    };

    while((opt = getopt_long(argc, argv, "", long_options, &option_index)) != -1){
        switch (opt) {
            case 'h':
                usage();
                exit(0);
            case 'o':
                resultPath = optarg;
                break;
            case 'i':
                datasetPath = optarg;
                break;
            case 'n':
                datasetName = optarg;
                break;
            case 'd':
                descriptor = optarg;
                break;
            case 'g':
                no_logs = true;
                break;
            case 's':
                id = atoi(optarg);
                break;
            case 'm':
                demo();
                exit(0);
            case '?':
                printf("Unknown option: %c\n",(char)optopt);
                usage();
                exit(-1);
        }
    }
    if(argc  < 11){
        cout << 11 - argc <<" more args are required." << endl;
        usage();
        exit(-1);
    }

    cout << "Check your args setting:" << endl;
    cout << "\toutput_path: " << resultPath << endl;
    cout << "\tinput_path: " << datasetPath << endl;
    cout << "\tdataset_name: " << datasetName << endl;
    cout << "\tdescriptor: " << descriptor << endl;
    cout << "\tstart_index: " << id << endl;
    cout << "\tno_logs: " << no_logs << endl;

    sleep(5);
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
        string analyse_csv = resultPath + "/" + datasetName + "_predator.csv";
        ofstream outFile;
        outFile.open(analyse_csv.c_str(), ios::out);
        outFile.setf(ios::fixed, ios::floatfield);
        outFile << "pair_name" << ',' << "corrected_or_no" << ',' << "inlier_num" << ',' << "total_num" << ','
                << "inlier_ratio" << ',' << "RE" << ',' << "TE" << endl;
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
            string corr_path = datasetPath + "/" + filename + "/corr_data.txt";
            string gt_mat_path = datasetPath + "/" + filename + "/GTmat.txt";
            string gt_label_path = datasetPath + "/" + filename + "/label.txt";
            string src_filename = "NULL";
            string des_filename = "NULL";
            string ov_label = "NULL";
            if(add_overlap){
                ov_label = datasetPath + "/" + filename + "@gt_ov.txt";
            }
            folderPath = resultPath + "/" + filename;
            string cmd = program_name + " " + datasetName + " " + src_filename + " " + des_filename + " " + corr_path + " " + gt_label_path + " " + gt_mat_path + " " + ov_label + " " + folderPath +  " " + descriptor;
            system(cmd.c_str());

            int inlier_num =0 , total_num  =0;
            double re=0, te=0;
            string check_file = folderPath + "/eva.txt";
            int corrected = 0;
            if (access(check_file.c_str(), 0))
            {
                cout << check_file << " Fail." << endl;
                corrected = 0;
            }
            else
            {
                FILE *f = fopen(check_file.c_str(), "r");
                fscanf(f, "%lf %lf", &re, &te);
                fclose(f);
                //if(re <= re_thresh && te <= te_thresh){
                cout << filename << " Success." << endl;
                corrected = 1;
                RE += re;
                TE += te;

            }

            //info
            string status = folderPath + "/status.txt";
            double mem_epoch, time_epoch;
            vector<double> time_number(4, 0);
            int correct_est_num;
            FILE *f = fopen(status.c_str(), "r");
            fscanf(f, "%lf %lf %d %d %d %lf %lf %lf %lf", &time_epoch, &mem_epoch, &correct_est_num, &inlier_num, &total_num, &time_number[0], &time_number[1], &time_number[2], &time_number[3]);
            fclose(f);

            cout << endl;
            outFile << filename << ',' << corrected << ',' << inlier_num << ',' << total_num << ',';
            outFile << setprecision(4) << inlier_num / (total_num / 1.0) << ',' << re << ',' << te << ','  << time_epoch << ',' << mem_epoch << ',' << correct_est_num << ',';
            outFile << setprecision(4) << time_number[0] << ',' << time_number[1] << ',' << time_number[2] << ',' << time_number[3] << endl;
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

        outFile.close();
    }
    else if (datasetName == "3dlomatch") {
        for (size_t i = id; i < 8; i++) {
            string analyse_csv = resultPath + "/" + threeDlomatch[i] + "_" + descriptor + ".csv";
            ofstream outFile;
            outFile.open(analyse_csv.c_str(), ios::out);
            outFile.setf(ios::fixed, ios::floatfield);
            outFile << "pair_name" << ',' << "corrected_or_no" << ',' << "inlier_num" << ',' << "total_num" << ','
                    << "inlier_ratio" << ',' << "RE" << ',' << "TE" << ',' << "time" << ',' << "mem" << ',' << "IP" << ',' << "IR" << ',' << "F1" <<endl;
            vector<string>matched = analyse("3dlomatch", resultPath + "/" + threeDlomatch[i],datasetPath + "/" + threeDlomatch[i], descriptor, outFile, id, i);
            scene_re_sum.push_back(RE);
            scene_te_sum.push_back(TE);

            cout << endl;
            cout << threeDlomatch[i] << ":" << endl;
            cout << endl;
            cout << threeDlomatch[i] << ":" << matched.size() / (scene_num[i] / 1.0) << endl;
            cout << "RE:" << RE / matched.size() << "\tTE:" << TE / matched.size() << endl;
            corrected += matched.size();
            scene_correct_num.push_back(matched.size());
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
    else if (datasetName == "3dmatch"){
        for (size_t i = id; i < 8; i++) {
            string analyse_csv = resultPath + "/" + threeDMatch[i] + "_" + descriptor + ".csv";
            ofstream outFile;
            outFile.open(analyse_csv.c_str(), ios::out);
            outFile.setf(ios::fixed, ios::floatfield);
            outFile << "pair_name" << ',' << "corrected_or_no" << ',' << "inlier_num" << ',' << "total_num" << ','
                    << "inlier_ratio" << ',' << "RE" << ',' << "TE" << ',' << "time" << ',' << "mem" << ',' << "IP" << ',' << "IR" << ',' << "F1" <<endl;
            vector<string>matched = analyse("3dmatch", resultPath + "/" + threeDMatch[i],datasetPath + "/" + threeDMatch[i], descriptor, outFile, id, i);
            scene_re_sum.push_back(RE);
            scene_te_sum.push_back(TE);
            cout << endl;
            cout << threeDMatch[i] << ":" << endl;
            cout << endl;
            cout << threeDMatch[i] << ":" << matched.size() / (scene_num[i] / 1.0) << endl;
            cout << "RE:" << RE / matched.size() << "\tTE:" << TE / matched.size() << endl;
            corrected += matched.size();
            scene_correct_num.push_back(matched.size());
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
    else if (datasetName == "ETH")
    {
        for (size_t i = id; i < 4; i++)
        {
            cout << i + 1 << ":" << ETH[i] << endl;
            string analyse_csv = resultPath + "/" + ETH[i] + "_" + descriptor + ".csv";
            ofstream outFile;
//            outFile.open(analyse_csv.c_str(), ios::out);
//            outFile.setf(ios::fixed, ios::floatfield);
//            outFile << "pair_name" << ',' << "corrected_or_no" << ',' << "inlier_num" << ',' << "total_num" << ',' << "inlier_ratio" << ',' << "est_rr" << ',' << "RE" << ',' << "TE" << ',' << "construction" << ',' << "search" << ',' << "selection" << ',' << "estimation" << endl;
            vector<string>matched = analyse("3dmatch", resultPath + "/" + ETH[i], datasetPath + "/" + ETH[i], descriptor, outFile, id, i);
            scene_re_sum.push_back(RE);
            scene_te_sum.push_back(TE);
            if (!matched.empty())
            {
                cout << endl;
                cout << ETH[i] << ":" << endl;
                for (auto t : matched)
                {
                    cout << "\t" << t << endl;
                }
                cout << endl;
                cout << ETH[i] << ":" << matched.size() << endl;
                cout << "success_est_rate:" << success_estimate_rate / (scene_num[i] / 1.0) << "RE:" << RE / matched.size() << "\tTE:" << TE / matched.size() << endl;
                corrected += matched.size();
                //total_success_est_rate.push_back(success_estimate_rate);
                scene_correct_num.push_back(matched.size());
            }
            outFile.close();
            matched.clear();
        }
        string detail_txt = resultPath + "/details.txt";
        ofstream outFile;
        outFile.open(detail_txt.c_str(), ios::out);
        outFile.setf(ios::fixed, ios::floatfield);
        for (size_t i = 0; i < 4; i++)
        {
            total_num += scene_num[i];
            total_re += scene_re_sum[i];
            total_te += scene_te_sum[i];
            cout << i + 1 << ":" << endl;
            outFile << i + 1 << ":" << endl;
            cout << "\tRR: " << scene_correct_num[i] << "/" << scene_num[i] << " " << scene_correct_num[i] / (scene_num[i] / 1.0) << endl;
            outFile << "\tRR: " << scene_correct_num[i] << "/" << scene_num[i] << " " << setprecision(4) << scene_correct_num[i] / (scene_num[i] / 1.0) << endl;
           // cout << "\tSuccess_est_rate: " << total_success_est_rate[i] / (scene_num[i] / 1.0) << endl;
            cout << "\tRE: " << scene_re_sum[i] / (scene_correct_num[i] / 1.0) << endl;
            outFile << "\tRE: " << setprecision(4) << scene_re_sum[i] / (scene_correct_num[i] / 1.0) << endl;
            cout << "\tTE: " << scene_te_sum[i] / (scene_correct_num[i] / 1.0) << endl;
            outFile << "\tTE: " << setprecision(4) << scene_te_sum[i] / (scene_correct_num[i] / 1.0) << endl;
        }
        cout << "total:" << endl;
        outFile << "total:" << endl;
        cout << "\tRR: " << corrected / (total_num / 1.0) << endl;
        outFile << "\tRR: " << setprecision(4) << corrected / (total_num / 1.0) << endl;
        //cout << "\tSuccess_est_rate: " << accumulate(total_success_est_rate.begin(), total_success_est_rate.end(), 0.0) / (total_num / 1.0) << endl;
        cout << "\tRE: " << total_re / (corrected / 1.0) << endl;
        outFile << "\tRE: " << setprecision(4) << total_re / (corrected / 1.0) << endl;
        cout << "\tTE: " << total_te / (corrected / 1.0) << endl;
        outFile << "\tTE: " << setprecision(4) << total_te / (corrected / 1.0) << endl;
        outFile.close();
    }
    else if (datasetName == "KITTI"){
        int pair_num = 1260;
        //string txt_path = datasetPath + "/" + descriptor;
        const string& txt_path = datasetPath;
        string analyse_csv = resultPath + "/KITTI_" + descriptor + ".csv";
        ofstream outFile;
        outFile.open(analyse_csv.c_str(), ios::out);
        outFile.setf(ios::fixed, ios::floatfield);
        outFile << "pair_name" << ',' << "corrected_or_no" << ',' << "inlier_num" << ',' << "total_num" << ',' << "inlier_ratio" << ',' << "RE" << ',' << "TE" << endl;
        vector<string>fail_pair;
        vector<double>time;
        for (int i = id; i < pair_num; i++)
        {
            time.clear();
            cout << "Pair " << i + 1 << "，total" << pair_num/*name_list.size()*/ << "，fail " << fail_pair.size() << endl;

            string filename = to_string(i);/*name_list[i]*/;
            string corr_path = txt_path + "/" + filename + '/' + descriptor + "@corr.txt";
            string gt_mat_path = txt_path + "/" + filename + '/' + descriptor + "@gtmat.txt";
            string gt_label_path = txt_path + "/" + filename + '/' + descriptor + "@gtlabel.txt";
            string src_filename = "NULL";
            string des_filename = "NULL";
            string ov_label = "NULL";
            if(add_overlap){
                ov_label = datasetPath + "/" + filename + "@gt_ov.txt";
            }
            folderPath = resultPath + "/" + filename;

            double re, te;
            double inlier_ratio, success_estimate, total_estimate;

            string cmd = program_name + " " + datasetName + " " + src_filename + " " + des_filename + " " + corr_path + " " + gt_label_path + " " + gt_mat_path + " " + ov_label + " " + folderPath +  " " + descriptor;
            system(cmd.c_str());

            int inlier_num =0 , total_num  =0;
            string check_file = folderPath + "/eva.txt";
            int corrected = 0;
            if (access(check_file.c_str(), 0))
            {
                cout << check_file << " Fail." << endl;
                corrected = 0;
                fail_pair.push_back(filename);
            }
            else
            {
                FILE *f = fopen(check_file.c_str(), "r");
                fscanf(f, "%lf %lf", &re, &te);
                fclose(f);
                //if(re <= re_thresh && te <= te_thresh){
                cout << filename << " Success." << endl;
                corrected = 1;
                RE += re;
                TE += te;
            }

            //info
            string status = folderPath + "/status.txt";
            double mem_epoch, time_epoch;
            vector<double> time_number(4, 0);
            int correct_est_num;
            FILE *f = fopen(status.c_str(), "r");
            fscanf(f, "%lf %lf %d %d %d %lf %lf %lf %lf", &time_epoch, &mem_epoch, &correct_est_num, &inlier_num, &total_num, &time_number[0], &time_number[1], &time_number[2], &time_number[3]);
            fclose(f);

            cout << endl;
            outFile << filename << ',' << corrected << ',' << inlier_num << ',' << total_num << ',';
            outFile << setprecision(4) << inlier_num / (total_num / 1.0) << ',' << re << ',' << te << ','  << time_epoch << ',' << mem_epoch << ',' << correct_est_num << ',';
            outFile << setprecision(4) << time_number[0] << ',' << time_number[1] << ',' << time_number[2] << ',' << time_number[3] << endl;
        }
        outFile.close();
        double success_num = pair_num - fail_pair.size();
        cout << "total:" << endl;
        cout << "\tRR:" << pair_num - fail_pair.size() << "/" << pair_num << " " << success_num / (pair_num / 1.0) << endl;
        cout << "\tRE:" << RE / (success_num / 1.0) << endl;
        cout << "\tTE:" << TE / (success_num / 1.0) << endl;
        cout << "fail pairs:" << endl;
        for (size_t i = 0; i < fail_pair.size(); i++)
        {
            cout << "\t" << fail_pair[i] << endl;
        }
    }
    else{
        exit(-1);
    }
    return 0;
}