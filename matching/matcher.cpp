/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

#include <assert.h>

#include <fstream>
#include <iomanip>
#include <algorithm>
#include <tuple>
#include <vector>
#include <math.h>
#include <chrono>
#include <stdlib.h>
#include "matcher.h"
#include <Eigen/Dense>
//delete this probably
#include <ctime>
#include "boost/filesystem.hpp"
namespace fs = boost::filesystem;

using namespace std;
using namespace Eigen;

namespace PQ
{


Matcher::Matcher(string code_file)
{
    N = 200;
    time.resize(10);
    nrof_matching = 0;
    codewords = NULL;
    description.push_back("minutiae similarity");

    description.push_back("obtaining corr");

    description.push_back("second order fast");

    description.push_back("second order original");

    dist_N = 50;
    table_dist.resize(dist_N*dist_N);
    max_nrof_templates = 0;
    int n = 0;
    for(int i=0;i<dist_N; ++i)
    {
        for(int j=i;j<dist_N;++j)
        {
            table_dist[i*dist_N+j] = sqrt((i*16.0)*(i*16.0) +  (j*16.0)*(j*16.0));
            table_dist[j*dist_N+i] = table_dist[i*dist_N+j];
        }
    }
    //    load code book
    ifstream is;
    is.open(code_file, ifstream::binary);
    // get length of file:
    is.seekg(0, ios::end);
    int length = is.tellg();

    if( length<=0 )
    {
        cout<<"codebook is empty!"<<endl;
    }
    is.seekg(0, ios::beg);

    nrof_subs = 0;
    nrof_clusters=0;
    sub_dim = 0;

    is.read(reinterpret_cast<char*>(&nrof_subs),sizeof(short));
    is.read(reinterpret_cast<char*>(&nrof_clusters),sizeof(short));
    is.read(reinterpret_cast<char*>(&sub_dim),sizeof(short));

    int len = nrof_subs*nrof_clusters*sub_dim;
    if(len<=0)
    {
        cout<<"codebook is empty!"<<endl;
//        return -1;
    }

    codewords = new float[len];
    float *pword = codewords;
    for(int i=0;i<nrof_subs; ++i)
    {
        for(int j=0; j<nrof_clusters; ++j)
        {
            is.read(reinterpret_cast<char*>(pword),sizeof(float)*sub_dim);
            pword += sub_dim;
        }
    }
}

int Matcher::List2List_matching(string latent_path, string rolled_path, string score_path)
{
    string template_file, score_file;
    //ifstream is;

    // //load score files
    // vector<string> score_files;
    // is.open(score_list_file, ifstream::binary);
    // // get length of file:
    // is.seekg(0, ios::end);
    // int length = is.tellg();

    // if( length<=0 )
    // {
    //     cout<<"score file list is empty!"<<endl;
    //     return -1;
    // }
    // is.seekg(0, ios::beg);
    // int nrof_score_files = 0;
    // while (!is.eof())
    // {
    //     getline(is, score_file);
    //     if(score_file.length()<10)
    //         break;
    //     score_files.push_back(score_file);
    //     cout<<"score file"<<score_file<<endl;
    //     ++nrof_score_files;
    // }
    // is.close();

    // // get latent original template filenames
    // is.open(latent_list_file, ifstream::binary);
    // // get length of file:
    // is.seekg(0, ios::end);
    // length = is.tellg();
    // if( length<=0 )
    // {
    //     cout<<"latent template list is empty!"<<endl;
    //     return -1;
    // }
    // is.seekg(0, ios::beg);
    // int nrof_latents = 0;
    // vector<string> latent_template_files;
    // while (!is.eof())
    // {
    //     getline(is, template_file);
    //     if(template_file.length()<10)
    //         break;
    //     latent_template_files.push_back(template_file);
    //     cout<<"template_file"<<template_file<<endl;
    //     ++nrof_latents;
    // }
    // is.close();

    int nrof_latents = 0;
    vector<fs::path> latent_template_files;
    fs::directory_iterator end_itr;
    for(fs::directory_iterator dir_itr(latent_path); dir_itr != end_itr; ++dir_itr)
    {
        if(dir_itr->path().extension() == ".dat")
        { 
            latent_template_files.push_back(dir_itr->path());
            cout<<"latent template file"<<dir_itr->path()<<endl;
            ++nrof_latents;
        }
    }
    if(nrof_latents <= 0)
    {
        cout<<"No latent templates found in directory: "<<latent_path<<endl;
        return -1;
    }

    register int i,j,k;

//This section loads all the latents into FPTemplate objects, which may or may not be desired
//    //load latent templates
//    vector<LatentFPTemplate> latent_templates(nrof_latents);
//    for(i=0; i<latent_template_files.size();++i)
//    {
//        //load latent template and create a latent FP object
//        cout<<"Loading latent #"<<i<<endl;
//        load_FP_template(latent_template_files[i], latent_templates[i]);
//    }

    // //load rolled original template filenames
    // is.open(rolled_list_file, ifstream::binary);
    // // get length of file:
    // is.seekg(0, ios::end);
    // length = is.tellg();
    // if( length<=0 )
    // {
    //     cout<<"rolled template list is empty!"<<endl;
    //     return -1;
    // }
    // is.seekg(0, ios::beg);
    // int nrof_rolled = 0;
    // vector<string> rolled_template_files;
    // while (!is.eof())
    // {
    //     getline(is, template_file);
    //     if(template_file.length()<10)
    //         break;
    //     rolled_template_files.push_back(template_file);
    //     ++nrof_rolled;
    // }
    // is.close();

    int nrof_rolled = 0;
    vector<fs::path> rolled_template_files;
    for(fs::directory_iterator dir_itr(rolled_path); dir_itr != end_itr; ++dir_itr)
    {
    if(dir_itr->path().extension() == ".dat")
        { 
            rolled_template_files.push_back(dir_itr->path());
            cout<<"rolled template file"<<dir_itr->path()<<endl;
            ++nrof_rolled;
        }
    }
    if(nrof_rolled <= 0)
    {
        cout<<"No rolled templates found in directory: "<<rolled_path<<endl;
        return -1;
    }
    cout<<"Gallery size: "<<nrof_rolled<<endl;

//This section loads all the rolled prints into FPTemplate objects, which may or may not be desired
    //load rolled templates
//    vector<RolledFPTemplate> rolled_templates(nrof_rolled);
//    for(int i=0; i<rolled_template_files.size();++i)
//    {
//        //load rolled original template and create a FP object
//        load_FP_template(rolled_template_files[i], rolled_templates[i]);
//    }
//
    {
        using namespace std::chrono;
        vector<high_resolution_clock::time_point> t(10);
        duration<double, std::milli> time_span;
        t[0] = high_resolution_clock::now();

        for(i=0;i<nrof_latents; ++i)
        {
            vector<float> scores(nrof_rolled, -1);
            //vector<vector<float>> scores(nrof_rolled);
            cout<<latent_template_files[i]<<endl;

            LatentFPTemplate latent_FP;
            //load latent original template and create a latent FP object
            load_FP_template(latent_template_files[i].string(), latent_FP);
            cout<<"Latent minutiae templates: "<<latent_FP.m_nrof_minu_templates<<endl;
            cout<<"Latent texture templates: "<<latent_FP.m_nrof_texture_templates<<endl;
            if(latent_FP.m_nrof_minu_templates<=0 && latent_FP.m_nrof_texture_templates<=0)
            {
				cout<<"No minutiae or texture templates found"<<endl;
                ofstream output;
                output.open(score_path + latent_template_files[i].stem().string() + ".csv");

                output<<0<<endl;
                output.close();

                continue;
            }

            using namespace std::chrono;
            high_resolution_clock::time_point t_start = high_resolution_clock::now();
            int result = 0;
	        #pragma omp parallel for num_threads(8) schedule(static,16)
            for(j=0;j<nrof_rolled; ++j)
            {
				// cout<<"latent: "<<i<<" rolled:"<<j<<"   "<<rolled_template_files[j]<<endl;

                RolledFPTemplate rolled_FP;
                if(load_FP_template(rolled_template_files[j].string(), rolled_FP)<0)
                {
                    rolled_FP.m_nrof_minu_templates=0;
                    rolled_FP.m_nrof_texture_templates = 0;
                };

                //vector<float> score;
                //One2One_matching_all_templates(latent_FP,rolled_FP,score);
                //scores[j] = score;

				vector<float> score;
				result = One2One_matching_selected_templates(latent_FP,rolled_FP,score);
                if(result == 1){
                    continue;
                }
                else if(result == 2){
                    cout<<"Comparison failed: rolled template is empty. Skipping."<<endl;
                    continue;
                }
				float final_score = score[0] + score[1] + score[2] + score[28]*0.3;
                scores[j] = final_score;
                //vector<float> selected_scores{score[0], score[1], score[2], score[28], final_score};
				//scores[j] = selected_scores;
            }
            if(result == 1){
                cout<<"Matching failed: latent template is empty. Skipping."<<endl;
                continue;
            }
            auto t_end = high_resolution_clock::now();
            duration<double, std::milli> duration = (t_end - t_start);
            //cout<<duration.count()<<endl;
			/**
            ofstream output;
            output.open("/home/cori/Lab/latentafisdemo/Matching_20181204/scores/01202019_2.csv", std::ios_base::app);

            for(j=0;j<nrof_rolled; ++j)
            {
                output<<scores[j]<<",";
            }
			output<<scores[scores.size()-1]<<endl;
            output.close();
			**/

			ofstream output;
			//output.setf(ios::fixed,ios::floatfield);
			//output.precision(3);
			output.open(score_path + latent_template_files[i].stem().string() + ".csv");
			//cout.setf(ios::fixed,ios::floatfield);
			//cout.precision(3);

			//output<<"filename,score"<<endl;
			for(j=0;j<nrof_rolled; ++j)
			{
                output<<rolled_template_files[j]<<","<<std::setprecision(3)<<std::fixed<<scores[j]<<endl;
				//output<<rolled_template_files[j]<<","<<std::setprecision(3)<<std::fixed<<scores[j][0]<<","<<scores[j][1]<<","<<scores[j][2]<<","<<scores[j][3]<<","<<scores[j][4]<<endl;
			}
			output.close();
        }
        t[1] = high_resolution_clock::now();
        time_span = t[1] - t[0];
        cout<<"Total matching duration (ms): "<<time_span.count()<<endl;

		auto timenow = chrono::system_clock::to_time_t(chrono::system_clock::now());
		//cout << ctime(&timenow) << endl;
    }
    return 0;
}

int Matcher::One2List_matching(string latent_template_file_string, string rolled_list_file, string score_path)
{
    //string template_file;
    //ifstream is;

    register int i,j,k;

    fs::path latent_template_file(latent_template_file_string);
    string score_file = score_path + latent_template_file.stem().string() + ".csv";

    //load rolled template filenames
    int nrof_rolled = 0;
    vector<fs::path> rolled_template_files;
    fs::directory_iterator end_itr;
    for(fs::directory_iterator dir_itr(rolled_list_file); dir_itr != end_itr; ++dir_itr)
    {
    if(dir_itr->path().extension() == ".dat")
        { 
            rolled_template_files.push_back(dir_itr->path());
            //cout<<"rolled template file"<<dir_itr->path()<<endl;
            ++nrof_rolled;
        }
    }
    if(nrof_rolled <= 0)
    {
        cout<<"No rolled templates found in directory: "<<rolled_list_file<<endl;
        return -1;
    }

    // Create a vector of indices
    // Allows us to output a sorted score list for use with the GUI
    vector<int> ind(rolled_template_files.size(), 0);
    for(int n; n != rolled_template_files.size(); n++){
        ind[n] = n;
    }

    {
        using namespace std::chrono;
        vector<high_resolution_clock::time_point> t(10);
        duration<double, std::milli> time_span;
        t[0] = high_resolution_clock::now();

        vector<float> scores(nrof_rolled, -1);
        cout<<"Latent Query: "<<latent_template_file<<endl;
        cout<<"Gallery size: "<<nrof_rolled<<endl;
        LatentFPTemplate latent_FP;
        //load latent original template and create a latent FP object
        load_FP_template(latent_template_file.string(), latent_FP);
        if(latent_FP.m_nrof_minu_templates<=0 && latent_FP.m_nrof_texture_templates<=0)
        {
            ofstream output;
            output.open(score_file);

            output<<0<<endl;
            output.close();

        }

        using namespace std::chrono;
        high_resolution_clock::time_point t_start = high_resolution_clock::now();
        int result = 0;
        #pragma omp parallel for num_threads(8) schedule(static,16)
        for(j=0;j<nrof_rolled; ++j)
        {
            //cout<<"latent: "<<i<<" rolled:"<<j<<"   "<<rolled_template_files[j]<<endl;

            RolledFPTemplate rolled_FP;
            if(load_FP_template(rolled_template_files[j].string(), rolled_FP)<0)
            {
                rolled_FP.m_nrof_minu_templates=0;
                rolled_FP.m_nrof_texture_templates = 0;
            };

            vector<float> score;
            result = One2One_matching_selected_templates(latent_FP,rolled_FP,score);
            if(result == 1){
                continue;
            }
            else if(result == 2){
                cout<<"Comparison failed: rolled template is empty. Skipping."<<endl;
                continue;
            }
            // One2One_matching_all_templates(latent_FP,rolled_FP,score);
            // One2One_matching_all_templates(latent_FP,rolled_templates[j],score);
            float final_score = score[0] + score[1] + score[2] + score[28]*0.3;
            scores[j] = final_score;
        }
        if(result == 1){
            cout<<"Matching failed: latent template is empty. Exiting."<<endl;
            return 1;
        }
        auto t_end = high_resolution_clock::now();
        duration<double, std::milli> duration = (t_end - t_start);
        //cout<<duration.count()<<endl;
        ofstream output;
        output.open(score_file);

        // Sort scores to create rank list
        sort(ind.begin(), ind.end(), [&](const int& a, const int& b){
                return (scores[a] > scores[b]);
            }
        );
        output<<"filename,score"<<endl;
        //generate correspondence files for top 24 only
        cout<<"Match Results"<<endl;
        cout<<"----------------"<<endl;
        cout<<"Rank     Filename      Score"<<endl;
        for(j=0; j<24; ++j)
        {
            if(j >= nrof_rolled){
                break;
            }
            output<<to_string(j+1)<<rolled_template_files[ind[j]]<<","<<scores[ind[j]]<<endl;
            RolledFPTemplate rolled_FP;
            load_FP_template(rolled_template_files[ind[j]].string(), rolled_FP);
            string latent_fname = latent_template_file.stem().string();
            string rolled_fname = rolled_template_files[ind[j]].stem().string();
            string corr_file = "/research/prip-tymoszek/LatentAFIS/scores/corr" + latent_fname + "_" + rolled_fname;
            vector<float> score;
            One2One_matching_selected_templates(latent_FP,rolled_FP,score, true, corr_file);
            cout<<to_string(j+1)<<"        "<<rolled_template_files[ind[j]].filename()<<"       "<<scores[ind[j]]<<endl;
        }
        output.close();
        t[1] = high_resolution_clock::now();
        time_span = t[1] - t[0];
        cout<<"Total matching duration (ms): "<<time_span.count()<<endl;
        
    }
    return 0;
}

int Matcher::One2One_matching_all_templates(LatentFPTemplate &latent_template, RolledFPTemplate &rolled_template, vector<float> & score)
{

    score.resize(latent_template.m_nrof_minu_templates + latent_template.m_nrof_texture_templates);
    std::fill(score.begin(), score.end(), 0);
//    vector<int> selected_ind{18-1, 10-1, 8-1, 4-1};


//    if(latent_template.m_nrof_minu_templates<=selected_ind[0] && latent_template.m_nrof_texture_templates<=0)
   if(latent_template.m_nrof_minu_templates<=0 && latent_template.m_nrof_texture_templates<=0)
   {
//        cout<<"latent template is empty"<<endl;
        return 1;
    }

    if(rolled_template.m_nrof_minu_templates<=0 && rolled_template.m_nrof_texture_templates<=0)
    {
//        cout<<"rolled template is empty"<<endl;
        return 2;
    }
    int i,j;

    using namespace std::chrono;
    vector<high_resolution_clock::time_point> t(10);

    t[0] = high_resolution_clock::now();
//


//    for(i=0;i<selected_ind.size() && rolled_template.m_nrof_minu_templates>0; ++i)
    for(i=0;i<latent_template.m_nrof_minu_templates && rolled_template.m_nrof_minu_templates; ++i)
    {
//        int ind = selected_ind[i];
//        if(latent_template.m_nrof_minu_templates<=ind)
//            continue;
        float s = One2One_minutiae_matching(latent_template.m_minu_templates[i], rolled_template.m_minu_templates[0]);
        score[i] = s;
    }
    t[1] = high_resolution_clock::now();

//    for(i=0;i<min(1,latent_template.m_nrof_texture_templates) && rolled_template.m_nrof_texture_templates>0 ; ++i)
    for(i=0;i<latent_template.m_nrof_texture_templates && rolled_template.m_nrof_texture_templates>0 ; ++i)
    {
        float s = One2One_texture_matching(latent_template.m_texture_templates[i], rolled_template.m_texture_templates[0]);
        score[i+latent_template.m_nrof_minu_templates] = s;
    }
//    t[2] = high_resolution_clock::now();
//    duration<double, std::milli> time_span = t[1] - t[0];
//
//    cout<<"minutiae template matching= " << time_span.count() <<endl;
//
//    time_span = t[2] - t[1];
//    cout<<"texture template matching= " << time_span.count() <<endl;
}

int Matcher::One2One_matching_selected_templates(LatentFPTemplate &latent_template, RolledFPTemplate &rolled_template, vector<float> & score, bool save_corr, string corr_file)
{
    score.resize(latent_template.m_nrof_minu_templates + latent_template.m_nrof_texture_templates);
    std::fill(score.begin(), score.end(), 0);
    vector<int> selected_ind{27-1, 3-1, 12-1};


    if(latent_template.m_nrof_minu_templates<=selected_ind[0] && latent_template.m_nrof_texture_templates<=0)
    {
        //cout<<"latent template is empty"<<endl;
        return 1;
    }

    if(rolled_template.m_nrof_minu_templates<=0 && rolled_template.m_nrof_texture_templates<=0)
    {
//        cout<<"rolled template is empty"<<endl;
        return 2;
    }
    int i,j;

    using namespace std::chrono;
    vector<high_resolution_clock::time_point> t(10);

    t[0] = high_resolution_clock::now();
//


    for(i=0;i<selected_ind.size() && rolled_template.m_nrof_minu_templates>0; ++i)
    {
        int ind = selected_ind[i];
        if(latent_template.m_nrof_minu_templates<=ind)
            continue;
        string one_corr_file = corr_file + "_" + to_string(i) + ".csv";
        float s = One2One_minutiae_matching(latent_template.m_minu_templates[ind], rolled_template.m_minu_templates[0], save_corr, one_corr_file);
        score[i] = s;
    }
    t[1] = high_resolution_clock::now();

    for(i=0;i<min(1,latent_template.m_nrof_texture_templates) && rolled_template.m_nrof_texture_templates>0 ; ++i)
    {
        float s = One2One_texture_matching(latent_template.m_texture_templates[i], rolled_template.m_texture_templates[0]);
        score[i+latent_template.m_nrof_minu_templates] = s;
    }
//    t[2] = high_resolution_clock::now();
//    duration<double, std::milli> time_span = t[1] - t[0];
//
//    cout<<"minutiae template matching= " << time_span.count() <<endl;
//
//    time_span = t[2] - t[1];
//    cout<<"texture template matching= " << time_span.count() <<endl;
}


float Matcher::One2One_minutiae_matching(MinutiaeTemplate &latent_minu_template, MinutiaeTemplate &rolled_minu_template, bool save_corr, string corr_file)
{
    ++nrof_matching;
    // step 1: compute pairwise similarity between descriptors

    int n_time = 0;
    register int i,j,k;

    int des_len = rolled_minu_template.m_des_length;
    if(des_len!=latent_minu_template.m_des_length){
        cout<<latent_minu_template.m_des_length<<endl;
	cout<<rolled_minu_template.m_des_length<<endl;
	}
    assert(des_len == latent_minu_template.m_des_length);



    float simi = 0.0;
//    float *p_latent_des, *p_latent_des0, *p_rolled_des;

//    vector<float> latent_simi_sum(latent_minu_template.m_nrof_minu);
//    vector<float> rolled_simi_sum(rolled_minu_template.m_nrof_minu);


    using namespace std::chrono;
    vector<high_resolution_clock::time_point> t(10);

//    t[n_time++] = high_resolution_clock::now();

   // MatrixXf a = Map<MatrixXf>(latent_minu_template.m_des,latent_minu_template.m_nrof_minu,des_len);
    Matrix<float, Eigen::Dynamic, Eigen::Dynamic> aa =  Map<Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(latent_minu_template.m_des,latent_minu_template.m_nrof_minu,des_len);
    Matrix<float, Eigen::Dynamic, Eigen::Dynamic> bb =  Map<Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(rolled_minu_template.m_des,rolled_minu_template.m_nrof_minu,des_len);


    //MatrixXf b = Map<MatrixXf>(rolled_minu_template.m_des,rolled_minu_template.m_nrof_minu,des_len);

    MatrixXf  simi_matrix=aa*bb.transpose();

    for(i=0; i<latent_minu_template.m_nrof_minu; ++i)
    {
        for(j = 0; j<rolled_minu_template.m_nrof_minu; ++j)
        {
            // simi_matrix(i,j) = (simi_matrix(i,j)+0.5);
            if(simi_matrix(i,j)<0)
                simi_matrix(i,j) = 0;
//            output<<simi_matrix(i,j)<<" ";
        }
//        output<<endl;
    }

    //  step 2:  similarity normalization
    VectorXf  rolled_simi_sum = simi_matrix.colwise().sum();
    VectorXf  latent_simi_sum = simi_matrix.rowwise().sum();

//    float *norm_simi_matrix = new float[rolled_minu_template.m_nrof_minu*latent_minu_template.m_nrof_minu]();

//
    int ind_1, ind_2 ;
    vector<float> norm_simi_matrix(latent_minu_template.m_nrof_minu*rolled_minu_template.m_nrof_minu);
    float norm_simi=0.0;
    for(i=0; i<latent_minu_template.m_nrof_minu; ++i)
    {
        ind_1 = i*rolled_minu_template.m_nrof_minu;
        for(j = 0; j<rolled_minu_template.m_nrof_minu; ++j)
        {
            ind_2  = ind_1 + j;
            norm_simi = simi_matrix(i,j)/(latent_simi_sum(i) + rolled_simi_sum(j) - simi_matrix(i,j)+0.000001); // //simi_matrix[ind_2]*
            norm_simi_matrix[ind_2] = norm_simi;
        }
    }
////
//
//    // cout<<" similarity normalization: "<< ms2-ms1;
    // step 3: find top N correspondences using norm_simi_matrix;
    // the sorting part can be replaced by a min-heap
    std::vector<int> y(norm_simi_matrix.size());
    std::iota(y.begin(), y.end(), 0);
    auto comparator = [&norm_simi_matrix](int a, int b){ return norm_simi_matrix[a] > norm_simi_matrix[b]; };
    // auto comparator = [&simi_matrix](int a, int b){ return simi_matrix[a] > simi_matrix[b]; };
    std::sort(y.begin(), y.end(), comparator);
////
//    for (i=0; i<10; ++i)
//       std::cout << y[i] << ' '<<norm_simi_matrix[y[i]]<<endl;

    std::vector<tuple<float, int, int>>corr;
    int topN = 120;
    if(rolled_minu_template.m_nrof_minu*latent_minu_template.m_nrof_minu<topN)
        topN = rolled_minu_template.m_nrof_minu*latent_minu_template.m_nrof_minu;
    for(i=0; i<topN ; ++i)
    {
        ind_1 = y[i]/rolled_minu_template.m_nrof_minu; // latent minutiae  index
        ind_2 = y[i] - ind_1*rolled_minu_template.m_nrof_minu; // rolled minutiae index
        // simi = simi_matrix[ind_1*rolled_minu_template.m_nrof_minu+ind_2];
        simi = simi_matrix(ind_1,ind_2);
        corr.push_back(make_tuple(simi,ind_1,ind_2));
    }
//



//
//
//    std::vector<tuple<float, int, int>>tmp_corr(latent_minu_template.m_nrof_minu), corr(N);
//    MatrixXf::Index max_index;
//    float max_val;
//    for(i=0;i<latent_minu_template.m_nrof_minu; ++i)
//    {
//        max_val = simi_matrix.row(i).maxCoeff(&max_index);
//        // tmp_corr.push_back(make_tuple(max_val,i,max_index));
//        tmp_corr[i] = make_tuple(max_val,i,max_index);
//        // cout<<"latent ID: "<<i<<" rolled ID: "<<max_index<<endl;
//    }
//    if(tmp_corr.size()>N)
//    {
//        std::vector<int> y(tmp_corr.size());
//        std::iota(y.begin(), y.end(), 0);
//        auto comparator = [&tmp_corr](int a, int b){ return get<0>(tmp_corr[a]) > get<0>(tmp_corr[b]); };
//        std::sort(y.begin(), y.end(), comparator);
//
//        for(i=0;i<N; ++i)
//        {
//            //corr.push_back(tmp_corr[y[i]]);
//            corr[i] = tmp_corr[y[i]];
//        }
//    }
//    else
//        corr = tmp_corr;
//
//
//    output.open("corr.txt");
//    for(i=0;i<N; ++i)
//    {
//        output<<get<0>(corr[i])<<" "<<get<1>(corr[i])<<" "<<get<2>(corr[i])<<endl;
//    }
//    output.close();



     // step 4: remove false correspondences using two graph matching


    int d_thr = 30;
    vector<tuple<float, int, int>> corr2 = LSS_R_Fast2_Dist_eigen(corr, latent_minu_template, rolled_minu_template, d_thr);


    vector<tuple<float, int, int>> corr3  = LSS_R_Fast2(corr2, latent_minu_template, rolled_minu_template, d_thr);

    if (save_corr){
        ofstream output;
        output.open(corr_file);
        for(int i = 0; i < corr3.size(); i++){
            output<<latent_minu_template.m_minutiae[get<1>(corr3[i])].x<<","<<latent_minu_template.m_minutiae[get<1>(corr3[i])].y
                <<","<<rolled_minu_template.m_minutiae[get<2>(corr3[i])].x<<","<<rolled_minu_template.m_minutiae[get<2>(corr3[i])].y<<endl;
        }
        output.close();
    }


    float score = 0.0;

    for(i=0; i<corr3.size(); ++i)
    {
        score += get<0>(corr3[i]);
    }
    //cout<<score<<endl;
    return score;

}

int Matcher::One2One_matching(string latent_file, string rolled_file)
{
    LatentFPTemplate latent_template;
    RolledFPTemplate rolled_template;
    load_FP_template(latent_file,latent_template);
    load_FP_template(rolled_file,rolled_template);

    vector<float> score;
    int ret = One2One_matching_selected_templates(latent_template, rolled_template, score);

    return ret;
}

float Matcher::One2One_texture_matching(LatentTextureTemplate &latent_texture_template, RolledTextureTemplatePQ &rolled_texture_template)
{
    ++nrof_matching;

    // step 1: compute pairwise similarity between descriptors
    int n_time = 0;
    register int i,j,k;

    int des_len = rolled_texture_template.m_des_length;
    //assert(des_len == latent_texture_template.m_des_length);


    // float * simi_matrix = new float [latent_texture_template.m_nrof_minu*rolled_minu_template.m_nrof_minu];
   float simi_matrix[MaxNRolledMinu*MaxNLatentMinu];
   memset(simi_matrix,0,MaxNRolledMinu*MaxNLatentMinu*sizeof(float));


    if(latent_texture_template.m_nrof_minu> MaxNLatentMinu)
        latent_texture_template.m_nrof_minu = MaxNLatentMinu;
    if(rolled_texture_template.m_nrof_minu> MaxNRolledMinu)
        rolled_texture_template.m_nrof_minu = MaxNRolledMinu;
//    vector<float> simi_matrix(latent_texture_template.m_nrof_minu*rolled_minu_template.m_nrof_minu);
//    vector<float>::iterator iter= simi_matrix.begin();

    float simi = 0.0;
    float *p_latent_des, *p_latent_des0, *p_rolled_des;

//    vector<float> latent_simi_sum(latent_texture_template.m_nrof_minu);
//    vector<float> rolled_simi_sum(latent_texture_template.m_nrof_minu);

    using namespace std::chrono;
    vector<high_resolution_clock::time_point> t(10);

    t[n_time++] = high_resolution_clock::now();
    register float dist0=0.0, dist1= 0.0, dist2= 0, dist3=0.0, dist4 = 0.0; //, dist5, dist6,dist7, dist8;
    register int code1 = 0, code2 = 0, code3 = 0, code4=0;
    float *p_dist_codewords0 = NULL, *p_dist_codewords1 = NULL, *p_dist_codewords2 =NULL;
    unsigned char *p_des0=NULL, *p_des1=NULL;

    int n=0;
    int nrof_clusters3 = nrof_clusters*3, nrof_clusters2 = nrof_clusters*2;
    int method = 1;
    if(method == 1)
    {
        for(i=0; i<latent_texture_template.m_nrof_minu; ++i)
        {
            p_dist_codewords0 = latent_texture_template.m_dist_codewords + i*nrof_subs*nrof_clusters;
            for(j=0; j<rolled_texture_template.m_nrof_minu; ++j)
            {
                dist1 = 6.;
                dist2 = 0.;
                dist3 = 0.;
                dist4 = 0.;
                p_dist_codewords1 = p_dist_codewords0;
                p_des0 = rolled_texture_template.m_desPQ + j* rolled_texture_template.m_des_length;
                for(k=0; k<nrof_subs; k+=4, p_dist_codewords1+=4*nrof_clusters)
                {
    //                code = 0.0;
                    code1 = *(p_des0+k);
                    dist1 -= *(p_dist_codewords1 + code1);

                    code2 = *(p_des0+k+1);
                    dist2 -= *(p_dist_codewords1 + code2 + nrof_clusters);

                    code3 = *(p_des0+k+2);
                    dist3 -= *(p_dist_codewords1 + code3 + nrof_clusters2);

                    code4 = *(p_des0+k+3);
                    dist4 -= *(p_dist_codewords1 + code4 + nrof_clusters3);

                }
                //dist = sqrt(dist);
    //            *(iter) =  dist;
                simi_matrix[n++] = (dist1+dist2)+ (dist3+dist4);
            }
        }
    }
    else if (method==2)
    {
        int B1=64, B2 = 64;
        for(i=0; i<latent_texture_template.m_nrof_minu-B1; i+=B1)
        {

            for(j=0; j<rolled_texture_template.m_nrof_minu-B2; j += B2)
            {

                for(int ii=i; ii<i+B1; ++ii)
                {
                    p_dist_codewords0 = latent_texture_template.m_dist_codewords + ii*nrof_subs*nrof_clusters;
                    for(int jj=j; jj<j+B2; ++jj)
                    {
                       dist1 = 6.;
                       dist2 = 0.;
                       dist3 = 0.;
                       dist4 = 0.;
                       p_dist_codewords1 = p_dist_codewords0;
                       p_des0 = rolled_texture_template.m_desPQ + jj* rolled_texture_template.m_des_length;


                        for(k=0; k<nrof_subs; k+=4, p_dist_codewords1+=4*nrof_clusters)
                        {
            //                code = 0.0;
                            code1 = *(p_des0+k);
                            dist1 -= *(p_dist_codewords1 + code1);

                            code2 = *(p_des0+k+1);
                            dist2 -= *(p_dist_codewords1 + code2 + nrof_clusters);

                            code3 = *(p_des0+k+2);
                            dist3 -= *(p_dist_codewords1 + code3 + nrof_clusters2);

                            code4 = *(p_des0+k+3);
                            dist4 -= *(p_dist_codewords1 + code4 + nrof_clusters3);

//                            code1 = *(p_des0+k+4);
//                            dist5 -= *(p_dist_codewords1 + code1 + nrof_clusters3+nrof_clusters);
//
//                            code2 = *(p_des0+k+5);
//                            dist6 -= *(p_dist_codewords1 + code2 + nrof_clusters3+nrof_clusters2);
//
//                            code3 = *(p_des0+k+6);
//                            dist7 -= *(p_dist_codewords1 + code3 + nrof_clusters3+nrof_clusters3);
//
//                            code4 = *(p_des0+k+7);
//                            dist8 -= *(p_dist_codewords1 + code4 + nrof_clusters3+nrof_clusters3);

                        }
                        //dist = sqrt(dist);
            //            *(iter) =  dist;
                        simi_matrix[ii*rolled_texture_template.m_nrof_minu+jj] = (dist1+dist2)+ (dist3+dist4);
                    }
                }
            }
        }
    }
    else if(method == 3)
    {
         int B1=64, B2 = 64;
        for(i=0; i<latent_texture_template.m_nrof_minu-B1; i+=B1)
        {
            p_dist_codewords0 = latent_texture_template.m_dist_codewords + i*nrof_subs*nrof_clusters;
            for(j=0; j<rolled_texture_template.m_nrof_minu-B2; j += B2)
            {
                p_dist_codewords1 = p_dist_codewords0;
                p_des0 = rolled_texture_template.m_desPQ + j* rolled_texture_template.m_des_length;
                for(int ii=i; ii<i+B1; ++ii)
                {
                    p_des1 = p_des0;
                    for(int jj=j; jj<j+B2; ++jj)
                    {
                       p_dist_codewords2 = p_dist_codewords1;
                       dist1 = 6.;
                       dist2 = 0.;
                       dist3 = 0.;
                       dist4 = 0.;

                        for(k=0; k<nrof_subs; k+=4, p_dist_codewords2+=4*nrof_clusters)
                        {
            //                code = 0.0;
                            code1 = *(p_des1+k);
                            dist1 -= *(p_dist_codewords2 + code1);

                            code2 = *(p_des1+k+1);
                            dist2 -= *(p_dist_codewords2 + code2 + nrof_clusters);

                            code3 = *(p_des1+k+2);
                            dist3 -= *(p_dist_codewords2 + code3 + nrof_clusters2);

                            code4 = *(p_des1+k+3);
                            dist4 -= *(p_dist_codewords2 + code4 + nrof_clusters3);

                        }
                        //dist = sqrt(dist);
            //            *(iter) =  dist;
                        simi_matrix[ii*rolled_texture_template.m_nrof_minu+jj] = (dist1+dist2)+ (dist3+dist4);
                        p_des1 +=  rolled_texture_template.m_des_length;
                    }
                    p_dist_codewords1 += nrof_subs*nrof_clusters;
                }
            }
        }
    }
    else if(method==4)
    {
        unsigned char *p_des3=NULL, *p_des2=NULL;
        for(i=0; i<latent_texture_template.m_nrof_minu; ++i)
        {
            p_dist_codewords0 = latent_texture_template.m_dist_codewords + i*nrof_subs*nrof_clusters;
            for(k=0; k<nrof_subs; ++k)
            {
                for(j=0; j<rolled_texture_template.m_nrof_minu-4; j+=4)
                {
                    n = i*rolled_texture_template.m_nrof_minu + j;
                    p_des0 = rolled_texture_template.m_desPQ + j* rolled_texture_template.m_des_length + k;
                    p_des1 = p_des0 + rolled_texture_template.m_des_length;
                    p_des2 = p_des1 + rolled_texture_template.m_des_length;
                    p_des3 = p_des2 + rolled_texture_template.m_des_length;

                    dist0 -= *(p_dist_codewords0 + *p_des0);

                    dist1 -= *(p_dist_codewords0 + *p_des1);

                    dist2 -= *(p_dist_codewords0 + *p_des2);

                    dist3 -= *(p_dist_codewords0 + *p_des3);

                    simi_matrix[n] += dist0;
                    simi_matrix[n+1] += dist1;
                    simi_matrix[n+2] += dist2;
                    simi_matrix[n+3] += dist3;
                }
            }
        }
    }
//
//    for(j=0; j<rolled_minu_template.m_nrof_minu; ++j)
//    {
//        dist1 = 0.;
//        dist2 = 0.;
//        dist3 = 0.;
//        dist4 = 0.;
//        p_des0 = rolled_minu_template.m_des + j* rolled_minu_template.m_des_length;
//        for(k=0; k<nrof_subs; k+=4)
//        {
////                code = 0.0;
//            code1 = *(p_des0+k);
//            code2 = *(p_des0+k+1);
//            code3 = *(p_des0+k+2);
//            code4 = *(p_des0+k+3);
//
//            for(i=0; i<latent_texture_template.m_nrof_minu; ++i)
//            {
//                p_dist_codewords0 = latent_texture_template.m_dist_codewords + i*nrof_subs*nrof_clusters + k*nrof_clusters;
//
//                dist1 = *(p_dist_codewords0 + code1);
//                dist2 = *(p_dist_codewords0 + code2 + nrof_clusters);
//                dist3 = *(p_dist_codewords0 + code3 + nrof_clusters2);
//                dist4 = *(p_dist_codewords0 + code4 + nrof_clusters3);
//                simi_matrix[i*rolled_minu_template.m_nrof_minu+j] -= (dist1+dist2)+ (dist3+dist4);
//
//            }
//
//
//        }
//    }



//    ofstream output;
//    output.open("minu_simi_new.txt");
//    for(i=0; i<latent_texture_template.m_nrof_minu; ++i)
//    {
//        for(j = 0; j<rolled_minu_template.m_nrof_minu; ++j)
//        {
//            // simi_matrix(i,j) = (simi_matrix(i,j)+0.5);
//            //if(simi_matrix(i,j)<0)
//            //    simi_matrix(i,j) = 0;
//            output<<simi_matrix[i*rolled_minu_template.m_nrof_minu+j]<<" ";
//        }
//        output<<endl;
//
//    }
//    output.close();
    //delete [] simi_matrix;
    //simi_matrix = NULL;
    t[n_time] = high_resolution_clock::now();
    duration<double, std::milli> time_span = t[n_time] - t[n_time-1];

    time[n_time-1]+=time_span.count() ;  // minutiae similarity
    similarity_time += time_span.count() ;
//    cout<<" minutiae similarity: " << time_span.count() <<endl;
    n_time++;

//
    std::vector<tuple<float, int, int>>tmp_corr(latent_texture_template.m_nrof_minu), corr(N);
    float max_val;
    float *psimi = simi_matrix;
    int max_index;
    for(i=0;i<latent_texture_template.m_nrof_minu; ++i)
    {

        max_index = std::distance(psimi, std::max_element(psimi, psimi+rolled_texture_template.m_nrof_minu));
        max_val = *(psimi + max_index);
        // tmp_corr.push_back(make_tuple(max_val,i,max_index));
        tmp_corr[i] = make_tuple(max_val,i,max_index);

        psimi += rolled_texture_template.m_nrof_minu;
        // cout<<"latent ID: "<<i<<" rolled ID: "<<max_index<<endl;
    }
    if(tmp_corr.size()>N)
    {
        std::vector<int> y(tmp_corr.size());
        std::iota(y.begin(), y.end(), 0);
        auto comparator = [&tmp_corr](int a, int b){ return get<0>(tmp_corr[a]) > get<0>(tmp_corr[b]); };
        std::sort(y.begin(), y.end(), comparator);

        for(i=0;i<N; ++i)
        {
            //corr.push_back(tmp_corr[y[i]]);
            corr[i] = tmp_corr[y[i]];
        }
    }
    else
        corr = tmp_corr;
//    output.open("corr_new.txt");
//    for(i=0;i<N; ++i)
//    {
//        output<<get<0>(corr[i])<<" "<<get<1>(corr[i])<<" "<<get<2>(corr[i])<<endl;
//    }
//    output.close();
//
    t[n_time] = high_resolution_clock::now();

    time_span = t[n_time] - t[n_time-1];

    time[n_time-1]+=time_span.count() ;  // obtaining initial correspondences
    n_time++;


     // step 4: remove false correspondences using two graph matching


    int d_thr = 30;
    vector<tuple<float, int, int>> corr2 = LSS_R_Fast2_Dist_lookup(corr, latent_texture_template, rolled_texture_template, d_thr);


    t[n_time] = high_resolution_clock::now();
    time_span = t[n_time] - t[n_time-1];
    time[n_time-1]+=time_span.count() ;   // second order graph matching: distance
    n_time++;

    vector<tuple<float, int, int>> corr3  = LSS_R_Fast2(corr2, latent_texture_template, rolled_texture_template, d_thr);


    t[n_time] = high_resolution_clock::now();
    time_span = t[n_time] - t[n_time-1];
    time[n_time-1]+=time_span.count() ;   // second order graph matching: original
    n_time++;

    float score = 0.0;

    for(i=0; i<corr3.size(); ++i)
    {
        score += get<0>(corr3[i]);
    }
//    cout<<score<<endl;
    return score;

}

int Matcher::load_FP_template(string tname, LatentFPTemplate & fp_template)
{
    fp_template.release();
    const short Max_Nrof_Minutiae = 2*1000; // including virtual minutiae. We only consider top 1000 minutiae including both real and virtual minutiae for each template.
    const short Max_Des_Length = 192;
    const short Max_BlkSize = 100;

    ifstream is;
    is.open(tname, ifstream::binary);
    // get length of file:
    is.seekg(0, ios::end);
    int length = is.tellg();

    if( length<=0 )
    {
//        cout<<"template is empty!"<<endl;
        return 1;
    }
    is.seekg(0, ios::beg);
    short header[12];
    short h,w,blkH,blkW;
    unsigned char nrof_minu_template,nrof_texture_template;
    short nrof_minutiae;


    short nrof_minutiae_feature;
    short des_len;
    int i,j;

    short x[Max_Nrof_Minutiae],y[Max_Nrof_Minutiae];
    float ori[Max_Nrof_Minutiae];
    float oimg[Max_BlkSize*Max_BlkSize];

    float des[Max_Nrof_Minutiae*Max_Des_Length];

    for(int i=0; i<12; i++){
        is.read(reinterpret_cast<char*>(&header[i]),sizeof(short));
    }

    is.read(reinterpret_cast<char*>(&h),sizeof(short));
    is.read(reinterpret_cast<char*>(&w),sizeof(short));
    is.read(reinterpret_cast<char*>(&blkH),sizeof(short));
    is.read(reinterpret_cast<char*>(&blkW),sizeof(short));
    is.read(reinterpret_cast<char*>(&nrof_minu_template),sizeof(unsigned char));
    if(blkH>50)
        blkH = 50;
    if(blkW>50)
        blkW = 50;
    for(i=0;i<nrof_minu_template; ++i)
    {
//        cout<<"Creating minutiae template #"<<i<<endl;
        is.read(reinterpret_cast<char*>(&nrof_minutiae),sizeof(short));
//        cout<<"Have "<<nrof_minutiae<<" minutiae"<<endl;
        if(nrof_minutiae<=0)
            continue;
        if(nrof_minutiae>Max_Nrof_Minutiae)
        {
            cout<<"Number of minutiae is larger than Max Number of Minutiae (latent):"<< nrof_minutiae << ">"<<Max_Nrof_Minutiae<<endl;
            return 2;
        }
        if(blkH>Max_BlkSize || blkW>Max_BlkSize)
        {
            cout<<"The size of the ridge flow is larger than maximum size:"<< Max_BlkSize<<endl;
            return 4;
        }
        is.read(reinterpret_cast<char*>(x),sizeof(short)*nrof_minutiae);
        is.read(reinterpret_cast<char*>(y),sizeof(short)*nrof_minutiae);
        is.read(reinterpret_cast<char*>(ori),sizeof(float)*nrof_minutiae);
        is.read(reinterpret_cast<char*>(&des_len),sizeof(short));

        is.read(reinterpret_cast<char*>(des),sizeof(float)*nrof_minutiae*des_len);

        MinutiaeTemplate minu_template(nrof_minutiae,x,y,ori,des_len,des,blkH, blkW, oimg);
        fp_template.add_template(minu_template);
    }

    is.read(reinterpret_cast<char*>(&nrof_texture_template),sizeof(unsigned char));

   for(i=0;i<nrof_texture_template; ++i)
    {
        is.read(reinterpret_cast<char*>(&nrof_minutiae),sizeof(short));
        if(nrof_minutiae<=0)
            continue;
        if(nrof_minutiae>Max_Nrof_Minutiae)
        {
            cout<<"Number of minutiae is larger than Max Number of Minutiae:"<< nrof_minutiae << ">"<< Max_Nrof_Minutiae<<endl;
            return -1;
        }
        is.read(reinterpret_cast<char*>(x),sizeof(short)*nrof_minutiae);
        is.read(reinterpret_cast<char*>(y),sizeof(short)*nrof_minutiae);
        is.read(reinterpret_cast<char*>(ori),sizeof(float)*nrof_minutiae);
       // is.read(reinterpret_cast<char*>(reliability),sizeof(float)*nrof_minutiae);
        is.read(reinterpret_cast<char*>(&des_len),sizeof(short));
//        if(des_len!=192)
//        {
//            is.close();
//            return -1;
//        }
        is.read(reinterpret_cast<char*>(des),sizeof(float)*nrof_minutiae*des_len);


        LatentTextureTemplate texture_template(nrof_minutiae,x,y,ori,des_len,des);
        //cout<<"Computing codewords..."<<endl;
        texture_template.compute_dist_to_codewords(codewords, nrof_subs,  sub_dim,  nrof_clusters);
        //cout<<"Codewords done."<<endl;
        fp_template.add_texture_template(texture_template);
    }
    is.close();

//    cout<<tname<<endl;
    return 0;
};

int Matcher::load_FP_template(string tname, RolledFPTemplate & fp_template)
{
    fp_template.release();
    const short Max_Nrof_Minutiae = 2*1000; // including virtual minutiae. We only consider top 1000 minutiae including both real and virtual minutiae for each template.
    const short Max_Des_Length = 192;
    const short Max_BlkSize = 100;

    ifstream is;
    is.open(tname, ifstream::binary);
    // get length of file:
    is.seekg(0, ios::end);
    int length = is.tellg();

    if( length<=10 )
    {
//        cout<<"template is empty!"<<endl;
        return 1;
    }
    is.seekg(0, ios::beg);
    short header[12];
    short h,w,blkH,blkW;
    unsigned char nrof_minu_template,nrof_texture_template;
    short nrof_minutiae;


    short nrof_minutiae_feature;
    short des_len=96;
    int i,j;

    short x[Max_Nrof_Minutiae],y[Max_Nrof_Minutiae];
    float ori[Max_Nrof_Minutiae];
    float reliability[Max_Nrof_Minutiae];
    float oimg[Max_BlkSize*Max_BlkSize];

    float des[Max_Nrof_Minutiae*Max_Des_Length];

    for(int i=0; i<12; i++){
        is.read(reinterpret_cast<char*>(&header[i]),sizeof(short));
    }
    is.read(reinterpret_cast<char*>(&h),sizeof(short));
    is.read(reinterpret_cast<char*>(&w),sizeof(short));
    is.read(reinterpret_cast<char*>(&blkH),sizeof(short));
    is.read(reinterpret_cast<char*>(&blkW),sizeof(short));
    is.read(reinterpret_cast<char*>(&nrof_minu_template),sizeof(unsigned char));
    if(blkH>50)
        blkH = 50;
    if(blkW>50)
        blkW = 50;
    for(i=0;i<nrof_minu_template; ++i)
    {
        is.read(reinterpret_cast<char*>(&nrof_minutiae),sizeof(short));
        if(nrof_minutiae<=0)
            continue;
        if(nrof_minutiae>Max_Nrof_Minutiae)
        {
            cout<<"Number of minutiae is larger than Max Number of Minutiae:"<< nrof_minutiae << ">"<< Max_Nrof_Minutiae<<endl;
            return 2;
        }
        if(blkH>Max_BlkSize || blkW>Max_BlkSize)
        {
            cout<<"The size of the ridge flow is larger than maximum size:"<< Max_BlkSize<<endl;
            return 4;
        }
        is.read(reinterpret_cast<char*>(x),sizeof(short)*nrof_minutiae);
        is.read(reinterpret_cast<char*>(y),sizeof(short)*nrof_minutiae);
        is.read(reinterpret_cast<char*>(ori),sizeof(float)*nrof_minutiae);
        is.read(reinterpret_cast<char*>(&des_len),sizeof(short));

        is.read(reinterpret_cast<char*>(des),sizeof(float)*nrof_minutiae*des_len);

        MinutiaeTemplate minu_template(nrof_minutiae,x,y,ori,des_len,des,blkH, blkW, oimg);
        fp_template.add_template(minu_template);
    }

    is.read(reinterpret_cast<char*>(&nrof_texture_template),sizeof(unsigned char));

   for(i=0;i<nrof_texture_template; ++i)
    {
        is.read(reinterpret_cast<char*>(&nrof_minutiae),sizeof(short));
        if(nrof_minutiae<=0)
            continue;
        if(nrof_minutiae>Max_Nrof_Minutiae)
        {
            cout<<"Number of minutiae is larger than Max Number of Minutiae:"<< nrof_minutiae << ">"<< Max_Nrof_Minutiae<<endl;
            return -1;
        }
        is.read(reinterpret_cast<char*>(x),sizeof(short)*nrof_minutiae);
        is.read(reinterpret_cast<char*>(y),sizeof(short)*nrof_minutiae);
        is.read(reinterpret_cast<char*>(ori),sizeof(float)*nrof_minutiae);
       // is.read(reinterpret_cast<char*>(reliability),sizeof(float)*nrof_minutiae);
        is.read(reinterpret_cast<char*>(&des_len),sizeof(short));
        // cout<<"des len = " << des_len<<endl;
       // des_len = 16;
//        if(des_len!=192)
//        {
//            is.close();
//            return -1;
//        }
        is.read(reinterpret_cast<char*>(des),sizeof(float)*nrof_minutiae*des_len);


        RolledTextureTemplatePQ texture_template(nrof_minutiae,x,y,ori,des_len,des);
        fp_template.add_texture_template(texture_template);
    }
    is.close();

//    cout<<tname<<endl;
    return 0;
};

int Matcher::load_single_template(string tname, TextureTemplate& texture_template)
{
    ifstream is;
    is.open(tname, ifstream::binary);
    // get length of file:
    is.seekg(0, ios::end);
    int length = is.tellg();

    if( length<=0 )
    {
        cout<<"template is empty!"<<endl;
        return -1;
    }
    is.seekg(0, ios::beg);
    short nrof_minutiae;
    short nrof_minutiae_feature;
    short des_len;

    is.read(reinterpret_cast<char*>(&nrof_minutiae),sizeof(short));
    is.read(reinterpret_cast<char*>(&nrof_minutiae_feature),sizeof(short));
    is.read(reinterpret_cast<char*>(&des_len),sizeof(short));

    if(nrof_minutiae_feature<3)
        return -1; // number of minutiae feature is not sufficient.
    texture_template.initialization(nrof_minutiae,des_len);

    short *loc = new short [nrof_minutiae];
    is.read(reinterpret_cast<char*>(loc),sizeof(short)*nrof_minutiae);
    texture_template.set_x(loc);

    is.read(reinterpret_cast<char*>(loc),sizeof(short)*nrof_minutiae);
    texture_template.set_y(loc);

    delete [] loc; loc = NULL;

    float *feature = new float [nrof_minutiae];

    is.read(reinterpret_cast<char*>(feature),sizeof(float)*nrof_minutiae);
    texture_template.set_ori(feature);

    for(int i=3; i<nrof_minutiae_feature; ++i)
    {
        // read addition features. But they are not useful here
        is.read(reinterpret_cast<char*>(feature),sizeof(float)*nrof_minutiae);
    }

    is.read(reinterpret_cast<char*>(texture_template.m_des),sizeof(float)*nrof_minutiae*des_len);


    is.close();

    cout<<tname<<endl;
    return 0;
};

int Matcher::load_single_PQ_template(string tname, RolledTextureTemplatePQ& minu_template)
{
    ifstream is;
    is.open(tname, ifstream::binary);
    // get length of file:
    is.seekg(0, ios::end);
    int length = is.tellg();

    if( length<=0 )
    {
        cout<<"template is empty!"<<endl;
        return -1;
    }
    is.seekg(0, ios::beg);
    short nrof_minutiae;
    short nrof_minutiae_feature;
    short des_len;

    is.read(reinterpret_cast<char*>(&nrof_minutiae),sizeof(short));
    is.read(reinterpret_cast<char*>(&nrof_minutiae_feature),sizeof(short));
    is.read(reinterpret_cast<char*>(&des_len),sizeof(short));

    if(nrof_minutiae_feature<3)
        return -1; // number of minutiae feature is not sufficient.
    minu_template.initialization(nrof_minutiae,des_len);

    short *loc = new short [nrof_minutiae];
    is.read(reinterpret_cast<char*>(loc),sizeof(short)*nrof_minutiae);
    minu_template.set_x(loc);

    is.read(reinterpret_cast<char*>(loc),sizeof(short)*nrof_minutiae);
    minu_template.set_y(loc);

    delete [] loc; loc = NULL;

//    float *feature = new float [nrof_minutiae];
////    float feature[100000];
//    is.read(reinterpret_cast<char*>(feature),sizeof(float)*nrof_minutiae);
//    minu_template.set_ori(feature);
//
//    for(int i=3; i<nrof_minutiae_feature; ++i)
//    {
//        // read addition features. But they are not useful here
//        is.read(reinterpret_cast<char*>(feature),sizeof(float)*nrof_minutiae);
//    }
//    delete [] feature;
//    feature = NULL;
//

     float feature[100000];
//     float *feature = feature2;
     is.read(reinterpret_cast<char*>(feature),sizeof(float)*nrof_minutiae);
    minu_template.set_ori(feature);

    for(int i=3; i<nrof_minutiae_feature; ++i)
    {
        // read addition features. But they are not useful here
        is.read(reinterpret_cast<char*>(feature),sizeof(float)*nrof_minutiae);
    }
//    delete [] feature;
//    feature = NULL;


    is.read(reinterpret_cast<char*>(minu_template.m_desPQ),sizeof(unsigned char)*nrof_minutiae*des_len);


    is.close();

    minu_template.init_des();
    cout<<tname<<endl;
    return 0;
//
//    ifstream is;
//    is.open(tname, ios::binary|ios::ate);
////    // get length of file:
////    is.seekg(0, ios::end);
//    ifstream::pos_type length = is.tellg();
////    cout<<(int)length<<endl;
//    char pChars[1000000];
//     is.seekg(0, ios::beg);
//    is.read(pChars, length);
//    is.close();
//    delete [] pChars;
//    pChars = NULL;
//    *read = length;
    return 0;

//    ifstream ifs(filename, ios::binary|ios::ate);
//    ifstream::pos_type pos = ifs.tellg();
//    int length = pos;
//    char *pChars = new char[length];
//    ifs.seekg(0, ios::beg);
//    ifs.read(pChars, length);
//    ifs.close();
//    *read = length;
//    return pChars;
//}
//
//int _tmain(int argc, _TCHAR* argv[])
//{
//    const char * filename = "polar00.map";
//    int read ;
//    char * pChars = ReadAllBytes(filename, &read);
//    delete[] pChars;
//    return 0;
//}
};

Matcher::Matcher(const Matcher& orig)
{

}

vector<tuple<float, int, int>>  Matcher::LSS_R_Fast2_Dist(vector<tuple<float, int, int>> &corr, SingleTemplate & latent_template, SingleTemplate & rolled_template, float d_thr)
{
    int num = corr.size();
    vector<float> H(num*num);

    vector<short> flag_latent(latent_template.m_nrof_minu),flag_rolled(rolled_template.m_nrof_minu);

    register int i,j,k;

    MinuPoint *p_latent_minutia_1, *p_latent_minutia_2, *p_rolled_minutia_1, *p_rolled_minutia_2;
    float dist_1, dist_2, dist;
    float dx_1, dy_1, dx_2, dy_2;

//    cout<<"minutiae"<<endl;
//    for(i=0; i<num; ++i)
//    {
//        p_latent_minutia_1 = & latent_template.m_minutiae[get<1>(corr[i])];
//        p_rolled_minutia_1 = & rolled_template.m_minutiae[get<2>(corr[i])];
//
//
//        cout<<"x=: " <<(p_latent_minutia_1->x*16+24)<<" y=:"<<(p_latent_minutia_1->y*16+24)<<"  x=: " <<(p_rolled_minutia_1->x*16+24)<<" y=:"<<(p_rolled_minutia_1->y*16+24)<<endl;
//    }
//
    for(i=0; i<num-1; ++i)
    {
        p_latent_minutia_1 = & latent_template.m_minutiae[get<1>(corr[i])];
        p_rolled_minutia_1 = & rolled_template.m_minutiae[get<2>(corr[i])];
        for(j=i+1; j<num;++j)
        {
            p_latent_minutia_2 = & latent_template.m_minutiae[get<1>(corr[j])];
            p_rolled_minutia_2 = & rolled_template.m_minutiae[get<2>(corr[j])];

            dx_1 = p_latent_minutia_1->x-p_latent_minutia_2->x;
            dx_2 = p_rolled_minutia_1->x-p_rolled_minutia_2->x;

//            dx_1 = dx_1*16;
//            dx_2 = dx_2*16;
//            if( abs(dx_1-dx_2)>d_thr)
 //               continue;
            dy_1 = p_latent_minutia_1->y-p_latent_minutia_2->y;
            dy_2 = p_rolled_minutia_1->y-p_rolled_minutia_2->y;

//            dy_1 = dy_1*16;
//            dy_2 = dy_2*16;
         //   if( abs(dy_1-dy_2)>d_thr)
          //      continue;



            dist_1 = (dx_1*dx_1)+(dy_1*dy_1);
            dist_1 = sqrt(dist_1);


            dist_2 = (dx_2*dx_2)+(dy_2*dy_2);
            dist_2 = sqrt(dist_2);

            dist = fabs(dist_1-dist_2);
         //  if(fabs(dist_1-dist_2)>d_thr)
          //      continue;

            H[i*num+j] = (30-dist)/(25.0);
            if(H[i*num+j]>1)
                H[i*num+j] = 1.0;
            else if(H[i*num+j]<0)
                H[i*num+j] = 0.0;

            H[j*num+i] = H[i*num+j];
        }
    }






    vector<float> S(num),S1(num);

    float s0 = 1.0/num;
    for(i=0; i<num; ++i)
        S[i] = get<0>(corr[i]);


    float sum = 0.0;
    for(i=0;i<5 ; ++i)
    {
        sum = 0.0;
        for(j=0;j<num; ++j)
        {
            S1[j] = 0;
            for(k=0; k<num;++k)
            {
                //if(H[j*num+k])
                S1[j] += H[j*num+k]*S[k];
            }
            sum += S1[j];
        }
        sum = 1.0/(sum+0.0001);
        for(j=0;j<num; ++j)
        {
            S[j] = S1[j]*sum;
        }

//        s0 = 0.0;
//        for(j=0;j<num; ++j)
//        {
//            s0 += S[j];
//        }
//        s0 = 1.0/s0;
//
//        for(j=0;j<num; ++j)
//        {
//           S[j] *= s0;
//        }
        // cout<<s0<<endl;

    }




    // sort the S



    // the sorting part can be replaced by a min-heap
    std::vector<int> y(S.size());
    std::iota(y.begin(), y.end(), 0);
    auto comparator = [&S](int a, int b){ return S[a] > S[b]; };
    std::sort(y.begin(), y.end(), comparator);


    vector<tuple<float, int, int>>  new_corr;
    vector<int>  selected_ind;
    short ind;
    for(i=0; i<num; ++i)
    {

        ind = y[i];
//        cout<<S[ind]<<"  ";
        if(S[ind]<0.0001)
            break;
        if(flag_latent[get<1>(corr[ind])] == 1 | flag_rolled[get<2>(corr[ind])] == 1)
            continue;

        if(i==0)
        {
            selected_ind.push_back(ind);
            new_corr.push_back(make_tuple(get<0>(corr[ind]),get<1>(corr[ind]),get<2>(corr[ind])));
//
//            cout<<"S="<<S[ind]<< " latent x = "<<latent_template.m_minutiae[get<1>(corr[ind])].x<<" latent y="<<latent_template.m_minutiae[get<1>(corr[ind])].y<<" latent ori="<<latent_template.m_minutiae[get<1>(corr[ind])].ori
//               << " rolled x="<< rolled_template.m_minutiae[get<2>(corr[ind])].x << " rolled y="<<rolled_template.m_minutiae[get<2>(corr[ind])].y<<
//                " rolled roi="<<rolled_template.m_minutiae[get<2>(corr[ind])].ori<<endl;

            flag_latent[get<1>(corr[ind])] = 1;
            flag_rolled[get<2>(corr[ind])] = 1;
        }
        else
        {
            int found =0;
            for(j=0;j<selected_ind.size(); ++j)
            {
                if(H[ind*num+selected_ind[j]]<0.00001)
                {
                    found = 1;
                    break;
                }
            }
            if(found==0)
            {
                selected_ind.push_back(ind);
                new_corr.push_back(make_tuple(get<0>(corr[ind]),get<1>(corr[ind]),get<2>(corr[ind])));
        //
//                cout<<"S="<<S[ind]<< " latent x = "<<latent_template.m_minutiae[get<1>(corr[ind])].x<<" latent y="<<latent_template.m_minutiae[get<1>(corr[ind])].y<<" latent ori="<<latent_template.m_minutiae[get<1>(corr[ind])].ori
//                       << " rolled x="<< rolled_template.m_minutiae[get<2>(corr[ind])].x << " rolled y="<<rolled_template.m_minutiae[get<2>(corr[ind])].y<<
//                        " rolled roi="<<rolled_template.m_minutiae[get<2>(corr[ind])].ori<<endl;


                flag_latent[get<1>(corr[ind])] = 1;
                flag_rolled[get<2>(corr[ind])] = 1;
            }
        }
    }

//
//    delete [] H; H = NULL;
//    delete [] flag_rolled; flag_rolled = NULL;
//    delete [] flag_latent; flag_latent = NULL;
//    delete [] S;  S = NULL;
//    for(i=0;i<new_corr.size(); ++i)
//    {
//        cout<<"latent x = "<<latent_template.m_minutiae[get<1>(new_corr[i])].x<<" latent y="<<latent_template.m_minutiae[get<1>(new_corr[i])].y<<" rolled x="
//              << rolled_template.m_minutiae[get<2>(new_corr[i])].x << " rolled y="<<rolled_template.m_minutiae[get<2>(new_corr[i])].y<<endl;
//    }
    // delete [] H; H=NULL;
    return new_corr;
};

vector<tuple<float, int, int>>  Matcher::LSS_R_Fast2_Dist_lookup(vector<tuple<float, int, int>> &corr, SingleTemplate & latent_template, SingleTemplate & rolled_template, float d_thr)
{
    int num = corr.size();
    float *H = new float [num*num]();
    //vector<float> H(num*num);
//    vector<bool> H(num*num);
    vector<short> flag_latent(latent_template.m_nrof_minu),flag_rolled(rolled_template.m_nrof_minu);

    register int i,j,k;

    MinuPoint *p_latent_minutia_1, *p_latent_minutia_2, *p_rolled_minutia_1, *p_rolled_minutia_2;
    float dist_1, dist_2, dist;
    int  dx_1, dy_1, dx_2, dy_2;

//    cout<<"minutiae"<<endl;
//    for(i=0; i<num; ++i)
//    {
//        p_latent_minutia_1 = & latent_template.m_minutiae[get<1>(corr[i])];
//        p_rolled_minutia_1 = & rolled_template.m_minutiae[get<2>(corr[i])];
//
//
//        cout<<"x=: " <<(p_latent_minutia_1->x*16+24)<<" y=:"<<(p_latent_minutia_1->y*16+24)<<"  x=: " <<(p_rolled_minutia_1->x*16+24)<<" y=:"<<(p_rolled_minutia_1->y*16+24)<<endl;
//    }
//
    for(i=0; i<num-1; ++i)
    {
        p_latent_minutia_1 = & latent_template.m_minutiae[get<1>(corr[i])];
        p_rolled_minutia_1 = & rolled_template.m_minutiae[get<2>(corr[i])];
        for(j=i+1; j<num;++j)
        {
//            H[i*num+j] = 0.0;
//            H[j*num+i] = 0.0;
            p_latent_minutia_2 = & latent_template.m_minutiae[get<1>(corr[j])];
            p_rolled_minutia_2 = & rolled_template.m_minutiae[get<2>(corr[j])];

            dx_1 = p_latent_minutia_1->x-p_latent_minutia_2->x;
            dx_2 = p_rolled_minutia_1->x-p_rolled_minutia_2->x;

            dx_1 = abs(dx_1);
            dx_2 = abs(dx_2);
//            if( fabs(dx_1-dx_2)>d_thr)
//                continue;
            dy_1 = p_latent_minutia_1->y-p_latent_minutia_2->y;
            dy_2 = p_rolled_minutia_1->y-p_rolled_minutia_2->y;

            dy_1 = abs(dy_1);
            dy_2 = abs(dy_2);
//            if( fabs(dy_1-dy_2)>d_thr)
//                continue;
//

            if(dx_1>=dist_N | dx_2>=dist_N | dy_1>=dist_N | dy_2>=dist_N)
                continue;


            dist_1 = table_dist[dx_1*dist_N+dy_1];//(dx_1*dx_1)+(dy_1*dy_1);
            //dist_1 = sqrt(dist_1);


//            dist_2 = (dx_2*dx_2)+(dy_2*dy_2);
//            dist_2 = sqrt(dist_2);
            dist_2 = table_dist[dx_2*dist_N+dy_2];

            dist = fabs(dist_1-dist_2);
            if(dist>d_thr)
                continue;

            H[i*num+j] = (30-dist)/(25.0);
            if(H[i*num+j]>1)
                H[i*num+j] = 1.0;
            else if(H[i*num+j]<0)
                H[i*num+j] = 0.0;
//            H[j*num+i] = true;
            H[j*num+i] = H[i*num+j];
        }
    }

//    ofstream output;
//    output.open("H_lookup.txt");
//    for(i=0;i<N; ++i)
//    {
//        for(j=0;j<N; ++j)
//        {
//            output<<H[i*N+j]<<" ";
//        }
//        output<<endl;
//    }
//    output.close();

    Matrix<float, Eigen::Dynamic, Eigen::Dynamic> aa =  Map<Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(H,num,num);

    //Matrix<float, Eigen::Dynamic, Eigen::Dynamic> bb =  Map<Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(rolled_minu_template.m_des,rolled_minu_template.m_nrof_minu,des_len);


    //MatrixXf b = Map<MatrixXf>(rolled_minu_template.m_des,rolled_minu_template.m_nrof_minu,des_len);

    float sum = 0.0;
    VectorXf b(num);
    VectorXf c;
    for(i=0; i<num; ++i)
        b(i) = get<0>(corr[i]);
    for(i=0;i<3 ; ++i)
    {
        c = aa*b;
        sum = c.sum();
        b = c*(1./(sum+0.00001));
    }

//    output.open("S_lookup.txt");
    vector<float> S(num);
    for(i=0;i<num; ++i)
    {
        S[i] = b(i);
//        output<<i<<"  "<<S[i]<<endl;
    }
//    output.close();
//

//    vector<float> S(num),S1(num);
//    float s0 = 1.0/num;
//    for(i=0; i<num; ++i)
//        S[i] = get<0>(corr[i]);
//
//
//    float sum = 0.0;
//    for(i=0;i<5 ; ++i)
//    {
//        sum = 0.0;
//        for(j=0;j<num; ++j)
//        {
//            S1[j] = 0;
//            for(k=0; k<num;++k)
//            {
//                if(H[j*num+k])
//                    S1[j] += S[k];
//            }
//            sum += S1[j];
//        }
//        sum = 1.0/(sum+0.0001);
//        for(j=0;j<num; ++j)
//        {
//            S[j] = S1[j]*sum;
//        }
//
////        s0 = 0.0;
////        for(j=0;j<num; ++j)
////        {
////            s0 += S[j];
////        }
////        s0 = 1.0/s0;
////
////        for(j=0;j<num; ++j)
////        {
////           S[j] *= s0;
////        }
//        // cout<<s0<<endl;
//
//    }
//
//


    // sort S
    std::vector<int> y(S.size());
    std::iota(y.begin(), y.end(), 0);
    auto comparator = [&S](int a, int b){ return S[a] > S[b]; };
    std::sort(y.begin(), y.end(), comparator);


    vector<tuple<float, int, int>>  new_corr;
    vector<int>  selected_ind;
    short ind;
    for(i=0; i<num; ++i)
    {

        ind = y[i];
//        cout<<S[ind]<<"  "<<get<1>(corr[ind])<<" "<<get<2>(corr[ind])<<endl;
        if(S[ind]<0.0001)
            break;
        if(flag_latent[get<1>(corr[ind])] == 1 | flag_rolled[get<2>(corr[ind])] == 1)
            continue;

        if(i==0)
        {
            selected_ind.push_back(ind);
            new_corr.push_back(make_tuple(get<0>(corr[ind]),get<1>(corr[ind]),get<2>(corr[ind])));
//            cout<<S[ind]<<"  "<<get<1>(corr[ind])<<" "<<get<2>(corr[ind])<<endl;
//
//            cout<<"S="<<S[ind]<< " latent x = "<<latent_template.m_minutiae[get<1>(corr[ind])].x<<" latent y="<<latent_template.m_minutiae[get<1>(corr[ind])].y<<" latent ori="<<latent_template.m_minutiae[get<1>(corr[ind])].ori
//               << " rolled x="<< rolled_template.m_minutiae[get<2>(corr[ind])].x << " rolled y="<<rolled_template.m_minutiae[get<2>(corr[ind])].y<<
//                " rolled roi="<<rolled_template.m_minutiae[get<2>(corr[ind])].ori<<endl;

            flag_latent[get<1>(corr[ind])] = 1;
            flag_rolled[get<2>(corr[ind])] = 1;
        }
        else
        {
            int found = 0;
            for(j=0;j<selected_ind.size(); ++j)
            {
                if(H[ind*num+selected_ind[j]]<0.00001)
                {
                    found = 1;
                    break;
                }
            }
            if(found==0)
            {
                selected_ind.push_back(ind);
                new_corr.push_back(make_tuple(get<0>(corr[ind]),get<1>(corr[ind]),get<2>(corr[ind])));
        //
//                cout<<"S="<<S[ind]<< " latent x = "<<latent_template.m_minutiae[get<1>(corr[ind])].x<<" latent y="<<latent_template.m_minutiae[get<1>(corr[ind])].y<<" latent ori="<<latent_template.m_minutiae[get<1>(corr[ind])].ori
//                       << " rolled x="<< rolled_template.m_minutiae[get<2>(corr[ind])].x << " rolled y="<<rolled_template.m_minutiae[get<2>(corr[ind])].y<<
//                        " rolled roi="<<rolled_template.m_minutiae[get<2>(corr[ind])].ori<<endl;


                flag_latent[get<1>(corr[ind])] = 1;
                flag_rolled[get<2>(corr[ind])] = 1;
            }
        }
    }


//    for(i=0;i<new_corr.size(); ++i)
//    {
//        cout<<"latent x = "<<latent_template.m_minutiae[get<1>(new_corr[i])].x<<" latent y="<<latent_template.m_minutiae[get<1>(new_corr[i])].y<<" rolled x="
//              << rolled_template.m_minutiae[get<2>(new_corr[i])].x << " rolled y="<<rolled_template.m_minutiae[get<2>(new_corr[i])].y<<endl;
//    }
    delete [] H; H=NULL;
    return new_corr;
};

vector<tuple<float, int, int>>  Matcher::LSS_R_Fast2_Dist_eigen(vector<tuple<float, int, int>> &corr, SingleTemplate & latent_template, SingleTemplate & rolled_template, float d_thr)
{
    int num = corr.size();
    float *H = new float [num*num]();
    //vector<float> H(num*num);

    vector<short> flag_latent(latent_template.m_nrof_minu),flag_rolled(rolled_template.m_nrof_minu);

    register int i,j,k;

    MinuPoint *p_latent_minutia_1, *p_latent_minutia_2, *p_rolled_minutia_1, *p_rolled_minutia_2;
    float dist_1, dist_2, dist;
    float dx_1, dy_1, dx_2, dy_2;

//    cout<<"minutiae"<<endl;
//    for(i=0; i<num; ++i)
//    {
//        p_latent_minutia_1 = & latent_template.m_minutiae[get<1>(corr[i])];
//        p_rolled_minutia_1 = & rolled_template.m_minutiae[get<2>(corr[i])];
//
//
//        cout<<"x=: " <<(p_latent_minutia_1->x*16+24)<<" y=:"<<(p_latent_minutia_1->y*16+24)<<"  x=: " <<(p_rolled_minutia_1->x*16+24)<<" y=:"<<(p_rolled_minutia_1->y*16+24)<<endl;
//    }
//
    for(i=0; i<num-1; ++i)
    {
        p_latent_minutia_1 = & latent_template.m_minutiae[get<1>(corr[i])];
        p_rolled_minutia_1 = & rolled_template.m_minutiae[get<2>(corr[i])];
        for(j=i+1; j<num;++j)
        {
//            H[i*num+j] = 0.0;
//            H[j*num+i] = 0.0;
            p_latent_minutia_2 = & latent_template.m_minutiae[get<1>(corr[j])];
            p_rolled_minutia_2 = & rolled_template.m_minutiae[get<2>(corr[j])];

            dx_1 = p_latent_minutia_1->x-p_latent_minutia_2->x;
            dx_2 = p_rolled_minutia_1->x-p_rolled_minutia_2->x;


            dy_1 = p_latent_minutia_1->y-p_latent_minutia_2->y;
            dy_2 = p_rolled_minutia_1->y-p_rolled_minutia_2->y;


            dist_1 = (dx_1*dx_1)+(dy_1*dy_1);
            dist_1 = sqrt(dist_1);
//            dist_1 = dist_1*16;

            dist_2 = (dx_2*dx_2)+(dy_2*dy_2);
            dist_2 = sqrt(dist_2);
//            dist_2 = dist_2*16;
            dist = fabs(dist_1-dist_2);
           if(dist>d_thr)
                continue;

            H[i*num+j] = (30-dist)/(25.0);
            if(H[i*num+j]>1)
                H[i*num+j] = 1.0;
            else if(H[i*num+j]<0)
                H[i*num+j] = 0.0;

            H[j*num+i] = H[i*num+j];
        }
    }

    Matrix<float, Eigen::Dynamic, Eigen::Dynamic> aa =  Map<Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(H,num,num);

    //Matrix<float, Eigen::Dynamic, Eigen::Dynamic> bb =  Map<Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(rolled_minu_template.m_des,rolled_minu_template.m_nrof_minu,des_len);


    //MatrixXf b = Map<MatrixXf>(rolled_minu_template.m_des,rolled_minu_template.m_nrof_minu,des_len);
//    ofstream output;
//    output.open("H_eigen.txt");
//    for(i=0;i<N; ++i)
//    {
//        for(j=0;j<N; ++j)
//        {
//            output<<H[i*N+j]<<" ";
//        }
//        output<<endl;
//    }
//    output.close();

    float sum = 0.0;
    VectorXf b(num);
    VectorXf c;
    for(i=0; i<num; ++i)
        b(i) = get<0>(corr[i]);
    for(i=0;i<5 ; ++i)
    {
        c = aa*b;
        sum = c.sum();
        b = c*(1./(sum+0.00001));
    }

//    output.open("S_eigen.txt");
    vector<float> S(num);
    for(i=0;i<num; ++i)
    {
        S[i] = b(i);
       // output<<i<<"  "<<S[i]<<endl;
    }
//    output.close();
    // sort the S



    // the sorting part can be replaced by a min-heap
    std::vector<int> y(S.size());
    std::iota(y.begin(), y.end(), 0);
    auto comparator = [&S](int a, int b){ return S[a] > S[b]; };
    std::sort(y.begin(), y.end(), comparator);


    vector<tuple<float, int, int>>  new_corr;
    vector<int>  selected_ind;
    short ind;
    for(i=0; i<num; ++i)
    {

        ind = y[i];
//        cout<<S[ind]<<"  ";
        if(S[ind]<0.0001)
            break;
        if(flag_latent[get<1>(corr[ind])] == 1 | flag_rolled[get<2>(corr[ind])] == 1)
            continue;

        if(i==0)
        {
            selected_ind.push_back(ind);
            new_corr.push_back(make_tuple(get<0>(corr[ind]),get<1>(corr[ind]),get<2>(corr[ind])));
//
//            cout<<"S="<<S[ind]<< " latent x = "<<latent_template.m_minutiae[get<1>(corr[ind])].x<<" latent y="<<latent_template.m_minutiae[get<1>(corr[ind])].y<<" latent ori="<<latent_template.m_minutiae[get<1>(corr[ind])].ori
//               << " rolled x="<< rolled_template.m_minutiae[get<2>(corr[ind])].x << " rolled y="<<rolled_template.m_minutiae[get<2>(corr[ind])].y<<
//                " rolled roi="<<rolled_template.m_minutiae[get<2>(corr[ind])].ori<<endl;

            flag_latent[get<1>(corr[ind])] = 1;
            flag_rolled[get<2>(corr[ind])] = 1;
        }
        else
        {
            int found =0;
            for(j=0;j<selected_ind.size(); ++j)
            {
                if(H[ind*num+selected_ind[j]]<0.00001)
                {
                    found = 1;
                    break;
                }
            }
            if(found==0)
            {
                selected_ind.push_back(ind);
                new_corr.push_back(make_tuple(get<0>(corr[ind]),get<1>(corr[ind]),get<2>(corr[ind])));
        //
//                cout<<"S="<<S[ind]<< " latent x = "<<latent_template.m_minutiae[get<1>(corr[ind])].x<<" latent y="<<latent_template.m_minutiae[get<1>(corr[ind])].y<<" latent ori="<<latent_template.m_minutiae[get<1>(corr[ind])].ori
//                       << " rolled x="<< rolled_template.m_minutiae[get<2>(corr[ind])].x << " rolled y="<<rolled_template.m_minutiae[get<2>(corr[ind])].y<<
//                        " rolled roi="<<rolled_template.m_minutiae[get<2>(corr[ind])].ori<<endl;


                flag_latent[get<1>(corr[ind])] = 1;
                flag_rolled[get<2>(corr[ind])] = 1;
            }
        }
    }

//
//    delete [] H; H = NULL;
//    delete [] flag_rolled; flag_rolled = NULL;
//    delete [] flag_latent; flag_latent = NULL;
//    delete [] S;  S = NULL;
//    for(i=0;i<new_corr.size(); ++i)
//    {
//        cout<<"latent x = "<<latent_template.m_minutiae[get<1>(new_corr[i])].x<<" latent y="<<latent_template.m_minutiae[get<1>(new_corr[i])].y<<" rolled x="
//              << rolled_template.m_minutiae[get<2>(new_corr[i])].x << " rolled y="<<rolled_template.m_minutiae[get<2>(new_corr[i])].y<<endl;
//    }
     delete [] H; H=NULL;
    return new_corr;
};

vector<tuple<float, int, int>>  Matcher::LSS_R_Fast2(vector<tuple<float, int, int>> &corr, SingleTemplate & latent_template, SingleTemplate & rolled_template, int d_thr)
{
    int num = corr.size();
    vector<bool> H(num*num);
    vector<short> flag_latent(latent_template.m_nrof_minu),flag_rolled(rolled_template.m_nrof_minu);

    register int i,j,k;

    MinuPoint *p_latent_minutia_1, *p_latent_minutia_2, *p_rolled_minutia_1, *p_rolled_minutia_2;
    float dist_1, dist_2;
    float angle_1, angle_2, angle_diff;
    float line_angle_1, line_angle_2;
    float dx_1,dx_2,dy_1,dy_2;


    for(i=0; i<num-1; ++i)
    {
        p_latent_minutia_1 = & latent_template.m_minutiae[get<1>(corr[i])];
        p_rolled_minutia_1 = & rolled_template.m_minutiae[get<2>(corr[i])];
        for(j=i+1; j<num;++j)
        {
            p_latent_minutia_2 = & latent_template.m_minutiae[get<1>(corr[j])];
            p_rolled_minutia_2 = & rolled_template.m_minutiae[get<2>(corr[j])];

//            if( p_latent_minutia_1->ori>4.0 | p_latent_minutia_2->ori>4.0)
//                cout<<p_latent_minutia_1->ori << "   "<<p_latent_minutia_2->ori<<endl;
            angle_1 = p_latent_minutia_1->ori-p_latent_minutia_2->ori;
            angle_1 = adjust_angle(angle_1);

            angle_2 = p_rolled_minutia_1->ori-p_rolled_minutia_2->ori;
            angle_2 = adjust_angle(angle_2);

            angle_diff = fabs(angle_1 - angle_2);

            if(angle_diff>PI)
               angle_diff = 2*PI - angle_diff;



            if(angle_diff>PI/4.)
                continue;

            dx_1 = p_latent_minutia_1->x-p_latent_minutia_2->x;
            dy_1 = p_latent_minutia_1->y-p_latent_minutia_2->y;



            line_angle_1 = -atan2(dy_1,dx_1);
            angle_1 = p_latent_minutia_1->ori - line_angle_1;
            angle_1 = adjust_angle(angle_1);


            dx_2 = p_rolled_minutia_1->x-p_rolled_minutia_2->x;
            dy_2 = p_rolled_minutia_1->y-p_rolled_minutia_2->y;

            line_angle_2 = -atan2(dy_2,dx_2);
            angle_2 = p_rolled_minutia_1->ori - line_angle_2;
            angle_2 = adjust_angle(angle_2);

            angle_diff = fabs(angle_1 - angle_2);

            if(angle_diff>PI)
               angle_diff = 2*PI - angle_diff;
            if(angle_diff>PI/6.)
                continue;



            angle_1 = p_latent_minutia_2->ori - line_angle_1;
            angle_1 = adjust_angle(angle_1);


            angle_2 = p_rolled_minutia_2->ori - line_angle_2;
            angle_2 = adjust_angle(angle_2);

            angle_diff = fabs(angle_1 - angle_2);
//
            if(angle_diff>PI)
               angle_diff = 2*PI - angle_diff;
            if(angle_diff>PI/6.)
                continue;

//               not needed because the distance has been checked before
//            if( abs(dx_1-dx_2)>2)
//                continue;

            H[i*num+j] = true;
            H[j*num+i] = true;
        }
    }

    vector<float> S(num),S1(num);

    float s0 = 1.0/num;
    for(i=0; i<num; ++i)
        S[i] = s0;

    float sum = 0.0;
    for(i=0;i<5 ; ++i)
    {
        sum = 0.0;
        for(j=0;j<num; ++j)
        {
            S1[j] = 0;
            for(k=0; k<num;++k)
            {
                if(H[j*num+k])
                   S1[j] += S[k];
            }
            sum += S1[j];
        }
        sum = 1.0/(sum+0.00001);
        for(j=0;j<num; ++j)
        {
            S[j] = S1[j]*sum;
        }
    }

    s0 = 0.0;
//    for(j=0;j<num; ++j)
//    {
//        s0 += S[j];
//    }
//    s0 = 1.0/s0;
//
//    for(j=0;j<num; ++j)
//    {
//       S[j] *= s0;
//    }


    // sort the S



    // the sorting part can be replaced by a min-heap
    std::vector<int> y(S.size());
    std::iota(y.begin(), y.end(), 0);
    auto comparator = [&S](int a, int b){ return S[a] > S[b]; };
    std::sort(y.begin(), y.end(), comparator);


    vector<tuple<float, int, int>>  new_corr;
    vector<int> selected_ind;
    short ind;
    for(i=0; i<num; ++i)
    {
        ind = y[i];
        if(S[ind]<0.001)
            break;
        if(flag_latent[get<1>(corr[ind])] == 1 | flag_rolled[get<2>(corr[ind])] == 1)
            continue;

        // new_corr.push_back(make_tuple(get<0>(corr[ind]),get<1>(corr[ind]),get<2>(corr[ind])));

//        flag_latent[get<1>(corr[ind])] = 1;
//        flag_rolled[get<2>(corr[ind])] = 1;

          if(i==0)
        {
            selected_ind.push_back(ind);
            new_corr.push_back(make_tuple(get<0>(corr[ind]),get<1>(corr[ind]),get<2>(corr[ind])));
//
//


            flag_latent[get<1>(corr[ind])] = 1;
            flag_rolled[get<2>(corr[ind])] = 1;
        }
        else
        {
            int found =0;
            for(j=0;j<selected_ind.size(); ++j)
            {
                if(!H[ind*num+selected_ind[j]])
                {
                    found = 1;
                    break;
                }
            }
            if(found==0)
            {
                selected_ind.push_back(ind);
                new_corr.push_back(make_tuple(get<0>(corr[ind]),get<1>(corr[ind]),get<2>(corr[ind])));
        //
//                cout<<"S="<<S[ind]<< " latent x = "<<latent_template.m_minutiae[get<1>(corr[ind])].x<<" latent y="<<latent_template.m_minutiae[get<1>(corr[ind])].y<<" latent ori="<<latent_template.m_minutiae[get<1>(corr[ind])].ori
//                       << " rolled x="<< rolled_template.m_minutiae[get<2>(corr[ind])].x << " rolled y="<<rolled_template.m_minutiae[get<2>(corr[ind])].y<<
//                        " rolled roi="<<rolled_template.m_minutiae[get<2>(corr[ind])].ori<<endl;


                flag_latent[get<1>(corr[ind])] = 1;
                flag_rolled[get<2>(corr[ind])] = 1;
            }

        }
    }

//
//    delete [] H; H = NULL;
//    delete [] flag_rolled; flag_rolled = NULL;
//    delete [] flag_latent; flag_latent = NULL;
//    delete [] S;  S = NULL;
    return new_corr;
};

float Matcher::adjust_angle(float angle)
{
    if(angle>PI)
        angle -= 2*PI;
    else if (angle<-PI)
    {
        angle += 2*PI;
    }
    return angle;
}

Matcher::~Matcher()
{
    if(codewords!=NULL)
    {
        delete [] codewords;
        codewords = NULL;
    }
}
}
