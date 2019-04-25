/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   main.cpp
 * Author: cori
 *
 * Created on November 4, 2018, 4:33 PM
 */

#include <cstdlib>

#include <stdio.h>
#include <omp.h>
#include <chrono>
#include<vector>
#include<iostream>
#include <type_traits>
#include <iostream>
#include <fstream>
#include "boost/filesystem.hpp"
#include "json.hpp"
#include "argparser.h"
#include "matcher.h"

using namespace std;
using namespace std::chrono;
using json = nlohmann::json;
namespace fs = boost::filesystem;


int main(int argc, char** argv)
{
    # ifdef _OPENMP
        printf("Compiled by an OpenMP-compliant implementation.\n");
    # endif

    fs::path config_file("afis.config");
    ifstream i((fs::current_path().parent_path() / config_file).string());
    json config;
    i >> config;

    ArgParser args(argc, argv);

    string codebook_path;
    if(args.cmdOptionExists("-c")){
        codebook_path = args.getCmdOption("-c");
    }else{
        codebook_path = config["CodebookPath"];
    }

    PQ::Matcher matcher(codebook_path);

    string score_path;
    if(args.cmdOptionExists("-s")){
        score_path = args.getCmdOption("-s");
    }else{
        cout<<"Missing argument for score directory. Using default from afis.config"<<endl;
        score_path = config["ScorePath"];
    }
    fs::create_directory(fs::path(score_path));

    string gallery_path;
    if(args.cmdOptionExists("-g")){
        gallery_path = args.getCmdOption("-g");
    }else{
        cout<<"Missing argument for gallery directory. Using default from afis.config"<<endl;
        gallery_path = config["GalleryTemplateDirectory"];
    }

    if(args.cmdOptionExists("-l")){
        string latent_filename = args.getCmdOption("-l");
        matcher.One2List_matching(latent_filename, gallery_path, score_path);
    }else if(args.cmdOptionExists("-ldir")){
        string latent_dirname = args.getCmdOption("-ldir");
        matcher.List2List_matching(latent_dirname, gallery_path, score_path);
    }else{
        cout<<"Missing argument for latent template or directory. Assuming batch matching, using default directory from afis.config"<<endl;
        string latent_dirname = config["LatentTemplateDirectory"];
        matcher.List2List_matching(latent_dirname, gallery_path, score_path);
    }

    return 0;
}
