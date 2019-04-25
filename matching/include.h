/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   include.h
 * Author: cori
 *
 * Created on November 4, 2018, 4:37 PM
 */

#ifndef INCLUDE_H
#define INCLUDE_H

#include<math.h>
#include <memory.h>
#include <vector>
#include <numeric>
using std::vector;
#define PI 3.1415926

class MinuPoint
{
    public:
        int x;
        int y;
        float ori;
        float reliability;
};

enum TemplateType {Texture,Minutiae};
enum FPType {Latent,Rolled};

//template base class
//used for latent and rolled minu templates in the original matcher
class SingleTemplate
{
    public:
        int m_nrof_minu;
        int m_nrof_feature;
        int m_des_length;
        int m_block_size;
        int m_blkH;
        int m_blkW;
        float * m_des;
        MinuPoint * m_minutiae;
        float *m_oimg;
        TemplateType m_template_type;
        SingleTemplate()
        {
            m_nrof_minu = 0;
            m_nrof_feature = 0;
            m_des_length = 0;
            m_template_type = TemplateType::Minutiae;
            m_block_size = 16;
            m_des = NULL;
            m_minutiae = NULL;
            m_oimg = NULL;
            m_blkH = 0;
            m_blkW = 0;
        };
        SingleTemplate(const int nrof_minutiae, const int des_length):m_nrof_minu(nrof_minutiae),m_des_length(des_length)
        {     
            m_des = new float[m_nrof_minu*m_des_length]();
            m_minutiae = new MinuPoint[m_nrof_minu]();
            m_block_size = 16;
        };
        
        

        void initialization(const int nrof_minutiae, const int des_length)
        {
            m_nrof_minu = nrof_minutiae;
            m_des_length = des_length;
            m_des = new float[m_nrof_minu*m_des_length]();
            m_minutiae = new MinuPoint[m_nrof_minu]();
            m_block_size = 16;
        };
        
        SingleTemplate(const SingleTemplate &temp)
        {          
            m_nrof_minu = temp.m_nrof_minu;
            m_des_length = temp.m_des_length;
            //copy minutiae descriptor
            m_des = new float[m_nrof_minu*m_des_length]();
            memcpy (m_des, temp.m_des, sizeof(float)*m_nrof_minu*m_des_length);
            m_minutiae = new MinuPoint[m_nrof_minu]();
            memcpy (m_minutiae, temp.m_minutiae, sizeof(MinuPoint)*m_nrof_minu);
            m_block_size = 16;
            
            m_template_type = temp.m_template_type;
            m_oimg = NULL;
            m_blkH = 0;
            m_blkW = 0;
            if(m_template_type==TemplateType::Minutiae)
            {
                m_blkH = temp.m_blkH;
                m_blkW = temp.m_blkW;
                m_oimg = new float[m_blkH*m_blkW]();
                memcpy (m_oimg, temp.m_oimg, sizeof(float)*m_blkH*m_blkW);
            }
        };
        SingleTemplate& operator=(const SingleTemplate &temp)
        {
            m_nrof_minu = temp.m_nrof_minu;
            m_des_length = temp.m_des_length;
            //copy minutiae descriptor
            m_des = new float[m_nrof_minu*m_des_length]();
            memcpy (m_des, temp.m_des, sizeof(float)*m_nrof_minu*m_des_length);
            m_minutiae = new MinuPoint[m_nrof_minu]();
            memcpy (m_minutiae, temp.m_minutiae, sizeof(MinuPoint)*m_nrof_minu);
            m_block_size = 16;
            
            m_template_type = temp.m_template_type;
            m_oimg = NULL;
            m_blkH = 0;
            m_blkW = 0;
            if(m_template_type==TemplateType::Texture)
            {
                m_blkH = temp.m_blkH;
                m_blkW = temp.m_blkW;
                m_oimg = new float[m_nrof_minu*m_des_length]();
            }
        }
        ~ SingleTemplate()
        {
            m_nrof_minu = 0;
            m_nrof_feature = 0;
            m_des_length = 0;
            m_block_size = 0;
            if(m_des)
            {
                delete [] m_des;
                m_des = NULL;
            }
            if(m_minutiae)
            {
                delete [] m_minutiae;
                m_minutiae = NULL;
            }
            if(m_oimg)
            {
                delete [] m_oimg;
                m_oimg = NULL;
            }
        };
        void release(void)
        {
            m_nrof_minu = 0;
            m_nrof_feature = 0;
            m_des_length = 0;
            m_block_size = 0;
            if(m_des)
            {
                delete [] m_des;
                m_des = NULL;
            }
            if(m_minutiae)
            {
                delete [] m_minutiae;
                m_minutiae = NULL;
            }
            if(m_oimg)
            {
                delete [] m_oimg;
                m_oimg = NULL;
            }
        }
        void set_x(const short *x)
        {
            for(int i=0;i<m_nrof_minu; ++i)
            {
                m_minutiae[i].x = x[i];
            }
        }
        void set_y(const short *y)
        {
            MinuPoint * pMinutiae = m_minutiae;
            for(int i=0;i<m_nrof_minu; ++i,++pMinutiae)
            {
                pMinutiae->y = y[i];
            }
        };
        void set_ori(const float *ori)
        {
            MinuPoint * pMinutiae = m_minutiae;
            for(int i=0;i<m_nrof_minu; ++i,++pMinutiae)
            {
                pMinutiae->ori = ori[i];
            }
        };
        void set_type(const TemplateType template_type){
            m_template_type = template_type;
        };
        void init_des(){
            m_des = new float[m_nrof_minu*m_des_length]();
        }

};

class MinutiaeTemplate:public SingleTemplate{
    public:
        MinutiaeTemplate()
        {
            m_template_type = TemplateType::Minutiae;
        };
        MinutiaeTemplate(const int nrof_minutiae, const int des_length):SingleTemplate(nrof_minutiae, des_length)
        {
            m_template_type = TemplateType::Minutiae;
        };
        // minutiae template initialization
        // minutiae, minutiae descriptor and ridge flow are included in minutiae template
        MinutiaeTemplate(const int nrof_minutiae, const short *x,const short *y,const float *ori,const int des_length, const float *des, const int blkH, const int blkW, const float *oimg):
        SingleTemplate(nrof_minutiae, des_length)
        {           
            m_template_type = TemplateType::Minutiae;
            
            SingleTemplate::m_blkH = blkH;
            SingleTemplate::m_blkW = blkW;
            
            // minutiae descriptor
            memcpy (m_des, des, sizeof(float)*m_nrof_minu*m_des_length);
            
            // minutiae
            set_x(x);
            set_y(y);
            set_ori(ori);
            
            m_oimg = NULL;
            
            // orientation field 
            m_oimg = new float[m_blkH*m_blkW]();
            memcpy (m_oimg, oimg, sizeof(float)*m_blkH*m_blkW);
            
        };
        void initialization(const int nrof_minutiae, const int des_length)
        {
            // release();
            m_nrof_minu = nrof_minutiae;
            m_des_length = des_length;
            m_des = new float[m_nrof_minu*m_des_length]();
            m_minutiae = new MinuPoint[m_nrof_minu]();
            m_block_size = 16;
        };
        
        ~ MinutiaeTemplate()
        {

        };
};

class TextureTemplate: public SingleTemplate
{
    public:
        TextureTemplate()
        {
            m_template_type = TemplateType::Texture;
        };
        TextureTemplate(const int nrof_minutiae, const int des_length):SingleTemplate(nrof_minutiae, des_length)
        {
            m_template_type = TemplateType::Texture;
        };
        // texture template initialization
        // Only minutiae and minutiae descriptors are included in texture template
        TextureTemplate(const int nrof_minutiae, const short *x,const short *y,const float *ori, const int des_length, const float *des):SingleTemplate(nrof_minutiae, des_length)
        {           
            //release();
            m_template_type = TemplateType::Texture;
            
            if(des){
                memcpy(m_des, des, sizeof(float)*m_nrof_minu*m_des_length);
            }
            set_x(x);
            set_y(y);
            set_ori(ori);
            
            m_oimg = NULL;
            m_blkH = 0;
            m_blkW = 0;
        };
        void initialization(const int nrof_minutiae, const int des_length)
        {
            m_nrof_minu = nrof_minutiae;
            m_des_length = des_length;
            m_des = new float[m_nrof_minu*m_des_length]();
            m_minutiae = new MinuPoint[m_nrof_minu]();
            m_block_size = 16;
        };
        
        ~ TextureTemplate()
        {

        };
};

class LatentTextureTemplate: public TextureTemplate
{
    public:
        float *m_dist_codewords;
        LatentTextureTemplate()
        {
            m_dist_codewords = NULL;
        };
        LatentTextureTemplate(const int nrof_minutiae, const int des_length):TextureTemplate(nrof_minutiae,des_length)
        {
           
            m_des = new float[m_nrof_minu*m_des_length]();
            m_minutiae = new MinuPoint[m_nrof_minu]();
            m_block_size = 16;
        };
        LatentTextureTemplate(const int nrof_minutiae, const short *x,const short *y,const float *ori, const int des_length, const float *des):
        TextureTemplate(nrof_minutiae, x, y, ori, des_length, des)
        {           
        };
        void initialization(const int nrof_minutiae, const int des_length, const int nrof_subs,  const int nrof_clusters)
        {
            m_nrof_minu = nrof_minutiae;
            m_des_length = des_length;
            m_des = new float[m_nrof_minu*m_des_length]();
           
            m_minutiae = new MinuPoint[m_nrof_minu]();
            m_block_size = 16;
        };
        
         void compute_dist_to_codewords( float *codewords, const int nrof_subs, const int sub_dim,  const int nrof_clusters)
        {
            m_dist_codewords = new float[m_nrof_minu*nrof_subs*nrof_clusters](); 
            
            float *pdes0, *pdes1, *pdes2; 
            float *pword0, *pword1, *pword2;
            int i, j, k, q;
            float dist = 0.0;
            for(i=0; i<m_nrof_minu ; ++i)
            {
                pdes0 = m_des +  i*m_des_length;
                
                for(j=0;j<nrof_subs ; ++j)
                {
                    pdes1 = pdes0 + j*sub_dim;
                    pword0 = codewords + j*nrof_clusters*sub_dim;
                    float min_v = 1000;
                    for(q=0; q<nrof_clusters; ++q)
                    {
                        pword1 = pword0 + q*sub_dim;
                        dist = 0.0; 
                        for(k=0, pdes2 = pdes1,pword2 = pword1; k<sub_dim; ++k,++pdes2,++pword2)
                        {
                            dist += (*pdes2-*pword2)* (*pdes2-*pword2);
                        }
                        m_dist_codewords[i*nrof_subs*nrof_clusters+j*nrof_clusters+q] = (dist);
                    }
                    
                }
            }


        };
        
        ~LatentTextureTemplate()
        {
        };
};

class RolledTextureTemplatePQ:public TextureTemplate
{
    public:
        unsigned char * m_desPQ;
        RolledTextureTemplatePQ()
        {
            m_nrof_minu = 0;
            m_nrof_feature = 0;
            m_des_length = 0;
            m_block_size = 16;
            m_des = NULL;
            m_minutiae = NULL;
        };
        RolledTextureTemplatePQ(const int nrof_minutiae, const int des_length):TextureTemplate(nrof_minutiae, des_length)
        {
           
            m_desPQ = new unsigned char[m_nrof_minu*m_des_length]();
        };
        
        RolledTextureTemplatePQ(const RolledTextureTemplatePQ & input_template)
        {
            m_nrof_minu = input_template.m_nrof_minu;
            m_des_length = input_template.m_des_length;
            
            m_block_size = input_template.m_block_size;
            m_block_size = input_template.m_block_size;
                    
            m_desPQ = new unsigned char[m_nrof_minu*m_des_length]();
            memcpy(m_desPQ, input_template.m_desPQ, sizeof(unsigned char)*m_nrof_minu*m_des_length);
            
            m_minutiae = new MinuPoint[m_nrof_minu]();
            memcpy(m_minutiae, input_template.m_minutiae, sizeof(MinuPoint)*m_nrof_minu);
        };
        
        
        RolledTextureTemplatePQ(const int nrof_minutiae, const short *x,const short *y,const float *ori, const int des_length, const float *des):
        TextureTemplate(nrof_minutiae, x, y, ori, des_length, NULL)
        {
            m_desPQ = new unsigned char[m_nrof_minu*m_des_length]();
            memcpy(m_desPQ, des, sizeof(char)*m_nrof_minu*m_des_length);
        };
        void initialization(const int nrof_minutiae, const int des_length)
        {
            // release();
            m_nrof_minu = nrof_minutiae;
            m_des_length = des_length;
            m_desPQ = new unsigned char[m_nrof_minu*m_des_length]();
            m_minutiae = new MinuPoint[m_nrof_minu]();
            m_block_size = 16;
        };
        
        ~RolledTextureTemplatePQ()
        {
            m_nrof_minu = 0;
            m_nrof_feature = 0;
            m_des_length = 0;
            m_block_size = 0;
            if(m_des)
            {
                delete [] m_des;
                m_des = NULL;
            }
            if(m_minutiae)
            {
                delete [] m_minutiae;
                m_minutiae = NULL;
            }
            if(m_desPQ)
            {
                delete [] m_desPQ;
                m_desPQ = NULL;
            }
        };
        void release(void)
        {
            m_nrof_minu = 0;
            m_nrof_feature = 0;
            m_des_length = 0;
            m_block_size = 0;
            if(m_des)
            {
                delete [] m_des;
                m_des = NULL;
            }
            if(m_minutiae)
            {
                delete [] m_minutiae;
                m_minutiae = NULL;
            }
            if(m_desPQ)
            {
                delete [] m_desPQ;
                m_desPQ = NULL;
            }
        }
        void set_x(short *x)
        {
            for(int i=0;i<m_nrof_minu; ++i)
            {
                m_minutiae[i].x = x[i];
            }
        }
        void set_y(short *y)
        {
            MinuPoint * pMinutiae = m_minutiae;
            for(int i=0;i<m_nrof_minu; ++i,++pMinutiae)
            {
                pMinutiae->y = y[i];
            }
        };
        void set_ori(float *ori)
        {
            MinuPoint * pMinutiae = m_minutiae;
            for(int i=0;i<m_nrof_minu; ++i,++pMinutiae)
            {
                pMinutiae->ori = ori[i];
            }
        };

};

class FPTemplate
{
    public:
        int m_nrof_minu_templates;
        int m_nrof_texture_templates;
        vector<MinutiaeTemplate> m_minu_templates;
        FPType m_FP_type;
        FPTemplate(FPType FP_type):m_FP_type(FP_type)
        {
            m_nrof_minu_templates = 0;
            m_nrof_texture_templates = 0;
        };
        void add_template(const MinutiaeTemplate & minutiae_template)
        {

            m_minu_templates.push_back(minutiae_template);
            m_nrof_minu_templates++;

        };
        virtual void add_texture_template(const TextureTemplate & texture_template){};
        ~FPTemplate()
        {
            release();
        };
        void release()
        {
            m_minu_templates.clear();
            m_nrof_minu_templates = 0;
            m_nrof_texture_templates = 0;
        }
};

class LatentFPTemplate:public FPTemplate{
public: 
    vector<LatentTextureTemplate> m_texture_templates;
    LatentFPTemplate():FPTemplate(Latent){};
    void release()
    {
        FPTemplate::release();
        m_texture_templates.clear();
    }
    void add_texture_template(const LatentTextureTemplate &texture_template)
    {
        m_texture_templates.push_back(texture_template);
        m_nrof_texture_templates++;
    };
    void release_texture_templates(){
        m_texture_templates.clear();
        m_nrof_texture_templates = 0;
    }
};

class RolledFPTemplate:public FPTemplate{
public: 
    vector<RolledTextureTemplatePQ> m_texture_templates;
    RolledFPTemplate():FPTemplate(Rolled){};
    void release()
    {
        FPTemplate::release();
        m_texture_templates.clear();
    }
    void release_texture_templates(){
        m_texture_templates.clear();
        m_nrof_texture_templates = 0;
    }
    void add_texture_template(const RolledTextureTemplatePQ & texture_template)
    {
        RolledTextureTemplatePQ texture_template_new(texture_template);
        m_texture_templates.push_back(texture_template_new);
        m_nrof_texture_templates++;
    };
};


#endif /* INCLUDE_H */
