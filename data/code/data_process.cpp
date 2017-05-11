#include "CImg.h"
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <fstream>
#include <sstream>

using namespace cimg_library;
using namespace std;

string src_path = "../images/";
string tar_path = "../processed_data/";
string log_path = "../logs/";

const int HEIGHT = 224;
const int WIDTH = 224;
const int CHANNLE = 3;
const int SCATTER_NUM = 10000;
unsigned char pixels_labels[HEIGHT * WIDTH * CHANNLE + 1];

struct picture {
    string path;
    int label;
};

void swap(picture* pic_dict, int left, int right) {
    picture tmp = pic_dict[left];
    pic_dict[left] = pic_dict[right];
    pic_dict[right] = tmp;
}

int main() {
    string train_test[2];
    train_test[0] = "train";
    train_test[1] = "val";

    for(int tt = 0; tt < 2; tt++) {
        ifstream logfile;
        string file_path = log_path + train_test[tt] + ".txt";
        logfile.open(file_path.c_str());
        
        string str_num;
        getline(logfile, str_num); //first line
        int pic_num = atoi(str_num.c_str());
        picture* pic_dict = new picture[pic_num];

        string file_name;
        for(int i = 0; i < pic_num; i++) {
            char path[20];
            getline(logfile, file_name);
            sscanf(file_name.c_str(), "%s%d", path, &pic_dict[i].label);
            pic_dict[i].path = path;
        }

        //scatter
        for(int i = 0; i < SCATTER_NUM; i++) {
            int left = rand() % pic_num;
            int right = rand() % pic_num;
            swap(pic_dict, left, right);
        }

        //data processing
        string tar_file = tar_path + train_test[tt] + ".bin";
        FILE* tarfile = fopen(tar_file.c_str(), "wb+");
        for(int i = 0; i < pic_num; i++) {
            string img_path = src_path + pic_dict[i].path + ".jpg";
            CImg<unsigned char>* img = new CImg<unsigned char>(img_path.c_str());

            int width = 224;
            int height = 224;
            int channel = 3;
            
            for(int x = 0; x < width; x++) {
                for(int y = 0; y < height; y++) {
                    for(int z = 0; z < channel; z++) {
                        if(img -> _spectrum == 3) {
                            pixels_labels[x * height * channel + y * channel + z + 1] = (*img)(y, x, z);
                        }
                        else {
                            pixels_labels[x * height * channel + y * channel + z + 1] = (*img)(y, z);
                        }
                    }
                }
            }

            pixels_labels[0] = (unsigned char) pic_dict[i].label;
            fwrite(pixels_labels, width * height * channel + 1, 1, tarfile);

            // img -> display();
            delete img;
        }

        fclose(tarfile);
    }

    
    return 0;
}


