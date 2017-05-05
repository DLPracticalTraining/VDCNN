#include "CImg.h"
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <fstream>
#include <sstream>

using namespace cimg_library;
using namespace std;

#define CLASS_NUM 20

string src_path = "../images/";
string tar_path = "../processed_data/";
string log_path = "../logs/";
string train_test[2];

const int HEIGHT = 224;
const int WIDTH = 224;
const int CHANNLE = 3;
unsigned char pixels_labels[HEIGHT * WIDTH * CHANNLE + 1];

string int2str(const int& integer) {
    stringstream stream;  
    stream << integer;  
    return stream.str();
}  

int main() {
    train_test[0] = "train";
    train_test[1] = "val";

    for(int i = 0; i < CLASS_NUM; i++) {
        for(int tt = 0; tt < 2; tt++) {
            ifstream logfile;
            string file_path = log_path + int2str(i) + '_' + train_test[tt] + ".txt";
            logfile.open(file_path.c_str());
            string tar_file = tar_path + int2str(i) + '_' + train_test[tt] + ".bin";
            FILE* tarfile = fopen(tar_file.c_str(), "wb+");
            
            string file_name;
            while(getline(logfile, file_name)) {
                string img_path = src_path + file_name + ".jpg";
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

                pixels_labels[0] = (unsigned char) i;
                fwrite(pixels_labels, width * height * channel + 1, 1, tarfile);
            
                // if(img -> _spectrum == 3) {
                //     int ccc = 1;
                //     for(int x = 0; x < width; x++) {
                //         for(int y = 0; y < height; y++) {
                //             for(int z = 0; z < channel; z++) {
                //                 printf("%u %u  ", pixels_labels[ccc], (*img)(x, y, z));
                //                 ccc++;
                //             }
                //         }
                //         printf("\n");
                //     }
                //     printf("%u\n", pixels_labels[ccc]);
                // }
                // img -> display();
                delete img;
            }

            fclose(tarfile);
        }

        printf("Class %d completed.\n", i);
    }
    
    return 0;
}


