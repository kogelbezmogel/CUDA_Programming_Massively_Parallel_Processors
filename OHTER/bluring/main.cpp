#include <iostream>
#include <string>
#include <assert.h>

#include "CImg.h"
#include "to_gray_scale.h"

using namespace cimg_library;


int main() {
    std::string img_path = "/home/kogel/repos/CUDA_learn/monkey.png";

    CImg<float> image;
    image.load(img_path.c_str());

    int height, width, spectrum;
    height = image.height();
    width = image.width();
    spectrum = image.spectrum();

    int img_len = image.width() * image.height() * image.depth() * image.spectrum();
    std::cout << "image size: " << image.width() << " x " <<  image.height() << " x " <<  image.depth() << " x " << image.spectrum() << "\n\n";
    
    // in CImg library img's color are stored as 3 separated planes
    float* img_data = image.data();
    float* img_data_interleaved = new float[img_len];

    for(int i = 0; i < image.width() * image.height(); i++) {
        for(int j = 0; j < image.spectrum(); j++) {
            img_data_interleaved[i * image.spectrum() + j] = img_data[i + j * image.width() * image.height() * image.depth()];
        }
    }

    // in normal it is xyzc
    image.permute_axes("cxyz");
    std::cout << "image size: " << image.width() << " x " <<  image.height() << " x " <<  image.depth() << " x " << image.spectrum() << "\n\n";
    img_data = image.data();

    for(int i = 0; i < img_len; i++) {
        assert(img_data[i] - img_data_interleaved[i] < 1e-4);
    }

    float *g_image_data = new float[image.width() * image.height() * image.depth()];
    convert_to_gray(g_image_data, img_data, img_len);

    CImg<float> gray_img(1, width, height, 1);
    gray_img._data = g_image_data;
    gray_img.permute_axes("yzcx");

    std::cout << "image size: " << gray_img.width() << " x " <<  gray_img.height() << " x " <<  gray_img.depth() << " x " << gray_img.spectrum() << "\n\n";
    
    CImgDisplay main_disp(gray_img, "The image");
    while (!main_disp.is_closed())
    {
        main_disp.wait();
    }

    delete [] img_data_interleaved;
    std::cout << "PASSED\n";
    return 0;
}