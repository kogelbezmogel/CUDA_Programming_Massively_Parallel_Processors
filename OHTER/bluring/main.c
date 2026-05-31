#include <stdio.h>
#include <png.h>
#include "blur.h"

int main(){

    char* image_path = "./monkey.png";

    FILE* fptr = fopen(image_path, "rb");

    png_bytepp row_pointers;
    png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    png_init_io(png_ptr, fptr);
    png_read_png(png_ptr, info_ptr, PNG_TRANSFORM_IDENTITY, NULL);
    row_pointers = png_get_rows(png_ptr, info_ptr);
    png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
    fclose(fptr);

    kernel_run();

    return 0;
}