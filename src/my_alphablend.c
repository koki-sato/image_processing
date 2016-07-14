/**
 * アルファブレンドを行うアプリケーション
 *
 * コンパイルの方法
 * gcc -Wall -O1 my_alphablend.c -o my_alphablend -ljpeg
 * (for OpenMP)
 * gcc -Wall -O1 -fopenmp my_alphablend.c -o my_alphablend -ljpeg
 *
 * 実行方法
 * ./my_alphablend (入力ファイル名1) (入力ファイル名2) (出力ファイル名)
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <jpeglib.h>
#ifdef _OPENMP
#include <omp.h>
#endif

typedef struct _image_t {
    int width;
    int height;
    JSAMPARRAY buf;
} image_t;

/**********************************************************************
 * RGB形式(image_t)の画像のメモリ確保を行う関数
 **********************************************************************/
void new_image(image_t* img, int w, int h) {
    int i;
    img->width = w;
    img->height = h;
    if ((img->buf = malloc(sizeof(JSAMPROW)*h)) == NULL) { exit(EXIT_FAILURE); }
    for (i = 0; i < h; i++) {
        if ((img->buf[i] = malloc(sizeof(JSAMPLE)*3*w)) == NULL) { exit(EXIT_FAILURE); }
    }
}

/**********************************************************************
 * 画像ファイルを読み込む関数
 **********************************************************************/
image_t* read_jpg(char* fname) {
    FILE* ifp;
    image_t* img;
    struct jpeg_decompress_struct cinfo;
    struct jpeg_error_mgr jerr;

    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_decompress(&cinfo);
    if ((ifp = fopen(fname, "rb")) == NULL) {
        fprintf(stderr, "Can't open input file.\n");
        exit(EXIT_FAILURE);
    }
    jpeg_stdio_src(&cinfo, ifp);
    jpeg_read_header(&cinfo, TRUE);
    jpeg_start_decompress(&cinfo);
    if ((img = malloc(sizeof(image_t))) == NULL) { exit(EXIT_FAILURE); }
    new_image(img, cinfo.output_width, cinfo.output_height);
    while (cinfo.output_scanline < cinfo.output_height) {
        jpeg_read_scanlines(&cinfo, img->buf+cinfo.output_scanline,
                            cinfo.output_height-cinfo.output_scanline);
    }
    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);

    return img;
}

/**********************************************************************
 * 画像ファイルを書き出す関数
 **********************************************************************/
void write_jpg(image_t* img, char* fname) {
    FILE* ofp;
    struct jpeg_compress_struct cinfo;
    struct jpeg_error_mgr jerr;

    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_compress(&cinfo);
    if ((ofp = fopen(fname, "wb")) == NULL) {
        fprintf(stderr, "Can't open output file.\n");
        exit(EXIT_FAILURE);
    }
    jpeg_stdio_dest(&cinfo, ofp);
    cinfo.image_width = img->width;
    cinfo.image_height = img->height;
    cinfo.input_components = 3;
    cinfo.in_color_space = JCS_RGB;
    jpeg_set_defaults(&cinfo);
    jpeg_set_quality(&cinfo, 80, TRUE);
    jpeg_start_compress(&cinfo, TRUE);
    jpeg_write_scanlines(&cinfo, img->buf, img->height);
    jpeg_finish_compress(&cinfo);
    jpeg_destroy_compress(&cinfo);
}

/**********************************************************************
 * アルファブレンドを行う関数
 **********************************************************************/
void alphablend_serial(JSAMPARRAY inbuf1, JSAMPARRAY inbuf2, JSAMPARRAY outbuf, int width, int height) {
    int i, j;

    for (i = 0; i < height; i++) {
        for (j = 0; j < width; j++) {
            outbuf[i][j*3]   = (unsigned char)(0.5*inbuf1[i][j*3]   + 0.5*inbuf2[i][j*3]);
            outbuf[i][j*3+1] = (unsigned char)(0.5*inbuf1[i][j*3+1] + 0.5*inbuf2[i][j*3+1]);
            outbuf[i][j*3+2] = (unsigned char)(0.5*inbuf1[i][j*3+2] + 0.5*inbuf2[i][j*3+2]);
        }
    }
}

/**********************************************************************
 * アルファブレンドを行う関数(並列+ループアンローリング)
 **********************************************************************/
#ifdef _OPENMP
void alphablend_parallel(JSAMPARRAY inbuf1, JSAMPARRAY inbuf2, JSAMPARRAY outbuf, int width, int height) {
    int i, j;

#pragma omp parallel for private(j)
    for (i = 0; i < height; i++) {
        for (j = 0; j < width; j += 4) {
          /* (i,j) */
          outbuf[i][j*3]   = (unsigned char)(0.5*inbuf1[i][j*3]   + 0.5*inbuf2[i][j*3]);
          outbuf[i][j*3+1] = (unsigned char)(0.5*inbuf1[i][j*3+1] + 0.5*inbuf2[i][j*3+1]);
          outbuf[i][j*3+2] = (unsigned char)(0.5*inbuf1[i][j*3+2] + 0.5*inbuf2[i][j*3+2]);
          /* (i,j+1) */
          outbuf[i][(j+1)*3]   = (unsigned char)(0.5*inbuf1[i][(j+1)*3]   + 0.5*inbuf2[i][(j+1)*3]);
          outbuf[i][(j+1)*3+1] = (unsigned char)(0.5*inbuf1[i][(j+1)*3+1] + 0.5*inbuf2[i][(j+1)*3+1]);
          outbuf[i][(j+1)*3+2] = (unsigned char)(0.5*inbuf1[i][(j+1)*3+2] + 0.5*inbuf2[i][(j+1)*3+2]);
          /* (i,j+2) */
          outbuf[i][(j+2)*3]   = (unsigned char)(0.5*inbuf1[i][(j+2)*3]   + 0.5*inbuf2[i][(j+2)*3]);
          outbuf[i][(j+2)*3+1] = (unsigned char)(0.5*inbuf1[i][(j+2)*3+1] + 0.5*inbuf2[i][(j+2)*3+1]);
          outbuf[i][(j+2)*3+2] = (unsigned char)(0.5*inbuf1[i][(j+2)*3+2] + 0.5*inbuf2[i][(j+2)*3+2]);
          /* (i,j+3) */
          outbuf[i][(j+3)*3]   = (unsigned char)(0.5*inbuf1[i][(j+3)*3]   + 0.5*inbuf2[i][(j+3)*3]);
          outbuf[i][(j+3)*3+1] = (unsigned char)(0.5*inbuf1[i][(j+3)*3+1] + 0.5*inbuf2[i][(j+3)*3+1]);
          outbuf[i][(j+3)*3+2] = (unsigned char)(0.5*inbuf1[i][(j+3)*3+2] + 0.5*inbuf2[i][(j+3)*3+2]);
        }
    }
}
#endif

/**********************************************************************
 * main関数
 **********************************************************************/
int main(int argc, char** argv) {
    int width, height;
    image_t* in_img1;
    image_t* in_img2;
    image_t* out_img;
    struct timeval stime, etime;
    char *in_file1, *in_file2, *out_file;

    /* パラメータチェック */
    if (argc != 4) {
        printf("Illegal parameters\n");
        exit(EXIT_FAILURE);
    }
    in_file1 = argv[1];
    in_file2 = argv[2];
    out_file = argv[3];

    /* 入力画像の読み込み */
    in_img1 = read_jpg(in_file1);
    in_img2 = read_jpg(in_file2);

    /* 画像の縦横を取得 */
    width  = in_img1->width;
    height = in_img1->height;

    /* 2つの入力画像の縦横が一致していない場合エラー終了 */
    if (width != in_img2->width || height != in_img2->height) {
        printf("Error\n");
        exit(EXIT_FAILURE);
    }

    /* 出力画像のメモリ確保 */
    if ((out_img = malloc(sizeof(image_t))) == NULL) { exit(EXIT_FAILURE); }
    new_image(out_img, width, height);

    /* 計測開始 */
    gettimeofday(&stime, NULL);

#ifdef _OPENMP
    alphablend_parallel(in_img1->buf, in_img2->buf, out_img->buf, width, height);
#else
    alphablend_serial(in_img1->buf, in_img2->buf, out_img->buf, width, height);
#endif

    /* 計測終了 */
    gettimeofday(&etime, NULL);

    /* 処理時間の表示 */
    printf("TIME: %f\n", etime.tv_sec + (double)etime.tv_usec*1e-6
                        -stime.tv_sec - (double)stime.tv_usec*1e-6);

    /* 出力画像の書き込み */
    write_jpg(out_img, out_file);

    return 0;
}
