#include "iostream"
#include <algorithm>
#include "time.h"
#include "string.h"
#include "io.h"

//#include "opencv2/opencv.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv/cv.h>
#include <opencv/highgui.h>

#define MAX_INT 20000000

using namespace std;
using namespace cv;

//Type of Min and Max value
typedef struct _MinMax
{
	double min;
	double max;
}MinMax;

Mat ReadImage(string fileName);
void rerange();
void fill_x_y();
int find_table(int y);
void locate(int l1, int l2, double l3);
void getL(Mat img);

Vec<float, 3> Airlight(Mat img, Mat dark);
Mat TransmissionMat(Mat dark);
Mat DarkChannelPrior(Mat img);

void RefineTrans(Mat trans);

void printMat(char * name, Mat m);

int deHaze(String fileName, Mat& dst);
int deHaze(Mat src, Mat& dst, bool ifFilter);

//自适应直方图去雾
void adaptHistEqualize(Mat src, Mat& dst);
void SSR(IplImage* src, int sigma, int scale);
void MedianBlurHaze(unsigned char * Scan0, int Width, int Height, int Stride, int DarkRadius, int MedianRadius, int P);
void MSR(IplImage* src, int sigma_1, int sigma_2, int sigma_3, int scale);