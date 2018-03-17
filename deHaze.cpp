#include "stdafx.h"
#include "guidedfilter.h"
#include "deHaze.h"

//#pragma comment( lib, "opencv_world310d.lib" ) 
using namespace std;
using namespace cv;


int _PriorSize = 10;			    //窗口大小,腐蚀时用的窗口大小
double _topbright = 0.001;		//亮度最高的像素比例,固定0.1%
double _w = 0.99;				//w,为保留雾的比例 越大，去雾越多
double lightness = 0.1;			//亮度调节
float t0 = 0.01;				//T(x)的最小值   因为不能让tx小于0 等于0 效果不好
int SizeH = 0;					//图片高度
int SizeW = 0;					//图片宽度
int SizeH_W = 0;				//图片中的像素总 数 H*W
Vec<float, 3> a;				//全球大气的光照值
Mat trans_refine;
Mat dark_out1;
Mat img;

//读入图片
Mat ReadImage(string fileName)
{

	Mat img = imread(fileName);

	SizeH = img.rows;
	SizeW = img.cols;
	SizeH_W = img.rows*img.cols;

	Mat real_img(img.rows, img.cols, CV_32FC3);
	img.convertTo(real_img, CV_32FC3);

	real_img = real_img / 255;

	return real_img;


	//读入图片 并其转换为3通道的矩阵后 
	//除以255 将其RBG确定在0-1之间
}



//计算暗通道
//J^{dark}(x)=min( min( J^c(y) ) )
Mat DarkChannelPrior(Mat img)
{
	Mat dark = Mat::zeros(img.rows, img.cols, CV_32FC1);//新建一个所有元素为0的单通道的矩阵

	for (int i = 0; i<img.rows; i++)
	{
		for (int j = 0; j<img.cols; j++)
		{

			dark.at<float>(i, j) = min(
				min(img.at<Vec<float, 3>>(i, j)[0], img.at<Vec<float, 3>>(i, j)[1]),
				min(img.at<Vec<float, 3>>(i, j)[0], img.at<Vec<float, 3>>(i, j)[2])
				);//就是两个最小值的过程
		}
	}
	erode(dark, dark_out1, Mat::ones(_PriorSize, _PriorSize, CV_32FC1));//这个函数叫腐蚀 做的是窗口大小的模板运算 ,对应的是最小值滤波,即 黑色图像中的一块块的东西

	return dark_out1;//这里dark_out1用的是全局变量，因为在其它地方也要用到
}
Mat DarkChannelPrior_(Mat img)//这个函数在计算tx用到，因为与计算暗通道一样都用到了求最小值的过程，变化不多，所以改了改就用这里了
{
	double A = (a[0] + a[1] + a[2]) / 3.0;//全球大气光照值 此处是3通道的平均值

	Mat dark = Mat::zeros(img.rows, img.cols, CV_32FC1);
	Mat dark_out = Mat::zeros(img.rows, img.cols, CV_32FC1);
	for (int i = 0; i<img.rows; i++)
	{
		for (int j = 0; j<img.cols; j++)
		{
			dark.at<float>(i, j) = min(
				min(img.at<Vec<float, 3>>(i, j)[0] / A, img.at<Vec<float, 3>>(i, j)[1] / A),
				min(img.at<Vec<float, 3>>(i, j)[0] / A, img.at<Vec<float, 3>>(i, j)[2] / A)
				);//同理


		}
	}
	erode(dark, dark_out, Mat::ones(_PriorSize, _PriorSize, CV_32FC1));//同上

	return dark_out;

}


//计算A的值
Vec<float, 3> Airlight(Mat img, Mat dark)//vec<float ,3>表示有3个大小的vector 类型为float
{
	int n_bright = _topbright*SizeH_W;

	Mat dark_1 = dark.reshape(1, SizeH_W);//这里dark_1是一个有图片像素那么多行的矩阵 方便下面循环计算

	vector<int> max_idx;

	float max_num = 0;

	Vec<float, 3> A(0, 0, 0);
	Mat RGBPixcels = Mat::ones(n_bright, 1, CV_32FC3);

	for (int i = 0; i<n_bright; i++)
	{
		max_num = 0;
		max_idx.push_back(max_num);
		for (float * p = (float *)dark_1.datastart; p != (float *)dark_1.dataend; p++)
		{
			if (*p>max_num)
			{
				max_num = *p;//记录光照的最大值

				max_idx[i] = (p - (float *)dark_1.datastart);//位置

				RGBPixcels.at<Vec<float, 3>>(i, 0) = ((Vec<float, 3> *)img.data)[max_idx[i]];//对应 的三个通道的值给RGBPixcels

			}
		}
		((float *)dark_1.data)[max_idx[i]] = 0;//访问过的标记为0，这样就不会重复访问
	}


	for (int j = 0; j<n_bright; j++)
	{

		A[0] += RGBPixcels.at<Vec<float, 3>>(j, 0)[0];
		A[1] += RGBPixcels.at<Vec<float, 3>>(j, 0)[1];
		A[2] += RGBPixcels.at<Vec<float, 3>>(j, 0)[2];

	}//将光照值累加

	A[0] /= n_bright;
	A[1] /= n_bright;
	A[2] /= n_bright;//除以总数   即取所有符合的点的平均值。

	return A;
}


//Calculate Transmission Matrix
Mat TransmissionMat(Mat dark)
{
	double A = (a[0] + a[1] + a[2]) / 3.0;
	for (int i = 0; i < dark.rows; i++)
	{
		for (int j = 0; j < dark.cols; j++)
		{
			double temp = (dark_out1.at<float>(i, j));
			double B = fabs(A - temp);
			//	conut++;
			//cout << conut << endl;
			//if (B==)
			if (B - 0.3137254901960784 < 0.0000000000001)//K=80    80/255=0.31   这里浮点数要这样做减法才能正确的比较
			{
				dark.at<float>(i, j) = (1 - _w*dark.at<float>(i, j))*
					(0.3137254901960784 / (B));//此处为改过的式子部分
			}
			else
			{
				dark.at<float>(i, j) = 1 - _w*dark.at<float>(i, j);
			}
			if (dark.at<float>(i, j) <= 0.2)//保证Tx不失真，因为会以上除出的结果会有不对
			{
				dark.at<float>(i, j) = 0.5;
			}
			if (dark.at<float>(i, j) >= 1)//同上
			{
				dark.at<float>(i, j) = 1.0;
			}

		}
	}

	return dark;
}
Mat TransmissionMat1(Mat dark)
{
	double A = (a[0] + a[1] + a[2]) / 3.0;
	for (int i = 0; i < dark.rows; i++)
	{
		for (int j = 0; j < dark.cols; j++)
		{

			dark.at<float>(i, j) = (1 - _w*dark.at<float>(i, j));

		}
	}

	return dark;
}
//Calculate Haze Free Image
Mat hazefree(Mat img, Mat t, Vec<float, 3> a, float exposure = 0)//此处的exposure的值表示去雾后应该加亮的值。
{
	double AAA = a[0];
	if (a[1] > AAA)
		AAA = a[1];
	if (a[2] > AAA)
		AAA = a[2];
	//取a中的最大的值


	//新开一个矩阵
	Mat freeimg = Mat::zeros(SizeH, SizeW, CV_32FC3);
	img.copyTo(freeimg);

	//两个迭代器，这样的写法可以不用两层循环，比较快点
	Vec<float, 3> * p = (Vec<float, 3> *)freeimg.datastart;
	float * q = (float *)t.datastart;

	for (; p<(Vec<float, 3> *)freeimg.dataend && q<(float *)t.dataend; p++, q++)
	{
		(*p)[0] = ((*p)[0] - AAA) / std::max(*q, t0) + AAA + exposure;
		(*p)[1] = ((*p)[1] - AAA) / std::max(*q, t0) + AAA + exposure;
		(*p)[2] = ((*p)[2] - AAA) / std::max(*q, t0) + AAA + exposure;
	}

	return freeimg;
}


void printMatInfo(char * name, Mat m)
{
	cout << name << ":" << endl;
	cout << "\t" << "cols=" << m.cols << endl;
	cout << "\t" << "rows=" << m.rows << endl;
	cout << "\t" << "channels=" << m.channels() << endl;
}

int deHaze(String fileName, Mat& dst)
{
	clock_t start, finish;
	double time;
	Mat dark_channel;
	Mat trans;
	Mat free_img;
	start = clock();

	Mat raw = imread(fileName);
	if (raw.empty())
		return 0;

	SizeH = raw.rows;
	SizeW = raw.cols;
	SizeH_W = raw.rows*raw.cols;

	Mat real_img(raw.rows, raw.cols, CV_32FC3);
	raw.convertTo(real_img, CV_32FC3);

	real_img = real_img / 255;
	real_img.copyTo(img);

	//计算暗通道
	//cout << "计算暗通道 ..." << endl;
	dark_channel = DarkChannelPrior(img);


	//计算全球光照值
	//cout << "计算A值 ..." << endl;
	a = Airlight(img, dark_channel);


	//计算tx
	//cout << "Reading Refine Transmission..." << endl;
	trans_refine = TransmissionMat(DarkChannelPrior_(img));


	//导向滤波 得到精细的透射率图
	Mat tran = guidedFilter(img, trans_refine, 60, 0.0001);

	//去雾
	//cout << "Calculating Haze Free Image ..." << endl;
	/*
	此处 如果用tran的话就是导向滤波部分
	如果是trans_refine 就没有用导向滤波 效果不是那么的好
	上面第四个参数是用来增加亮度的，0.1比较好
	*/
	free_img = hazefree(img, tran, a, 0.1);
	
	//计算使用时间
	finish = clock();
	time = (double)(finish - start) / CLOCKS_PER_SEC;
	cout << "Total Time Cost: " << time << "s" << endl;

	//显示
	imshow("原图", img);
	imshow("去雾后", free_img);
	//保存图片的代码

	//imwrite("output.jpg", free_img * 255);

	waitKey();
	cout << endl;
	return 1;
}

int deHaze(Mat src, Mat& dst,bool ifFilter)
{
	clock_t start, finish;
	double time;
	Mat dark_channel;
	Mat trans;
	Mat free_img;

	img.release();
	start = clock();

	if (src.empty())
		return 0;

	SizeH = src.rows;
	SizeW = src.cols;
	SizeH_W = src.rows*src.cols;

	Mat real_img(src.rows, src.cols, CV_32FC3);
	src.convertTo(real_img, CV_32FC3);

	real_img = real_img / 255;
	real_img.copyTo(img);

	//计算暗通道
	//cout << "计算暗通道 ..." << endl;
	dark_channel = DarkChannelPrior(img);


	//计算全球光照值
	//cout << "计算A值 ..." << endl;
	a = Airlight(img, dark_channel);


	//计算tx
	//cout << "Reading Refine Transmission..." << endl;
	trans_refine = TransmissionMat(DarkChannelPrior_(img));


	//导向滤波 得到精细的透射率图
	Mat tran = guidedFilter(img, trans_refine, 60, 0.0001);

	//去雾
	//cout << "Calculating Haze Free Image ..." << endl;
	/*
	此处 如果用tran的话就是导向滤波部分
	如果是trans_refine 就没有用导向滤波 效果不是那么的好
	上面第四个参数是用来增加亮度的，0.1比较好
	*/
	if (ifFilter)
		free_img = hazefree(img, tran, a, lightness);
	else
		free_img = hazefree(img, trans_refine, a, lightness);
	//计算使用时间
	finish = clock();
	time = (double)(finish - start) / CLOCKS_PER_SEC;

	free_img.copyTo(dst);
	//cout << "Total Time Cost: " << time << "s" << endl;

	//显示
	//imshow("原图", img);
	//imshow("去雾后", free_img);
	//保存图片的代码

	//imwrite("output.jpg", free_img * 255);

	//waitKey(33);
	//cout << endl;
	return 1;
}

//自适应直方图去雾
void adaptHistEqualize(Mat src, Mat& dst)
{
	cv::Mat clahe_img;
	cv::cvtColor(src, clahe_img, CV_BGR2Lab);
	std::vector<cv::Mat> channels(3);
	cv::split(clahe_img, channels);

	cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
	clahe->setClipLimit(4);
	cv::Mat dst1;
	clahe->apply(channels[0], dst1);
	dst1.copyTo(channels[0]);
	cv::merge(channels, clahe_img);

	cv::cvtColor(clahe_img, dst, CV_Lab2BGR);
}
//中值去雾算法
void MedianBlurHaze(unsigned char * Scan0, int Width, int Height, int Stride, int DarkRadius, int MedianRadius, int P)
{
	int  X, Y, Diff, Min, F;
	unsigned char* Pointer, *DarkP, *FilterP, *FilterPC;
	unsigned char * DarkChannel = (unsigned char*)malloc(Width * Height);
	unsigned char * Filter = (unsigned char*)malloc(Width * Height);
	unsigned char * FilterClone = (unsigned char*)malloc(Width * Height);

	for (Y = 0; Y < Height; Y++)
	{
		Pointer = Scan0 + Y * Stride;
		DarkP = DarkChannel + Y * Width;             // 由实际图像计算得到的图像暗通道     
		for (X = 0; X < Width; X++)
		{
			Min = *Pointer;
			if (Min > *(Pointer + 1)) Min = *(Pointer + 1);
			if (Min > *(Pointer + 2)) Min = *(Pointer + 2);
			*DarkP = (unsigned char)Min;
			DarkP++;
			Pointer += 3;
		}
	}
	memcpy(Filter, DarkChannel, Width * Height);                        // 求全局大气光A时会破坏DarkChannel中的数据

	//MinValue(DarkChannel, Width, Height, Width, DarkRadius);                // 求取暗通道值

	// 利用暗通道来估算全局大气光值A
	int Sum, Value, Threshold = 0;
	int SumR = 0, SumG = 0, SumB = 0, AtomR, AtomB, AtomG, Amount = 0;
	int* Histgram = (int*)calloc(256, sizeof(int));
	for (Y = 0; Y < Width * Height; Y++) Histgram[DarkChannel[Y]]++;
	for (Y = 255, Sum = 0; Y >= 0; Y--)
	{
		Sum += Histgram[Y];
		if (Sum > Height * Width * 0.01)
		{
			Threshold = Y;                                        // 选取暗通道值中前1%最亮的像素区域为候选点
			break;
		}
	}
	AtomB = 0; AtomG = 0; AtomR = 0;
	for (Y = 0, DarkP = DarkChannel; Y < Height; Y++)
	{
		Pointer = Scan0 + Y * Stride;
		for (X = 0; X < Width; X++)
		{
			if (*DarkP >= Threshold)                            //    在原图中选择满足候选点的位置的像素作为计算全局大气光A的信息                        
			{
				SumB += *Pointer;
				SumG += *(Pointer + 1);
				SumR += *(Pointer + 2);
				Amount++;
			}
			Pointer += 3;
			DarkP++;
		}
	}
	AtomB = SumB / Amount;
	AtomG = SumG / Amount;
	AtomR = SumR / Amount;

	memcpy(DarkChannel, Filter, Width * Height);                        // 恢复DarkChannel中的数据
	//MedianBlur(Filter, Width, Height, Width, MedianRadius, 50);          // 步骤1：使用中值滤波平滑，这样处理的重要性是在平滑的同时保留了图像中的边界部分，但是实际这里用中值滤波和用高斯滤波效果感觉差不多
	memcpy(FilterClone, Filter, Width * Height);

	DarkP = DarkChannel;
	FilterP = Filter;
	for (Y = 0; Y < Height * Width; Y++)              //利用一重循环来计算提高速度
	{
		Diff = *DarkP - *FilterP;                    //通过对|DarkP －FilterP |执行中值滤波来估计的局部标准差，这样可以保证标准差估计的鲁棒性
		if (Diff < 0) Diff = -Diff;
		*FilterP = (unsigned char)Diff;
		DarkP++;
		FilterP++;
	}
	//MedianBlur(Filter, Width, Height, Width, MedianRadius, 50);

	FilterPC = FilterClone;
	FilterP = Filter;
	for (Y = 0; Y < Height * Width; Y++)
	{
		Diff = *FilterPC - *FilterP;                    // 步骤2：然后考虑到有较好对比度的纹理区域可能没有雾， 这部分区域就不需要做去雾处理
		if (Diff < 0) Diff = 0;                            // 这里可以这样做是因为在最后有个max(....,0)的过程，
		*FilterP = (unsigned char)Diff;
		FilterPC++;
		FilterP++;
	}

	DarkP = DarkChannel;
	FilterP = Filter;

	for (Y = 0; Y < Height * Width; Y++)
	{
		Min = *FilterP * P / 100;
		if (*DarkP > Min)
			*FilterP = Min;                                // 获得满足约束条件的大气光幕
		else
			*FilterP = *DarkP;
		DarkP++;
		FilterP++;
	}

	FilterP = Filter;
	for (Y = 0; Y < Height; Y++)
	{
		Pointer = Scan0 + Y * Stride;
		for (X = 0; X < Width; X++)
		{
			F = *FilterP++;
			if (AtomB != F)
				Value = AtomB *(*Pointer - F) / (AtomB - F);
			else
				Value = *Pointer;
			//*Pointer++ = Clamp(Value);
			if (AtomG != F)
				Value = AtomG * (*Pointer - F) / (AtomG - F);
			else
				Value = *Pointer;
			//*Pointer++ = Clamp(Value);
			if (AtomR != F)
				Value = AtomR *(*Pointer - F) / (AtomR - F);
			else
				Value = *Pointer;
			//*Pointer++ = Clamp(Value);
		}
	}
	free(Histgram);
	free(Filter);
	free(DarkChannel);
	free(FilterClone);
}
/********************************************************************************
单尺度Retinex图像增强程序
src为待处理图像
sigma为高斯模糊标准差
scale为对比度系数
*********************************************************************************/
void SSR(IplImage* src, int sigma, int scale)
{
	IplImage* src_fl = cvCreateImage(cvGetSize(src), IPL_DEPTH_32F, src->nChannels);
	IplImage* src_fl1 = cvCreateImage(cvGetSize(src), IPL_DEPTH_32F, src->nChannels);
	IplImage* src_fl2 = cvCreateImage(cvGetSize(src), IPL_DEPTH_32F, src->nChannels);
	float a = 0.0, b = 0.0, c = 0.0;
	cvConvertScale(src, src_fl, 1.0, 1.0);//转换范围，所有图像元素增加1.0保证cvlog正常
	cvLog(src_fl, src_fl1);

	cvSmooth(src_fl, src_fl2, CV_GAUSSIAN, 0, 0, sigma);        //SSR算法的核心之一，高斯模糊

	cvLog(src_fl2, src_fl2);
	cvSub(src_fl1, src_fl2, src_fl);//Retinex公式，Log(R(x,y))=Log(I(x,y))-Log(Gauss(I(x,y)))

	//计算图像的均值、方差，SSR算法的核心之二
	//使用GIMP中转换方法：使用图像的均值方差等信息进行变换
	//没有添加溢出判断
	CvScalar mean;
	CvScalar dev;
	cvAvgSdv(src_fl, &mean, &dev, NULL);//计算图像的均值和标准差
	double min[3];
	double max[3];
	double maxmin[3];
	for (int i = 0; i<3; i++)
	{
		min[i] = mean.val[i] - scale*dev.val[i];
		max[i] = mean.val[i] + scale*dev.val[i];
		maxmin[i] = max[i] - min[i];
	}
	float* data2 = (float*)src_fl->imageData;
	for (int i = 0; i<src_fl2->width; i++)
	{
		for (int j = 0; j<src_fl2->height; j++)
		{
			data2[j*src_fl->widthStep / 4 + 3 * i + 0] = 255 * (data2[j*src_fl->widthStep / 4 + 3 * i + 0] - min[0]) / maxmin[0];
			data2[j*src_fl->widthStep / 4 + 3 * i + 1] = 255 * (data2[j*src_fl->widthStep / 4 + 3 * i + 1] - min[1]) / maxmin[1];
			data2[j*src_fl->widthStep / 4 + 3 * i + 2] = 255 * (data2[j*src_fl->widthStep / 4 + 3 * i + 2] - min[2]) / maxmin[2];
		}
	}


	cvConvertScale(src_fl, src, 1, 0);
	cvReleaseImage(&src_fl);
	cvReleaseImage(&src_fl1);
	cvReleaseImage(&src_fl2);
}

/********************************************************************************
多尺度Retinex图像增强程序  （一般选用3尺度）
src为待处理图像
sigma为高斯模糊标准差
scale为对比度系数
*********************************************************************************/
void MSR(IplImage* src, int sigma_1, int sigma_2, int sigma_3, int scale)
{

	IplImage* src_fl = cvCreateImage(cvGetSize(src), IPL_DEPTH_32F, src->nChannels);
	IplImage* src_fl1 = cvCreateImage(cvGetSize(src), IPL_DEPTH_32F, src->nChannels);
	IplImage* src_fl2 = cvCreateImage(cvGetSize(src), IPL_DEPTH_32F, src->nChannels);

	IplImage* src1_fl = cvCreateImage(cvGetSize(src), IPL_DEPTH_32F, src->nChannels);
	IplImage* src1_fl1 = cvCreateImage(cvGetSize(src), IPL_DEPTH_32F, src->nChannels);
	IplImage* src1_fl2 = cvCreateImage(cvGetSize(src), IPL_DEPTH_32F, src->nChannels);

	IplImage* src2_fl = cvCreateImage(cvGetSize(src), IPL_DEPTH_32F, src->nChannels);
	IplImage* src2_fl1 = cvCreateImage(cvGetSize(src), IPL_DEPTH_32F, src->nChannels);
	IplImage* src2_fl2 = cvCreateImage(cvGetSize(src), IPL_DEPTH_32F, src->nChannels);




	cvConvertScale(src, src_fl, 1.0, 1.0);//转换范围，所有图像元素增加1.0保证cvlog正常
	src1_fl = cvCloneImage(src_fl);
	src2_fl = cvCloneImage(src_fl);

	cvLog(src_fl, src_fl1);
	cvLog(src1_fl, src1_fl1);
	cvLog(src2_fl, src2_fl1);

	cvSmooth(src_fl, src_fl2, CV_GAUSSIAN, 0, 0, sigma_1);        //MSR算法的核心之一，高斯模糊
	cvSmooth(src1_fl, src1_fl2, CV_GAUSSIAN, 0, 0, sigma_2);
	cvSmooth(src2_fl, src2_fl2, CV_GAUSSIAN, 0, 0, sigma_3);

	cvLog(src_fl2, src_fl2);
	cvLog(src1_fl2, src1_fl2);
	cvLog(src2_fl2, src2_fl2);

	cvSub(src_fl1, src_fl2, src_fl);//Retinex公式，Log(R(x,y))=Log(I(x,y))-Log(Gauss(I(x,y)))
	cvSub(src1_fl1, src1_fl2, src1_fl1);
	cvSub(src2_fl1, src2_fl2, src2_fl1);

	cvConvertScale(src_fl, src_fl, 1.0 / 3.0, 0.0);//每个尺度对应的权值为1/3
	cvConvertScale(src1_fl1, src1_fl1, 1.0 / 3.0, 0.0);
	cvConvertScale(src2_fl1, src2_fl1, 1.0 / 3.0, 0.0);

	cvAdd(src_fl, src1_fl1, src1_fl1);
	cvAdd(src1_fl1, src2_fl1, src2_fl1);

	//计算图像的均值、方差，MSR算法的核心之二
	//使用GIMP中转换方法：使用图像的均值方差等信息进行变换
	//没有添加溢出判断
	CvScalar mean;
	CvScalar dev;
	cvAvgSdv(src2_fl1, &mean, &dev, NULL);//计算图像的均值和标准差
	double min[3];
	double max[3];
	double maxmin[3];
	for (int i = 0; i<3; i++)
	{
		min[i] = mean.val[i] - scale*dev.val[i];
		max[i] = mean.val[i] + scale*dev.val[i];
		maxmin[i] = max[i] - min[i];
	}
	float* data2 = (float*)src2_fl1->imageData;
	for (int i = 0; i<src2_fl1->width; i++)
	{
		for (int j = 0; j<src2_fl1->height; j++)
		{
			data2[j*src2_fl1->widthStep / 4 + 3 * i + 0] = 255 * (data2[j*src2_fl1->widthStep / 4 + 3 * i + 0] - min[0]) / maxmin[0];
			data2[j*src2_fl1->widthStep / 4 + 3 * i + 1] = 255 * (data2[j*src2_fl1->widthStep / 4 + 3 * i + 1] - min[1]) / maxmin[1];
			data2[j*src2_fl1->widthStep / 4 + 3 * i + 2] = 255 * (data2[j*src2_fl1->widthStep / 4 + 3 * i + 2] - min[2]) / maxmin[2];
		}
	}


	cvConvertScale(src2_fl1, src, 1, 0);
	cvReleaseImage(&src_fl);
	cvReleaseImage(&src_fl1);
	cvReleaseImage(&src_fl2);
	cvReleaseImage(&src1_fl);
	cvReleaseImage(&src1_fl1);
	cvReleaseImage(&src1_fl2);
	cvReleaseImage(&src2_fl);
	cvReleaseImage(&src2_fl1);
	cvReleaseImage(&src2_fl2);

}