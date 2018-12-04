// OpenImgTest.cpp : 定义控制台应用程序的入口点。

#include "stdafx.h"

//OpenCV 头文件包含
#include <stdio.h>
#include "cv.h"
#include <fstream>
#include "highgui.h"
#include "cxcore.h"
#include "cv.hpp"
#include "opencv.hpp"
#include "selective_search.hpp"
#include "opencv2/core/core.hpp"  
#include "opencv2/highgui/highgui.hpp"  
#include "opencv2/imgproc/imgproc.hpp"  
#include <iostream>
//图像拼接 
#include <opencv2\stitching\stitcher.hpp>

using namespace  cv;
using namespace  std;
//////////////////////////////////////////////////////////////////////////////////
//自定义函数声明
//void SetROI_IplImage(IplImage *img);
//void OpenVideoFile();
//void OpenUSB();
//int TestImgPingJie();


//说明：选择搜索
int SelectiveSearch( )
{
	//遍历目录下图像文件来获取图像名称
	string dir_path = "F:\\刑侦2800\\cars\\";
	string Newdir_path = "F:\\库\\课件\\OpenCV\\OpenCV实验课程序\\OpenImgTest\\OpenImgTest\\缩减后图片文档\\刑侦2800\\cars\\";
	
	Directory dir;
	vector<string> fileNames = dir.GetListFiles(dir_path, "*.jpg", false);

	for (int i = 0; i < fileNames.size(); i++)
	{
		//1 获得图像文件名 get image name
		
		string fileName = fileNames[i];
		string fileFullName = dir_path + fileName;
		//string WritefileFullName = Newdir_path + fileName;


		cout << "File name:" << fileName << endl;
		cout << "Full path:" << fileFullName << endl;

		//2 加载图像 load image  
		cv::Mat srcimg = cv::imread(fileFullName, cv::IMREAD_COLOR);

		//尺寸调整  
		int H, W, S = 600;
		H = srcimg.rows; W = srcimg.cols;
		int T = H;
		if (W > T) T = W;
		if (T > S)
		{
			float t = float(S) / float(T);
			H = int(H*t);  W = int(W*t);
		}

		Mat img;
		resize(srcimg, img, Size(W, H), 0, 0, INTER_LINEAR);
		//存图像
		//imwrite(fileName, img);

		// selective search
		//auto proposals = ss::selectiveSearch( img, 500, 0.8, 50, 20000, 100000, 2.5 );
		auto proposals = ss::selectiveSearch(img, 500, 0.8, 50, 5000, 120000, 5.0);

		

		//2 建立Txt文件,不同图片不同文档
		FILE *fp;
		char tname[20];
		strcpy(tname, fileName.c_str());
	    strcat(tname, ".txt");
		fp = fopen(tname, "w+");

		int k = 1;
		for (auto &&rect : proposals)
		{
				cv::rectangle(img, rect, cv::Scalar(0, 255, 0), 3, 8);
				//保存图片
				Mat image_cut = Mat(img, rect);
				Mat image_copy = image_cut.clone();
				string Img_Name = "F:\\库\\课件\\OpenCV\\OpenCV实验课程序\\OpenImgTest\\OpenImgTest\\裁剪后图片\\" + fileName + to_string(k)+".jpg";
				k += 1;
				imwrite(Img_Name, image_copy);
	
				//3 X y H W写入Txt
			    fprintf(fp, "%d %d %d %d\n", rect.x, rect.y, rect.width, rect.height);		
		}
		//4 关闭Txt
		fclose(fp);
		cv::imshow("result", img);
		//cv::waitKey(0);
	}
	return true;
}


//////////////////////////////////////////////////////////////////////////////////
int _tmain(int argc, _TCHAR* argv[])
{
	SelectiveSearch();
	//TestImgPingJie();
	//OpenUSB();
	//OpenVideoFile();
	return 0;
}





	/*
	//说明：用传统OpenCV 数据类型打开与显示图像 
	char FileName[]="F:\\邱论文\\邱——第一篇论文\\ScSPM\\image\\刑侦2800\\car\\.jpg";




    IplImage *img=cvLoadImage(FileName);
    cvNamedWindow("Picture", 0);
    cvShowImage("Picture", img);

	SetROI_IplImage(img);

   	//说明：用OpenCV2 Mat数据类型打开与显示图像
	Mat Img2=imread(FileName);
	imshow("MatImg",Img2);
	
	 
	cvWaitKey(0);
    cvReleaseImage(&img); //内存释放
    cvDestroyWindow("Picture"); //关闭窗口
    
	Img2.release();

    return 0;
	
}
*/
/*
//////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////
//说明1：设置兴趣区
void SetROI_IplImage(IplImage *img)
{
	//1）将图片lena.png中感兴趣区域（左上角坐标：180,210；右下角坐标：380,410）在原图上用红色标注；
    CvRect R;
	R.x=180;  R.y=210; R.width=380-180;  R.height=410-210;
	cvSetImageROI(img, R);//设置ROI  
    cvAddS(img, cvScalar(0,0,200), img);//将蓝色通道增加150 
	cvResetImageROI(img);//释放ROI，否则，只会显示ROI区域
	cvShowImage("SetROI", img);

	//2）将感兴趣区域放大两倍，并显示；
	//2.1 剪取ROI区域
	IplImage *RoiImg; 
	RoiImg=cvCreateImage(cvSize(R.width,R.height),8,3);//创建图像空间 
	cvSetImageROI(img, R);//设置ROI  
	cvCopy(img,RoiImg);  //提取ROI  
	cvResetImageROI(img);//释放ROI，否则，只会显示ROI区域
	cvShowImage("ROI", RoiImg);

	//2.2 放大二部
	CvSize size;
    double scale=2;
    size.width=RoiImg->width*scale;
    size.height=RoiImg->height*scale;
	IplImage *RoiImg2; 
    //创建图片并缩放
    RoiImg2=cvCreateImage(size,RoiImg->depth,RoiImg->nChannels);
		// ·CV_INTER_NN - 最近-邻居插补
		// ·CV_INTER_LINEAR - 双线性插值（默认方法）
		// ·CV_INTER_AREA - 像素面积相关重采样。当缩小图像时，该方法可以避免波纹的出现。当放大图像时，类似于方法CV_INTER_NN。
		// ·CV_INTER_CUBIC - 双三次插值。)
    cvResize( RoiImg,RoiImg2,CV_INTER_AREA);
	cvShowImage("ROI二倍放大", RoiImg2);

	//3）将放大后图像RoiImg2第44行的第30-70个像素置为蓝色，并显示。
	for (int r=40;r<=44;r++)
		for (int c=30;c<=170;c++)
		{
			long p=r*RoiImg2->widthStep+c*3;
			RoiImg2->imageData[p++]=(uchar)255;
			RoiImg2->imageData[p++]=(uchar)0;
			RoiImg2->imageData[p++]=(uchar)0;
		}
	cvShowImage("画兰线", RoiImg2);


	cvWaitKey(0);
	//释放恋情与图像窗口
    cvReleaseImage(&RoiImg);    
    cvReleaseImage(&RoiImg2);    
    cvDestroyWindow("ROI二倍放大"); 
    cvDestroyWindow("SetROI");
    cvDestroyWindow("ROI"); 
    cvDestroyWindow("画兰线"); 
}



void processiamge(Mat &frame)  
{  
    circle(frame, Point(cvRound(frame.cols / 2), cvRound(frame.rows / 2)), 150, Scalar(0, 0, 255), 2, 8);  
} 
//说明2：打开播放视频文件
void OpenVideoFile()
{
    string filename = "E:\\E_编书DSP图像处理\\DSP图像处理\\试验样图\\test.avi";//打开的视频文件  
    VideoCapture capture;  
    capture.open(filename);  
  
    double rate = capture.get(CV_CAP_PROP_FPS);//获取视频文件的帧率  
    int delay = cvRound(1000.000 / rate);  
  
    if (!capture.isOpened())//判断是否打开视频文件  
    {  
		printf("%s/n","视频文件打不开....");
        return;  
    }  
          
    while (true)  
    {  
        Mat frame;  
        capture >> frame;//读出每一帧的图像  
        if (frame.empty()) break;  
        imshow("处理前视频", frame);  
        processiamge(frame);  
        imshow("处理后视频", frame);  
        if (waitKey(delay)==27)
			break;
	}  

	//关闭视频文件
    capture.release();
}


//说明3：打开USB摄像头，且录制视频文件
void OpenUSB()
{
	//写入视频文件名
    std::string outFlie = "D:/1.avi";
    //视频写入对象
    //cv::VideoWriter writer;

    VideoCapture capture(0);//如果是笔记本，0打开的是自带的摄像头，1 打开外接的相机  
    double rate = 20;//视频的帧率  
    //获得帧的宽高
    int w = static_cast<int>(capture.get(CV_CAP_PROP_FRAME_WIDTH));
    int h = static_cast<int>(capture.get(CV_CAP_PROP_FRAME_HEIGHT));
	
	Size videoSize(w,h);  
    VideoWriter writer("D:\\VideoTest.avi", CV_FOURCC('D', 'I', 'V', 'X'), rate, videoSize);  
    //打开视频文件，准备写入
    //writer.open(outFlie, -1, rate, videoSize, true);
	
	Mat frame;  
  
    while (capture.isOpened())  
    {  
        capture >> frame;  
        //writer << frame;  
		writer.write(frame);
        imshow("video", frame);  
        if (waitKey(20) == 27)//27是键盘摁下esc时，计算机接收到的ascii码值  
        {  
            break;  
        }  
    }  

	//关闭视频文件
	writer.release();
	capture.release();


}

//说明：OpenCv实现两幅图像的拼接
int TestImgPingJie()
{
    string srcFile[3] = { "E:\\A西安邮电学院\\1课件PPT\\2017-2018-1\\2研究生OpenCV实验\\PPT\\图像拼接图像\\100-0038_img.jpg",
		                  "E:\\A西安邮电学院\\1课件PPT\\2017-2018-1\\2研究生OpenCV实验\\PPT\\图像拼接图像\\100-0039_img.jpg", 
		                  "E:\\A西安邮电学院\\1课件PPT\\2017-2018-1\\2研究生OpenCV实验\\PPT\\图像拼接图像\\100-0040_img.jpg"};
    
	string dstFile = "E:\\A西安邮电学院\\1课件PPT\\2017-2018-1\\2研究生OpenCV实验\\PPT\\图像拼接图像\\result.jpg";
    vector<Mat> imgs;
    for (int i = 0; i<3; ++i)
	{
        Mat img = imread(srcFile[i]);
        if (img.empty())
        {
            cout << "Can't read image '" << srcFile[i] << "'\n";
            //system("pause");
            return -1;
        }
        imgs.push_back(img);
    }
    cout << "Please wait..." << endl;

    Mat pano;
	// 调用createDefault函数生成默认的参数  
    Stitcher stitcher = Stitcher::createDefault(false);
    //Stitcher stitcher = Stitcher::createDefault(try_use_gpu);  
  
	 // 使用stitch函数进行拼接  
	 Stitcher::Status status = stitcher.stitch(imgs, pano);  

    if (status != Stitcher::OK)
    {
        cout << "拼接失败 Can't stitch images, error code=" << int(status) << endl;
        system("pause");
        return -1;
    }
    imwrite(dstFile, pano);
    namedWindow("图像拼接结果");
     ("图像拼接结果", pano);

    waitKey(0);

    destroyWindow("图像拼接结果");
    //system("pause");
    return(0);
}

*/


/*
	//打开摄像头
    cv::VideoCapture captrue(0);
    //视频写入对象
    cv::VideoWriter write;
    //写入视频文件名
    std::string outFlie = "D:/1.avi";
    //获得帧的宽高
    int w = static_cast<int>(captrue.get(CV_CAP_PROP_FRAME_WIDTH));
    int h = static_cast<int>(captrue.get(CV_CAP_PROP_FRAME_HEIGHT));
    cv::Size S(w, h);
    //获得帧率
    double r = captrue.get(CV_CAP_PROP_FPS);
    //打开视频文件，准备写入
    write.open(outFlie, -1, r, S, true);

    //打开失败
    if (!captrue.isOpened())
    {
        return 1;
    }
    bool stop = false;
    cv::Mat frame;
    //循环
    while (!stop)
    {
        //读取帧
        if (!captrue.read(frame))
            break;
        cv::imshow("Video", frame);
        //写入文件
        write.write(frame);
        if (cv::waitKey(10) > 0)
        {
            stop = true;
        }
    }
    //释放对象
    captrue.release();
    write.release();



*/


/*
#include <cv.h>   
#include <highgui.h>     
#include <string>   
#include <iostream>   
#include <algorithm>   
#include <iterator>  
  
#include <stdio.h>  
#include <string.h>  
#include <ctype.h>  
  
using namespace cv;  
using namespace std;  
  
void help()  
{  
    printf(  
            "\nDemonstrate the use of the HoG descriptor using\n"  
            "  HOGDescriptor::hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());\n"  
            "Usage:\n"  
            "./peopledetect (<image_filename> | <image_list>.txt)\n\n");  
}  
  
int main(int argc, char** argv)  
{  
    Mat img;  
    FILE* f = 0;  
    char _filename[1024];  
      
    if( argc == 1 )  
    {  
        printf("Usage: peopledetect (<image_filename> | <image_list>.txt)\n");  
        return 0;  
    }  
      
    img = imread(argv[1]);  
  
    if( img.data )  
    {  
        strcpy(_filename, argv[1]);  
    }  
    else  
    {  
        f = fopen(argv[1], "rt");  
        if(!f)  
        {  
            fprintf( stderr, "ERROR: the specified file could not be loaded\n");  
            return -1;  
        }  
    }  
  
    HOGDescriptor hog;  
    hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());//得到检测器  
    namedWindow("people detector", 1);  
  
    for(;;)  
    {  
        char* filename = _filename;  
        if(f)  
        {  
            if(!fgets(filename, (int)sizeof(_filename)-2, f))  
                break;  
            //while(*filename && isspace(*filename))  
            //  ++filename;  
            if(filename[0] == '#')  
                continue;  
            int l = strlen(filename);  
            while(l > 0 && isspace(filename[l-1]))  
                --l;  
            filename[l] = '\0';  
            img = imread(filename);  
        }  
        printf("%s:\n", filename);  
        if(!img.data)  
            continue;  
          
        fflush(stdout);  
        vector<Rect> found, found_filtered;  
        double t = (double)getTickCount();  
        // run the detector with default parameters. to get a higher hit-rate  
        // (and more false alarms, respectively), decrease the hitThreshold and  
        // groupThreshold (set groupThreshold to 0 to turn off the grouping completely).  
        hog.detectMultiScale(img, found, 0, Size(8,8), Size(32,32), 1.05, 2);  
        t = (double)getTickCount() - t;  
        printf("tdetection time = %gms\n", t*1000./cv::getTickFrequency());  
        size_t i, j;  
        for( i = 0; i < found.size(); i++ )  
        {  
            Rect r = found[i];  
            for( j = 0; j < found.size(); j++ )  
                if( j != i && (r & found[j]) == r)  
                    break;  
            if( j == found.size() )  
                found_filtered.push_back(r);  
        }  
        for( i = 0; i < found_filtered.size(); i++ )  
        {  
            Rect r = found_filtered[i];  
            // the HOG detector returns slightly larger rectangles than the real objects.  
            // so we slightly shrink the rectangles to get a nicer output.  
            r.x += cvRound(r.width*0.1);  
            r.width = cvRound(r.width*0.8);  
            r.y += cvRound(r.height*0.07);  
            r.height = cvRound(r.height*0.8);  
            rectangle(img, r.tl(), r.br(), cv::Scalar(0,255,0), 3);  
        }  
        imshow("people detector", img);  
        int c = waitKey(0) & 255;  
        if( c == 'q' || c == 'Q' || !f)  
            break;  
    }  
    if(f)  
        fclose(f);  
    return 0;  
}  


#include "cv.h"  
#include "highgui.h"  
  
#include <stdio.h>  
#include <stdlib.h>  
#include <string.h>  
#include <assert.h>  
#include <math.h>  
#include <float.h>  
#include <limits.h>  
#include <time.h>  
#include <ctype.h>  
using namespace std;  
  
static CvMemStorage* storage = 0;  
static CvHaarClassifierCascade* cascade = 0;  
  
void detect_and_draw( IplImage* image );  
  
const char* cascade_name =  
"G:/OpenCV2.3.1/data/haarcascades/haarcascade_frontalface_alt.xml";  
/* "haarcascade_profileface.xml";*/  
  /*
int main()  
{  
    CvCapture* capture = 0;  
  
    cascade = (CvHaarClassifierCascade*)cvLoad( cascade_name, 0, 0, 0 );  
  
    if( !cascade )  
    {  
        fprintf( stderr, "ERROR: Could not load classifier cascade/n" );  
        //fprintf( stderr,  
            //"Usage: facedetect --cascade=/"<cascade_path>"/[filename|camera_index]/n" );  
        return -1;  
    }  
    storage = cvCreateMemStorage(0);  
  
  
    cvNamedWindow( "result", 1 );  
  
  
    const char* filename = "H:/test/face05.jpg";  
    IplImage* image = cvLoadImage(filename );  
  
    if( image )  
    {  
        detect_and_draw( image );  
        cvWaitKey(0);  
        cvReleaseImage( &image );  
    }  
  
    cvDestroyWindow("result");  
    cvWaitKey(0);  
    return 0;  
}  
    */
   /* 
void detect_and_draw( IplImage* img )  
{  
    static CvScalar colors[] =   
    {  
        {{0,0,255}},  
        {{0,128,255}},  
        {{0,255,255}},  
        {{0,255,0}},  
        {{255,128,0}},  
        {{255,255,0}},  
        {{255,0,0}},  
        {{255,0,255}}  
    };  
  
    double scale = 1.3;  
    IplImage* gray = cvCreateImage( cvSize(img->width,img->height), 8, 1 );  
    IplImage* small_img = cvCreateImage( cvSize( cvRound (img->width/scale),  
        cvRound (img->height/scale)),  
        8, 1 );  
    int i;  
  
    cvCvtColor( img, gray, CV_BGR2GRAY );  
    cvResize( gray, small_img, CV_INTER_LINEAR );  
    cvEqualizeHist( small_img, small_img );  
    cvClearMemStorage( storage );  
   */
   /*  
    if( cascade )  
    {  
        double t = (double)cvGetTickCount();  
        CvSeq* faces = cvHaarDetectObjects( small_img, cascade, storage,  
            1.1, 2, 0/*CV_HAAR_DO_CANNY_PRUNING,   
            cvSize(30, 30) );  
        t = (double)cvGetTickCount() - t;  
        printf( "detection time = %gms/n", t/((double)cvGetTickFrequency()*1000.) );  
        for( i = 0; i < (faces ? faces->total : 0); i++ )  
        {  
            CvRect* r = (CvRect*)cvGetSeqElem( faces, i );  
            CvPoint center;  
            int radius;  
            center.x = cvRound((r->x + r->width*0.5)*scale);  
            center.y = cvRound((r->y + r->height*0.5)*scale);  
            radius = cvRound((r->width + r->height)*0.25*scale);  
            cvCircle( img, center, radius, colors[i%8], 3, 8, 0 );  
        }  
    }  
    */
   /* 
    cvShowImage( "result", img );  
    cvReleaseImage( &gray );  
    cvReleaseImage( &small_img );  
} 
  */
   /* 

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/gpu/gpu.hpp>
//#include <stdio.h>


using namespace cv;

int main(int argc, char** argv)
{
	Mat img;
	vector<Rect> people;
	img = imread("xingren.jpg",1);

	//定义HOG对象，采用默认参数，或者按照下面的格式自己设置
	HOGDescriptor defaultHog;
		//(cv::Size(64, 128), cv::Size(16, 16), cv::Size(8, 8), 
								//cv::Size(8, 8),9, 1, -1, 
								//cv::HOGDescriptor::L2Hys, 0.2, true, 
								//cv::HOGDescriptor::DEFAULT_NLEVELS);

	//设置SVM分类器，用默认分类器
	defaultHog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());

	//对图像进行多尺度行人检测，返回结果为矩形框
	defaultHog.detectMultiScale(img, people,0,Size(8,8),Size(0,0),1.03,2);

	//画长方形，框出行人
	for (int i = 0; i < people.size(); i++)
	{
		Rect r = people[i];
		rectangle(img, r.tl(), r.br(), Scalar(0, 0, 255), 3);
	}

	namedWindow("检测行人", CV_WINDOW_AUTOSIZE);
	imshow("检测行人", img);
	waitKey(0);

	return 0;
}


1. cv::Mat -> IplImage
cv::Mat matimg = cv::imread ("heels.jpg");
IplImage* iplimg;
*iplimg = IplImage(matimg);
2. IplImage -> cv::Mat
IplImage* iplimg = cvLoadImage("heels.jpg");
cv::Mat matimg;
matimg = cv::Mat(iplimg);

3.动态地址+at()访问元素

#include<opencv2\opencv.hpp>   
#include<opencv2\highgui\highgui.hpp>

using namespace std;
using namespace cv;
  */
   /* 
int main(int argc, char** argv)
{
    Mat img = imread("lena.jpg",1); 
    Mat img1 = img.clone();
    int div = 64;
   // 方法3：用at访问  

    // 访问多通道元素  //
    int rows = img1.rows;
    int cols = img1.cols;
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            //在这里访问每个通道的元素,注意，成员函数at(int y,int x)的参数
            img1.at<Vec3b>(i,j)[0] = img1.at<Vec3b>(i, j)[0] / div*div + div / 2;
            img1.at<Vec3b>(i, j)[1] = img1.at<Vec3b>(i, j)[1] / div*div + div / 2;
            img1.at<Vec3b>(i, j)[2] = img1.at<Vec3b>(i, j)[2] / div*div + div / 2;

        }
    }

    imshow("lena", img1);

   //访问单通道元素 
    Mat img2;
    cvtColor(img, img2, COLOR_RGB2GRAY);

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            //在这里访问每个通道的元素,注意，成员函数at(int y,int x)的参数
            img2.at<uchar>(i, j) = img2.at<uchar>(i, j) / div*div + div / 2;
        }
    }

    imshow("lena2", img2);

    waitKey(0);
    return 0;
}
  */
   /* 
#include<opencv2\opencv.hpp>   
#include<opencv2\highgui\highgui.hpp>

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
    Mat img = imread("lena.jpg", 1); 
    if (img.empty())
    {
        cout << "fail to read image" << endl;
        return -1;
    }
    Mat img1 = img.clone();
    int div = 64;

   // 方法1：用指针访问 //
    //多通道访问法1
    int rows = img1.rows;
    int cols = img1.cols; 
    for (int i = 0; i < rows; i++)
    {
        //uchar* p = img1.ptr<uchar>(i);  //获取第i行的首地址
        for (int j = 0; j < cols; j++)
        {
            //在这里操作具体元素
            uchar *p = img1.ptr<uchar>(i, j);
            p[0] = p[0] / div*div + div / 2;
            p[1] = p[1] / div*div + div / 2;
            p[2] = p[2] / div*div + div / 2;
        }
    }

    imshow("lean", img1);


    //多通道访问法2
    Mat img3 = img.clone();
    int channels = img3.channels(); //获取通道数
    int rows3 = img3.rows;
    int cols3 = img3.cols* channels; //注意，是列数*通道数
    for (int i = 0; i < rows3; i++)
    {
        uchar* p = img3.ptr<uchar>(i);  //获取第i行的首地址
        for (int j = 0; j < cols3; j++)
        {
            //在这里操作具体元素
            p[j] = p[j] / div*div + div / 2;
            p[j+1] = p[j+1] / div*div + div / 2;
            p[j+2] = p[j+2] / div*div + div / 2;
        }
    }

    imshow("lean3", img3);

    //单通道图像
    Mat img2 = img.clone();
    cvtColor(img2, img2, COLOR_BGR2GRAY);
    for (int i = 0; i < img2.rows; i++)
    {
        uchar* p = img2.ptr<uchar>(i);  //获取第i行的首地址
        for (int j = 0; j < img2.cols; j++)
        {
            //在这里操作具体元素
            p[j] = p[j] / div*div + div / 2;
        }
    }

    imshow("lean2", img2);
    waitKey(0);
    return 0;
}
  */
   /* 
MatA.at<int>(1, 1) = 0;

1.一般的Mat定义方法：cv::Mat M(height,width,<Type>)，例：

　　cv::Mat M(480,640,CV_8UC3); 表示定义了一个480行640列的矩阵，矩阵的每个单元的由三个(C3:3 Channel)8位无符号整形(U Unsigned U8 8位)构成。

2.将已有数组赋给Mat矩阵的方法：

　　cv::Mat M = cv::Mat(height,width,<Type>,data)，例：

    float K[3][3] = {fc[0], 0, cc[0], 0, fc[1], cc[1], 0, 0, 1};    //摄像机内参数矩阵K
    cv::Mat mK = cv::Mat(3,3,CV_32FC1,K);    //内参数K Mat类型变量
3.类似matlab：zeros(),ones(),eyes()的初始化方法:

　　cv::Mat M = cv::Mat::eye(height,width,<Type>)

　　cv::Mat M = cv::Mat::ones(height,width,<Type>)

　　cv::Mat M = cv::Mat::zeros(height,width,<Type>)

4.对于小矩阵给定数值的赋值方法：
  */
   /* 
     
#include <iostream>      
#include <opencv2/opencv.hpp>      
      
      
int main(int argc, char** argv)      
{      
    cv::Mat image = cv::imread("test.bmp");      
    if (image.empty())      
    {      
        std::cout<<"read image failed"<<std::endl;      
    }      
          
    // 1. 定义HOG对象      
    cv::HOGDescriptor hog(cv::Size(48, 96), cv::Size(16, 16), cv::Size(8, 8), cv::Size(8, 8), 9, 1,-1, cv::HOGDescriptor::L2Hys, 0.2, true, cv::HOGDescriptor::DEFAULT_NLEVELS);     
          
      
    // 2. 设置SVM分类器      
    hog.setSVMDetector(cv::HOGDescriptor::getDaimlerPeopleDetector());   // 采用已经训练好的行人检测分类器      
      
    // 3. 在测试图像上检测行人区域      
    std::vector<cv::Rect> regions;      
    hog.detectMultiScale(image, regions, 0, cv::Size(8,8), cv::Size(32,32), 1.05, 1);      
      
    // 显示      
    for (size_t i = 0; i < regions.size(); i++)      
    {      
        cv::rectangle(image, regions[i], cv::Scalar(0,0,255), 2);      
    }      
      
    cv::imshow("hog", image);      
    cv::waitKey(0);      
      
    return 0;      
} 

*/



