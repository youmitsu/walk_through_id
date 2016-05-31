// testApp.cpp : コンソール アプリケーションのエントリ ポイントを定義します。
//

#include "stdafx.h"
#include <stdio.h>
#include "stdlib.h"
#include <iostream>
#include <opencv2\highgui/highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\core\core.hpp>

using namespace std;
using namespace cv;

/***************************************************
*                                                  *
* メソッド名：average_point                        *
* 入力：1フレームのマーカーのx座標とy座標          *
* 出力：平均値(x,y座標)                            *
* 詳細：フレームで得られたxy座標の平均値を計算する *
*                                                  *
****************************************************/
Point average_point(int xsum, int ysum, int count){
	if (count == 0){
		Point p = { 0, 0 };
		return p;
	}
	else{
		Point p = { xsum / count, ysum / count };
		return p;
	}
}

int _tmain(int argc, _TCHAR* argv[])
{
	VideoCapture video("Hoshino.avi");
	int width, height, gray_point, color_point;
	const int rm = 170; //マーカーの色取得に用いるr成分
	const int gm = 170; //マーカーの色取得に用いるg成分
	const int bm = 200; //マーカーの色取得に用いるb成分
	const int video_size = video.get(CV_CAP_PROP_FRAME_COUNT);  //ビデオのフレーム数
	vector<Point> points;

	namedWindow("test");
	while (1){
		Mat frame, gray_frame;
		video >> frame;
		width = frame.rows;
		height = frame.cols;
		if (frame.empty() || video.get(CV_CAP_PROP_POS_AVI_RATIO) == 1){
			break;
		}
		waitKey(0);
		int pcount = 0;
		int xsum = 0;
		int ysum = 0;
		for (int y = 6*frame.rows/10; y < 7*frame.rows/10; y++){
			Vec3b* ptr = frame.ptr<Vec3b>(y);
			for (int x = 0; x < frame.cols; x++){
				Vec3b bgr = ptr[x];
			    if ( bgr[0] <= bm && bgr[1] >= gm && bgr[2] >= rm){
					pcount++;
					xsum += x;
					ysum += y;
		//			ptr[x] = Vec3b(255, 0, 0);
             	}
			}
		}
		//cout << average_point(xsum, ysum, pcount) << endl;
		
		points.push_back(average_point(xsum, ysum, pcount));
		circle(frame, average_point(xsum, ysum, pcount), 5, Scalar(0, 0, 255), -1);
		imshow("test", frame);
		waitKey(30);
	}
	waitKey(0);
	return 0;
}

