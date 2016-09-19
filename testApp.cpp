// testApp.cpp : コンソール アプリケーションのエントリ ポイントを定義します。
//

#include "stdafx.h"
#include <stdio.h>
#include "stdlib.h"
#include <iostream>
#include <fstream>
#include <opencv2\highgui/highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\core\core.hpp>
#include <unordered_map>
//#include "fftw3.h"
//#pragma comment(lib, "libfftw3-3.lib")
//#pragma comment(lib, "libfftw3f-3.lib")
//#pragma comment(lib, "libfftw3l-3.lib")
#define PI 3.141592
#define LOOKUP_SIZE 100                                  //ルックアップテーブルのデフォルトサイズ
#define LABEL_KIND_NUM 12                                 //取得したいラベルの種類数
#define AROUND_PIXEL_X 200                               //現在の座標の周りの探索する際のXの範囲
#define AROUND_PIXEL_Y 50                                //                              Yの範囲
#define ID_COUNT 4                                       //データとなる動画の数
#define COLOR_DECIDE_LENGTH 9                            //色空間を定義するのに必要な要素数 ex){rs, re, gs, ge, bs, be}の配列
#define MODE_KIND 3
#define FEATURE_KIND 2

/*******「誰の」「何の処理か」「特徴量」を設定********/
#define ID 2                                             //0:星野, 1:秀野, 2:羽田, 3:北沢
#define MODE 1                                           //0:ラベリングモード 1:追跡モード 2:再生モード
#define FEATURE 0                                        //0:股の角度、1:膝の角度
#define HIST 1                                           //ヒストグラム出力
#define COLOR 0                                          //色特徴空間生成

using namespace std;
using namespace cv;

/*************定数群(なんか怖いから配列系はconstにした)****************/
const string video_urls[ID_COUNT] = { "Hoshino.avi", "Shuno.avi", "Haneda.avi", "Kitazawa.avi" };
const int use_start_frames[ID_COUNT] = { 400, 210, 532, 1832 };
const int use_frame_nums[ID_COUNT] = { 32, 38, 36, 38 };
const int rgb_thresh = 10;

enum JOINT{
	HEAD = 1,
	NECK = 2,
	LEFT_SHOULDER = 3,
	RIGHT_SHOULDER = 4,
	LEFT_ELBOW = 5,
	RIGHT_ELBOW = 6,
	LEFT_WRIST = 7,
	RIGHT_WRIST = 8,
	ANKLE = 9,
	LEFT_KNEE = 10,
	RIGHT_KNEE = 11,
	LEFT_HEEL = 12,
	RIGHT_HEEL = 13
};

string joint_names[LABEL_KIND_NUM] = {"頭", "左", "a", "b", "c", "d", "e", "f", "g", "h", "k", "l"};

/************グローバル変数群***********/
string video_url;                                            //使用する動画のURL
int use_start_frame;                                         //動画から使う最初のフレーム
int use_frame_num;                                           //使用するフレーム数
int use_end_frame;                                           //動画から使う最後のフレーム
int label_num_by_id[LABEL_KIND_NUM];                         //取得したい関節に該当するラベル番号を格納
unordered_map<int, int> lookup_table;                        //ルックアップテーブル
int latest_label_num = 0;                                    //ラベリングで使用する
int width;                                                   //画像の幅
int height;                                                  //高さ
vector<double> angles;                                       //フレームごとの関節の角度
const string output_labels_filename[ID_COUNT] = { "output_labels_hoshino.txt",  "output_labels_shuno.txt",
"output_labels_haneda.txt", "output_labels_kitazawa.txt" };
int height_min, height_max, width_min, width_max;
int	resized_width, resized_height, resized_mean;

//グローバル変数の初期化
void init_config(){
	try{
		if (ID < 0 || ID >= ID_COUNT){ throw "Exception: IDが範囲外です。"; }
		if (MODE < 0 || MODE >= MODE_KIND){ throw "Exception: MODEが範囲外です。"; }
		if (FEATURE < 0 || FEATURE >= FEATURE_KIND){ throw "Exception: FEATUREが範囲外です。"; }
	}
	catch (char *e){
		cout << e;
	}
	video_url = video_urls[ID];
	use_start_frame = use_start_frames[ID];
	use_frame_num = use_frame_nums[ID];
	use_end_frame = use_start_frame + use_frame_num;
}

struct YRGB{
    int y;
	int r;
	int g;
	int b;
};

//vectorをキーとするハッシュマップを使用するためのクラス
class HashVI{
public:
	size_t operator()(const vector<int> &x) const {
		const int C = 997;      // 素数
		size_t t = 0;
		for (int i = 0; i != x.size(); ++i) {
			t = t * C + x[i];
		}
		return t;
	}
};

/*********************************************************************************
*                                                                                *
*  Labelクラス                                                                   *
*  private:                                                                      *
*    char name : ラベルの名前                                                    *
*    Point cog : ラベルの重心位置                                                *
*    prev_points : 前のフレームの座標                                            *
*    current_points : 現在のフレームの座標                                       *
*  public:                                                                       *
*    デフォルトコンストラクタ                                                    *
*    コンストラクタ                                                              *
*      引数：名前, 現在の座標, 最初ラベルの重心, ラベルの色空間                  *
*      動作：                                                                    *
*	   　・引数の値をメンバ変数にセット                                          *
*		 ・空のprev_pointsを初期化                                               *
*     メンバ関数                                                                 *
*	   ・get_name():名前を返す                                                   *
*	   ・get_color_space():ラベルの色空間取得                                    *
*	   ・get_current_points():現在の座標取得                                     *
*	   ・get_prev_points():1フレーム前の座標取得                                 *
*	   ・get_cog():重心の一覧取得                                                *
*	   ・set_current_points(Point p):pをcurrent_pointsにpush_backする            *
*	   ・set_cog(Point p):pをcogにpush_backする                                  *
*	   ・calc_and_set_cog():current_pointsの座標から重心を計算し、cogにセットする*
*	   ・change_ptr():current_pointsをクリアし、prev_pointsに移す。              *
*	   ・clear_prev_points():prev_pointsをクリアする                             *
*                                                                                *
**********************************************************************************/
class Label{
private:
	int label_id;
	string name;
	vector<Point> cog;
	vector<Point> prev_points;
	vector<Point> current_points;
	Point prev_back_up;
public:
	Label(){}
	Label(int label_id, string name, vector<Point> current_points, Point first_cog)
		: label_id(label_id), name(name), current_points(current_points)
	{
		vector<Point> pp;
		prev_points = pp;
		cog.push_back(first_cog);
	}
	int get_id(){ return label_id; }
	string get_name() { return name; }
	vector<Point> get_current_points(){ return current_points; }
	vector<Point> get_prev_points(){ return prev_points; }
	vector<Point> get_cog(){ return cog; }
	Point get_prev_back_up(){ return prev_back_up; }
	void set_prev_back_up();
	void set_current_points(Point p);
	void set_cog(Point p);
	void calc_and_set_cog();
	void change_ptr();
	void clear_prev_points();
};

void Label::set_current_points(Point p){
	current_points.push_back(p);
}

void Label::set_cog(Point p){
	cog.push_back(p);
}

void Label::calc_and_set_cog(){
	int maxX = 0;
	int minX = 10000;
	int maxY = 0;
	int minY = 10000;
	Point p;
	vector<Point> points = current_points;
	for (auto itr = points.begin(); itr != points.end(); ++itr){
	p = *itr;
	if (p.x > maxX){
	    maxX = p.x;
	}
	if (p.x < minX){
    	minX = p.x;
	}
	if (p.y > maxY){
    	maxY = p.y;
	}
	if (p.y < minY){
    	minY = p.y;
	}
	}
	Point cog{ (maxX + minX) / 2, (maxY + minY) / 2 };
	set_cog(cog);
}

void Label::change_ptr(){
	vector<Point> ptr = current_points;
	current_points = prev_points;
	prev_points = ptr;
}

void Label::set_prev_back_up(){
	if (prev_points.size() != 0){
		prev_back_up = prev_points[0];
	}
}

void Label::clear_prev_points(){
	set_prev_back_up();
	prev_points.clear();
}

unordered_map<vector<int>, int, HashVI> *labels;  //key：座標、value：ラベル番号
vector<int> label_list;                          //全ラベルの一覧
unordered_map<int, Vec3b> label_color_list;      //ラベルごとの色

//座標の正当性チェック
bool point_validation(int x, int y, int width, int height, int dimension = 2, int z = NULL, int depth = NULL,
	int w = NULL, int time = NULL){
	if (dimension == 2){
		if (x < 0 || x > width || y < 0 || y > height) {
			return true;
		}
		else{
			return false;
		}
	}
	else if (dimension == 3){
		if (x < 0 || x > width || y < 0 || y > height || z < 0 || z > depth) {
			return true;
		}
		else{
			return false;
		}
	}
	else if (dimension == 4){
		if (x < 0 || x > width || y < 0 || y > height || z < 0 || z > depth || w < 0 || w > time) {
			return true;
		}
		else{
			return false;
		}
	}
	else{
		try{
			throw("正しい次元を指定してください");
		}
		catch (char *e){
			cout << e;
		}
	}
}

//ランダムなRGB値を返す
Scalar get_label_color(){
	const int MAX_VALUE = 255;
	unsigned int r = rand() % MAX_VALUE;
	unsigned int g = rand() % MAX_VALUE;
	unsigned int b = rand() % MAX_VALUE;
	return Scalar(r, g, b);
}

//最大値と最小値を求める
void change_min_and_max_value(int x, int y, int *max_x, int *max_y,
	int *min_x, int *min_y){
	if (x > *max_x){
		*max_x = x;
	}
	else if (x < *min_x){
		*min_x = x;
	}
	if (y > *max_y){
		*max_y = y;
	}
	else if (y < *min_y){
		*min_y = y;
	}
}

//指定したデータ点がマップ内に存在するか
bool label_exist (vector<int> yrgb){
	auto itr = labels->find(yrgb);
	if (itr != labels->end()){
		return true;
	}
	else{
		return false;
	}
	/*int label = labels[yrgb];
	if (label == 0){
		labels.erase(yrgb);
		return false;
	}
	else{
		return true;
	}*/
}

//教師付き分類
int search_label_with_supervised_classification(vector<int> yrgb){
	const int mask = 11;
	int y, r, g, b, ty, tr, tg, tb;
	y = yrgb[0];
	r = yrgb[1];
	g = yrgb[2];
	b = yrgb[3];
	vector<int> yrgb2;
	double dist;
	double min_dist = 1000000000.0;
	int min_label = -1;

	for (ty = y - (int)(mask / 4); ty <= y + (int)(mask / 4); ty++){
		for (tr = r - (int)(mask / 2); tr <= r + (int)(mask / 2); tr++){
			for (tg = g - (int)(mask / 2); tg <= g + (int)(mask / 2); tg++){
				for (tb = b - (int)(mask / 2); tb <= b + (int)(mask / 2); tb++){
					yrgb2 = {ty, tr, tg, tb};
					if (label_exist(yrgb2)){
						double dist = sqrt((ty - y)*(ty - y) + (tr - r)*(tr - r) + (tg - g)*(tg - g) + (tb - b)*(tb - b));
						if (dist < min_dist){
							min_dist = dist;
							min_label = (*labels)[yrgb2];
						}
					}
					/*
					if (temp_label != 0){
						double dist = sqrt((ty - y)*(ty - y) + (tr - r)*(tr - r) + (tg - g)*(tg - g) + (tb - b)*(tb - b));
						if (dist < min_dist){
							min_dist = dist;
							min_label = temp_label;
						}
					}*/
				}
			}
		}
	}
	return min_label;
}

//y値をresized_heightによって正規化する
int height_normalize(int y){
	return (int)(((double)y - (double)height_min) / (double)resized_mean * 2000);
}

//画像からラベルを探索する
void search_points_from_image(Mat& frame, Label* parts[]){
	int r, g, b, y, label, normal_y;
	const int rgb_thresh = 10;
	Point p;
	Vec3b val;
	vector<int> yrgb;
	for (int y = height_min; y < height_max + 1; y++){
		Vec3b* ptr = frame.ptr<Vec3b>(y);
		for (int x = width_min; x < width_max + 1; x++){
			p = { x, y };
			val = ptr[x];

			normal_y = height_normalize(y);
			r = val[2];
			g = val[1];
			b = val[0];
			yrgb = { normal_y, r, g, b };

			if (r >= rgb_thresh && g >= rgb_thresh && b >= rgb_thresh){
				label = search_label_with_supervised_classification(yrgb);
				if (label != -1){
					parts[label-1]->set_current_points(p);
				}
			}
		}
	}
}

//Labelクラスを初期化(※リファクタリングしたいなー)
void init_label_class(Mat& frame, Label* parts[]){
	int height = frame.rows;
	int width = frame.cols;
	//ラベルごとの最大値の{ankle[x],ankle[y],left_knee[x],left_knee[y],x,y,...,x,y}
	int max_points[LABEL_KIND_NUM*2] = {}; 
	//ラベルごとの最小値の{ankle[x],ankle[y],left_knee[x],left_knee[y],x,y,...,x,y}
	int min_points[LABEL_KIND_NUM*2];
	for (int i = 0; i < LABEL_KIND_NUM*2; i++){ min_points[i] = 100000000; }

	//ラベルごとの座標を保持するvectorを定義
	vector<Point> parts_points[LABEL_KIND_NUM];
	for (int i = 0; i < LABEL_KIND_NUM; i++){
		vector<Point> v;
		parts_points[i] = v;
	}

	//分類
	int normal_y, label;
	uchar r, g, b;
	Point p;
	Vec3b val;
	vector<int> yrgb;
	
	for (int y = height_min; y < height_max+1; y++){
		Vec3b* ptr = frame.ptr<Vec3b>(y);
		for (int x = width_min; x < width_max+1; x++){
			p = { x, y };
			val = ptr[x];

			normal_y = height_normalize(y);
			r = val[2];
			g = val[1];
			b = val[0];
			yrgb = { normal_y, r, g, b };
		
			if (r >= rgb_thresh && g >= rgb_thresh && b >= rgb_thresh){
				label = search_label_with_supervised_classification(yrgb);
				if (label != -1){
					parts_points[label-1].push_back(Point{ x, y });
					change_min_and_max_value(x, y, &max_points[(label-1) * 2], &max_points[(label-1) * 2 + 1], &min_points[(label-1) * 2], &min_points[(label-1) * 2 + 1]);
				}
			}
		}
	}
	
    //それぞれのラベルにおける重心を求める
	Point cogs[LABEL_KIND_NUM];
	int cog_x, cog_y;
	for (int i = 0; i < LABEL_KIND_NUM; i++){
		cog_x = (max_points[i*2] + min_points[i*2]) / 2;
	    cog_y = (max_points[i*2+1] + min_points[i*2+1]) / 2;
		Point cog_point{ cog_x, cog_y };
		cogs[i] = cog_point;
	}

	//全ラベルのLabelクラスの初期化
	for (int i = 0; i < LABEL_KIND_NUM; i++){
		*(parts[i]) = { i, joint_names[i], parts_points[i], cogs[i] };
	}
}

//全ラベルのprev_pointsとcurrent_pointsを入れ替え,current_pointsをクリアする
void change_prev_and_current(Label* parts[]){
	for (int i = 0; i < LABEL_KIND_NUM; i++){
		parts[i]->clear_prev_points();
		parts[i]->change_ptr();
	}
}

void set_cog_each_label(Label* parts[]){
	for (int i = 0; i < LABEL_KIND_NUM; i++){
		parts[i]->calc_and_set_cog();
	}
}

//3点を与えられたときに角度を求める
//c:角度の基準点、a,b:それ以外
void evaluate_angle(Point c, Point a, Point b){
	int cx = c.x;
	int cy = c.y;
	int ax = a.x;
	int ay = a.y;
	int bx = b.x;
	int by = b.y;
	int ax_cx = ax - cx;
	int ay_cy = ay - cy;
	int bx_cx = bx - cx;
	int by_cy = by - cy;
	float cos = ((ax_cx*bx_cx) + (ay_cy*by_cy)) / ((sqrt((ax_cx*ax_cx) + (ay_cy*ay_cy))*sqrt((bx_cx*bx_cx) + (by_cy*by_cy))));
	float angle = acosf(cos);
	if (angle > PI / 2){ angle = PI-angle; }
	angles.push_back(angle);
}

//腰と右膝、左膝から成す角度を求め、anglesにpushする
void evaluate_angle_ankle_and_knees(Point* parts){
//	evaluate_angle(ankle, right_knee, left_knee);
}

void evaluate_front_knee_angle(Point* parts){
//	evaluate_angle(left_knee, ankle, left_heel);
}

//ただの動画再生のためのメソッド
void play(VideoCapture& video){
	int count = 0;
	Mat frame;
	while (1){
		count++;
		video >> frame;
		width = frame.cols;
		height = frame.rows;
		if (frame.empty() || video.get(CV_CAP_PROP_POS_AVI_RATIO) == 1){
			break;
		}
		//対象のフレームまではスキップ
		if (count < use_start_frame){
			continue;
		}
		cout << count << endl;
		imshow("test", frame);
		waitKey(30);
	}
}
vector<YRGB> data;
//画像の前処理（ノイズ除去,リサイズ,など）
Mat resize_and_preproc(Mat& src, bool first=false){
	/*************ノイズ除去*************/
	const int mask = 9;
	Mat filtered_img;
	medianBlur(src, filtered_img, mask);
	/***********画像のリサイズ************/
	int y, x;
	//const int extra_y_size = 20;
	height_min = 1000000000;
	height_max = 0;
	width_min = 1000000000;
	width_max = 0;
	for (y = 0; y < height; y++){
		Vec3b* ptr = filtered_img.ptr<Vec3b>(y);
		for (x = 0; x < width; x++){
			Vec3b c = ptr[x];
			if (c[2] > 20 && c[1] > 20 && c[0] > 20){
				//		rectangle(filtered_img, Point{ x, y }, Point{ x, y }, Scalar(255, 0, 255));
				if (y < height_min){
					height_min = y;
				}
				if (y > height_max){
					height_max = y;
				}
				if (x < width_min){
					width_min = x;
				}
				if (x > width_max){
					width_max = x;
				}
			}
		}
	}
	resized_width = width_max - width_min;
	resized_height = height_max - height_min;
	resized_mean = (height_max + height_min) / 2;
	Mat resized_img(src, Rect(width_min, height_min, resized_width, resized_height));
	/***************クラスタリング用のデータ構築****************/
	if (first){
		YRGB p;
		for (y = height_min; y <= height_max; y++){
			Vec3b* ptr = src.ptr<Vec3b>(y);
			for (x = width_min; x < width_max; x++){
				Vec3b c = ptr[x];
				if (c[2] > 10 && c[1] > 10 && c[0] > 10){
					//change_label_feature_space(y, c[2], c[1], c[0], true);
					//cout << height_normalize(y) << endl;
					p = { height_normalize(y), c[2], c[1], c[0] };
					data.push_back(p);
				}
			}
		}
	}
	return resized_img;
}

//重複している点がないかをチェック
bool check_distinct_points(YRGB *kCenter, YRGB data, int count){
	for (int i = 0; i < count; i++){
		YRGB center = kCenter[i];
		if (center.y == data.y && center.r == data.r && center.g == data.g && center.b == data.b){
			return false;
		}
		else{
			return true;
		}
	}
}

void k_means_clustering(){
	const int k = 12;   //ラベルの数
	const int nStart = 9;
	double minWcv = 10000000000;     //最小のクラスタ内分散和
	YRGB bestCenter[k];
	YRGB p, q;
	unordered_map<vector<int>, int, HashVI> *all_labels;
	all_labels = new unordered_map<vector<int>, int, HashVI>[nStart];
	for (int h = 0; h < nStart; h++){
		if (data.empty()){
			cout << "dataがないよ" << endl;
		}
		unordered_map<vector<int>, int, HashVI> temp_labels;
		all_labels[h] = temp_labels;
		//vector<YRGB> kCenter;
		YRGB kCenter[k];
		YRGB total[k];
		double clsCount[k];
		double dis, disMin;
		int randIndex;
		bool changed = false;
		int nData = data.size();
		YRGB dp;
		int dy, cy, minIndex;
		int dr, dg, db, cr, cg, cb;
		vector<int> yrgb; //clsLabelのためやむを得ず（ほんとはstructでやりたい）

		//クラスタセンタが変化しなくなるまで繰り返す
		int iterCount = 0;
		//初期クラスタの選択(kmeans++)
		int cls_init_count = 0;
		do{
			double init_max_dist = 0.0;
			double init_y, init_r, init_g, init_b, init_temp_dist;
			YRGB init_p, max_p;
			if (cls_init_count == 0){
				randIndex = (int)rand() % (nData + 1);
				kCenter[cls_init_count].y = (int)data[randIndex].y;
				kCenter[cls_init_count].r = (int)data[randIndex].r;
				kCenter[cls_init_count].g = (int)data[randIndex].g;
				kCenter[cls_init_count].b = (int)data[randIndex].b;
				//最大距離計算&2個目のクラスタセンタ決定
				cls_init_count++;
				for (auto itr = data.begin(); itr != data.end(); ++itr){
					init_p = *itr;
					init_y = init_p.y;
					init_r = init_p.r;
					init_g = init_p.g;
					init_b = init_p.b;
					init_temp_dist = sqrt((init_y - kCenter[0].y)*(init_y - kCenter[0].y) + (init_r - kCenter[0].r)*(init_r - kCenter[0].r) + (init_g - kCenter[0].g)*(init_g - kCenter[0].g) + (init_b - kCenter[0].b)*(init_b - kCenter[0].b));
					if (init_temp_dist > init_max_dist){
						max_p = init_p;
						init_max_dist = init_temp_dist;
					}
				}
				kCenter[cls_init_count] = max_p;
			}
			else{
				//それぞれのクラスタセンタに対する最短点を求める
				YRGB second_p, min_p;
				vector<YRGB> temp_min_points;
				vector<double> temp_max_dists;
				double second_y, second_r, second_g, second_b;
				for (int a = 0; a < cls_init_count; a++){
					double init_min_dist = 1000000000000;
					second_p = kCenter[a];
					second_y = second_p.y;
					second_r = second_p.r;
					second_g = second_p.g;
					second_b = second_p.b;
					for (auto itr = data.begin(); itr != data.end(); ++itr){
						init_p = *itr;
						init_y = init_p.y;
						init_r = init_p.r;
						init_g = init_p.g;
						init_b = init_p.b;
						init_temp_dist = sqrt((init_y - second_y)*(init_y - second_y) + (init_r - second_r)*(init_r - second_r) + (init_g - second_g)*(init_g - second_g) + (init_b - second_b)*(init_b - second_b));
						if (init_temp_dist < init_min_dist && init_temp_dist != 0.0 && !check_distinct_points(kCenter, init_p, cls_init_count)){
							min_p = init_p;
							init_min_dist = init_temp_dist;
						}
					}
					temp_min_points.push_back(min_p);
					temp_max_dists.push_back(init_min_dist);
				}
				int index = 0;
				int max_index = 0;
				for (auto itr = temp_max_dists.begin(); itr != temp_max_dists.end(); ++itr){
					double dist = *itr;
					if (dist >= init_max_dist){
						max_index = index;
					}
					index++;
				}
				kCenter[cls_init_count] = temp_min_points[max_index];
			}
			cls_init_count++;
		} while (cls_init_count < k);
		do{
			//クラスタ割り当て
			changed = false;
			for (int i = 0; i < k; i++){
				clsCount[i] = 0;
				total[i].y = 0;
				total[i].r = 0;
				total[i].g = 0;
				total[i].b = 0;
			}
			all_labels[h].clear();
			//各データ点とクラスタセンタ間との距離を計算
			for (int i = 0; i < nData; i++){
				yrgb.clear();
				dp = data[i];
				dy = dp.y;
				dr = dp.r;
				dg = dp.g;
				db = dp.b;
				disMin = 10000000000;
				for (int j = 0; j < k; j++){
					cy = kCenter[j].y;
					cr = kCenter[j].r;
					cg = kCenter[j].g;
					cb = kCenter[j].b;
					dis = sqrt((dy - cy)*(dy - cy) + (dr - cr)*(dr - cr) + (dg - cg)*(dg - cg) + (db - cb)*(db - cb));
					//		cout << dis << endl;
					if (dis != 0){
						if (dis < disMin){
							disMin = dis;
							minIndex = j + 1;
						}
					}
					else{
						minIndex = j + 1;
						break;
					}
				}
				yrgb = { dy, dr, dg, db };
				all_labels[h].insert(make_pair(yrgb, minIndex));
				total[minIndex - 1].y += dy;
				total[minIndex - 1].r += dr;
				total[minIndex - 1].g += dg;
				total[minIndex - 1].b += db;
				clsCount[minIndex - 1]++;
			}
			//新しいクラスタセンタを得る
			//クラスタ内の平均値の算出
			int countMatch = 0;
			YRGB mean[k];
			for (int i = 0; i < k; i++){
				mean[i].y = total[i].y / clsCount[i];
				mean[i].r = total[i].r / clsCount[i];
				mean[i].g = total[i].g / clsCount[i];
				mean[i].b = total[i].b / clsCount[i];
				if (mean[i].y == kCenter[i].y && mean[i].r == kCenter[i].r
					&& mean[i].g == kCenter[i].g && mean[i].b == kCenter[i].b){
					countMatch++;
				}
				kCenter[i] = mean[i];
			}
			//新しいクラスタセンタが同じ点であるかどうかの判定
			if (countMatch == k){
				changed = false;
			}
			else{
				changed = true;
			}
			iterCount++;
		} while (changed);

		//クラスタ内分散和の計算
		int temp_label;
		int var = 0;
		vector<int> tp;
		for (int g = 0; g < nData; g++){
			p = data[g];
			tp = { p.y, p.r, p.g, p.b };
			temp_label = all_labels[h].at(tp);
			q = kCenter[temp_label];
			double v_y = (p.y - q.y)*(p.y - q.y);
			double v_r = (p.r - q.r)*(p.r - q.r);
			double v_g = (p.g - q.g)*(p.g - q.g);
			double v_b = (p.b - q.b)*(p.b - q.b);
			var += round(sqrt(v_y + v_r + v_g + v_b))/1000000000000;
		}
		double wcv = var / nData;
		if (wcv < minWcv){
			minWcv = wcv;
			for (int l = 0; l < k; l++){
				bestCenter[l] = kCenter[l];
			}
			labels = &(all_labels[h]);
		}
		else{
			all_labels[h].clear();
		}
	}
	//メモリの解放
	data.clear();
	data.shrink_to_fit();
}

int _tmain(int argc, _TCHAR* argv[])
{
	init_config();
	//ファイル出力用のファイル名定義
	string* filename;
	filename = new string[use_frame_num];
	ostringstream oss;
	for (int i = 0; i < use_frame_num; i++){
		oss << i << ".png";
		string str = oss.str();
		filename[i] = str;
		oss.str("");
	}

	VideoCapture video(video_url);
	const int video_size = video.get(CV_CAP_PROP_FRAME_COUNT);  //ビデオのフレーム数

	switch (MODE){
	case 2:
		play(video);
		break;
	default:
		break;
	}
	Label parts1, parts2, parts3, parts4, parts5, parts6, parts7, parts8, parts9, parts10, parts11, parts12;
	Label* parts[LABEL_KIND_NUM] = { &parts1, &parts2, &parts3, &parts4, &parts5, &parts6,
		&parts7, &parts8, &parts9, &parts10, &parts11, &parts12 };
	//腰、左膝、右膝、左足首、右足首のLabelインスタンスを宣言

	ofstream ofs("output_angles.txt");
	namedWindow("test");
	
	Mat dst_img, resized_img;
	int count = 0;
	while (1){
		count++;
		Mat& frame = dst_img;
		video >> frame;
		width = frame.cols;
		height = frame.rows;
		if (frame.empty() || video.get(CV_CAP_PROP_POS_AVI_RATIO) == 1){
			break;
		}
		//対象のフレームまではスキップ
		if (count < use_start_frame){
			continue;
		}
		else if (count == use_start_frame){
			resized_img = resize_and_preproc(frame, true);
			k_means_clustering();
			init_label_class(frame, parts);
		}
		else if(count >= use_end_frame){
			//対象となるフレームが終わったらループを抜ける
			break;
		}
		else{
			if (MODE == 1){
				resized_img = resize_and_preproc(frame);
				change_prev_and_current(parts);
				search_points_from_image(frame, parts);
				set_cog_each_label(parts);
			}
			else{
				break;
			}
		}
		/*
		for (int y = 0; y < height; y++){
			Vec3b* ptr = frame.ptr<Vec3b>(y);
			for (int x = 0; x < width; x++){
				vector<int> point{ x, y };
				Vec3b v = ptr[x];
				if (left_knee_color_space[0] < v[2] && left_knee_color_space[1] > v[2] &&
					left_knee_color_space[2] < v[1] && left_knee_color_space[3] > v[1] &&
					left_knee_color_space[4] < v[0] && left_knee_color_space[5] > v[0]){
					ptr[x] = Vec3b(255, 0, 0);
				}
			}
		}
        */
		if (MODE == 1){
	/*		vector<Point> pp = right_knee.get_current_points();
			for (auto itr = pp.begin(); itr != pp.end(); ++itr){
				Point p = *itr;
				rectangle(dst_img, p, p, Scalar(0, 0, 255));
			}*/
	//		rectangle(dst_img, Point{ left_knee.get_cog()[count - use_start_frame].x - (AROUND_PIXEL_X / 2), left_knee.get_cog()[count - use_start_frame].y - (AROUND_PIXEL_Y / 2) },
	//			Point{ left_knee.get_cog()[count - use_start_frame].x + (AROUND_PIXEL_X / 2), left_knee.get_cog()[count - use_start_frame].y + (AROUND_PIXEL_Y / 2) }, Scalar(0, 0, 255));
	/*		circle(dst_img, ankle.get_cog()[count - use_start_frame], 5, Scalar(0, 0, 255), -1);
			circle(dst_img, left_knee.get_cog()[count - use_start_frame], 5, Scalar(0, 255, 0), -1);
			circle(dst_img, right_knee.get_cog()[count - use_start_frame], 5, Scalar(255, 0, 0), -1);
			circle(dst_img, left_heel.get_cog()[count - use_start_frame], 5, Scalar(0, 255, 255), -1);
			circle(dst_img, right_heel.get_cog()[count - use_start_frame], 5, Scalar(255, 0, 255), -1);
			circle(dst_img, head.get_cog()[count - use_start_frame], 5, Scalar(255, 255, 0), -1);*/
			for (int i = 0; i < LABEL_KIND_NUM; i++){
			//	cout << parts[i]->get_cog()[count - use_start_frame].x << ", " << parts[i]->get_cog()[count - use_start_frame].y << endl;
			//	circle(dst_img, parts[i]->get_cog()[count - use_start_frame], 5, get_label_color(), -1);
			}

			Scalar test_colors[LABEL_KIND_NUM] = { { 0, 0, 255 }, { 0, 255, 0 }, { 255, 0, 0}, { 0, 255, 255 },
			{ 255, 255, 0 }, { 255, 0, 255 }, { 0, 0, 125 }, { 0, 125, 0 } ,
			{ 125, 0, 0 }, { 0, 125, 125 }, { 125, 0, 125 }, { 125, 125, 0 } };

			vector<vector<Point>> test_points;
			for (int m = 0; m < LABEL_KIND_NUM; m++){
				vector<Point> test_point = parts[m]->get_current_points();
				test_points.push_back(test_point);
			}
			int n = 0;
			for (auto itr = test_points.begin(); itr != test_points.end(); ++itr){
				vector<Point> test_point = *itr;
				for (auto itr2 = test_point.begin(); itr2 != test_point.end(); ++itr2){
					Point p = *itr2;
					rectangle(dst_img, p, p, test_colors[n]);
				}
				n++;
			}
		}
		try{
			imwrite(filename[count - use_start_frame], dst_img);
		}
		catch (runtime_error& ex){
			printf("failure");
			return 1;
		}
		imshow("test", frame);
		waitKey(30);
	}
/*	if (MODE == 1 && HIST == 0){
		for (int i = 0; i < use_frame_num; i++){
			cout << i << "フレーム目:" << angles[i] << endl;
			ofs << i << ", " << angles[i] << endl;
			circle(dst_img, ankle.get_cog()[i], 1, Scalar(0, 0, 255), -1);
			circle(dst_img, left_knee.get_cog()[i], 1, Scalar(0, 255, 100), -1);
			circle(dst_img, right_knee.get_cog()[i], 1, Scalar(255, 0, 0), -1);
			circle(dst_img, left_heel.get_cog()[i], 1, Scalar(255, 217, 0), -1);
			circle(dst_img, right_heel.get_cog()[i], 1, Scalar(255, 0, 255), -1);
			circle(dst_img, head.get_cog()[i], 1, Scalar(255, 255, 0), -1);
		}
	}*/
	
	try{
		imwrite("出力結果.png", dst_img);
	}
	catch (runtime_error& ex){
		printf("failure");
		return 1;
	}
	/**********フーリエ変換とプロット***********/
	if (MODE == 1 && HIST == 0){
		FILE *fp = _popen("wgnuplot_pipes.exe", "w");
		if (fp == NULL){
			return -1;
		}

		const int N = use_frame_num;

		/*
		fftw_complex *in, *out;
		fftw_plan p;
		in = (fftw_complex *)fftw_malloc(sizeof(fftw_complex)* N);
		out = (fftw_complex *)fftw_malloc(sizeof(fftw_complex)* N);
		p = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);

		for (int i = 0; i<N; i++){
		in[i][0] = angles[i];
		in[i][1] = 0;
		}
		fftw_execute(p);
		*/
		ofstream fout("output.dat");
		//double scale = 1. / N;
		for (int i = 0; i < N; i++){
			//	fout << i << " " << abs(out[i][0] * scale) << " " << abs(out[i][1] * scale) << endl;
			fout << i << " " << angles[i] << endl;
		}
		fout.close();
		/*
		fftw_destroy_plan(p);
		fftw_free(in);
		fftw_free(out);
		*/
		fputs("plot \"output.dat\"", fp);
		fflush(fp);
		cin.get();
		_pclose(fp);
	}
	/**********************************************/

	delete[] filename;

	namedWindow("ラベリング結果");
	imshow("ラベリング結果", dst_img);
	cout << "プログラムの終了" << endl;
	cin.get();
	waitKey(0);
	return 0;
}