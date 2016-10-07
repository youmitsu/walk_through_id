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
#define LABEL_KIND_NUM 13                                 //取得したいラベルの種類数
#define AROUND_PIXEL_X 200                               //現在の座標の周りの探索する際のXの範囲
#define AROUND_PIXEL_Y 50                                //                              Yの範囲
#define ID_COUNT 4                                       //データとなる動画の数
#define COLOR_DECIDE_LENGTH 9                            //色空間を定義するのに必要な要素数 ex){rs, re, gs, ge, bs, be}の配列
#define MODE_KIND 3
#define FEATURE_KIND 2
#define EXTRA 10
#define PARTS_LENGTH LABEL_KIND_NUM+EXTRA
#define LABEL_SIZE_THRESH 20
#define COLOR_KIND 4

/*******「誰の」「何の処理か」「特徴量」を設定********/
#define ID 1                                             //0:星野, 1:秀野, 2:羽田, 3:北沢
#define MODE 1                                           //0:ラベリングモード 1:追跡モード 2:再生モード
#define FEATURE 0                                        //0:股の角度、1:膝の角度
#define HIST 1                                           //ヒストグラム出力
#define COLOR 0                                          //色特徴空間生成

using namespace std;
using namespace cv;

/*************定数群(なんか怖いから配列系はconstにした)****************/
const string video_urls[ID_COUNT] = { "Hoshino.avi", "Shuno.avi", "Haneda.avi", "Kitazawa.avi" };
const int use_start_frames[ID_COUNT] = { 400, 210, 568, 1832 };
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

enum COLOR_DATA{
	GREEN = 1,
	YELLOW = 2,
	PINK = 3,
	BLUE = 4
};

/************グローバル変数群***********/
string video_url;                                            //使用する動画のURL
int use_start_frame;                                         //動画から使う最初のフレーム
int use_frame_num;                                           //使用するフレーム数
int use_end_frame;                                           //動画から使う最後のフレーム
int label_num_by_id[LABEL_KIND_NUM];                         //取得したい関節に該当するラベル番号を格納
unordered_map<int, int> lookup_table;                        //ルックアップテーブル
vector<double> angles;                                       //フレームごとの関節の角度
const string output_labels_filename[ID_COUNT] = { "output_labels_hoshino.txt",  "output_labels_shuno.txt",
"output_labels_haneda.txt", "output_labels_kitazawa.txt" };

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

struct XYRGB{
	int x;
    int y;
	int r;
	int g;
	int b;
};

//関節のモデルを定義
/************順番**************
   1:頭
   2:首
   3:左肩
   4:右肘
   5:右手首
   6:左肘
   7:左手首
   8:腰
   9:左膝
   10:右膝
   11:左足首
   12:右足首
*******************************/
/*
XYRGB joint_position_models[LABEL_KIND_NUM] = { { 112, 28, 120, 230, 150 }, { 125, 88, 179, 155, 202 }, { 156, 131, 56, 121, 174 },
{ 86, 215, 131, 250, 201 }, { 8, 260, 147, 127, 178 }, { 160, 240, 255, 255, 161 }, { 173, 368, 255, 255, 166 }, { 106, 279, 76, 153, 200 }, { 57, 491, 215, 238, 137 },
{ 130, 495, 133, 243, 179 }, { 50, 610, 215, 238, 137 }, { 120, 610, 147, 127, 178 } };*/

Point joint_position_models[LABEL_KIND_NUM] = { {67, 32 }, { 58, 129 }, { 65, 188 }, { 46, 151 }, { 36, 270 },
{ 3, 353 }, { 57, 329 }, { 83, 304 }, { 82, 460 }, { 29, 573 }, { 70, 573 }, { 15, 750 }, {84, 734} };

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
	int minX;
	int minY;
	bool is_parts;
	int color_type;
public:
	Label(){}
	Label(vector<Point> current_points, Point first_cog, int minX, int minY)
		: current_points(current_points), minX(minX), minY(minY), is_parts(false), color_type(0)
	{
		vector<Point> pp;
		prev_points = pp;
		cog.push_back(first_cog);
	}
	int get_id(){ return label_id; }
	string get_name() { return name; }
	vector<Point> get_current_points(){ return current_points; }
	int current_size(){ return current_points.size(); }
	vector<Point> get_prev_points(){ return prev_points; }
	vector<Point> get_cog(){ return cog; }
	Point get_prev_back_up(){ return prev_back_up; }
	int get_minY(){ return minY; };
	int get_minX(){ return minX; };
	void set_prev_back_up();
	void set_current_points(Point p);
	void set_cog(Point p);
	void calc_and_set_cog();
	void change_ptr();
	void clear_prev_points();
	void set_joint_mean(int id, string name);
	void change_is_parts(){ is_parts = true; }
	bool get_is_parts(){ return is_parts; }
	int get_color_type(){ return color_type; }
	void set_color_type(int input_type);
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

void Label::set_joint_mean(int id, string joint_name){
	label_id = id;
	name = joint_name;
}

void Label::set_color_type(int input_type){
	color_type = input_type;
}

unordered_map<vector<int>, int, HashVI> labels;  //key：座標、value：ラベル番号
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
Scalar get_random_color(){
	const int MAX_VALUE = 255;
	unsigned int r = rand() % MAX_VALUE;
	unsigned int g = rand() % MAX_VALUE;
	unsigned int b = rand() % MAX_VALUE;
	return Scalar(r, g, b);
}

Vec3b get_label_color(){
	const int MAX_VALUE = 255;
	unsigned int r = rand() % MAX_VALUE;
	unsigned int g = rand() % MAX_VALUE;
	unsigned int b = rand() % MAX_VALUE;
	return Vec3b(r, g, b);
}

//周辺座標のラベルを取得
int gather_around_label(vector<int> point, int width, int height){
	if (point_validation(point[0], point[1], width, height)){
		return -1;
	}
	else{
		return labels[point];
	}
}

//複数のラベル種類の存在有無の判定
bool many_kind_label(vector<int> labels){
	int kind1, kind2;
	int count = 0;
	for (auto itr = labels.begin(); itr != labels.end(); ++itr){
		if (count == 0){
			kind1 = *itr;
		}
		kind2 = *itr;
		count++;
	}
	if (kind1 != kind2){
		return true;
	}
	else{
		return false;
	}
}

//画素に新たなラベルを割り当てる
void assign_label(int x, int y, int width, int height ,int* latest_label_num){
	int l; //ラベルの一時代入用
	/********変数宣言*********************************************
	* point: 注目点                                              *
	* leftup: 左上の座標                                         *
	* up: 真上の座標                                             *
	* rightup: 右上の座標                                        *
	* left: 真左の座標                                           *
	* valid_labels: 不正な座標(-1,1)などが含まれていないラベル群 *
	**************************************************************/
	vector<int> point{ x, y }, leftup{ x - 1, y - 1 }, up{ x, y - 1 },
		rightup{ x + 1, y - 1 }, left{ x - 1, y }, valid_labels;

	//valid_labelsに不正な座標以外を代入
	l = gather_around_label(leftup, width, height);
	if (l != -1){
		valid_labels.push_back(l);
	}
	l = gather_around_label(up, width, height);
	if (l != -1){
		valid_labels.push_back(l);
	}
	l = gather_around_label(rightup, width, height);
	if (l != -1){
		valid_labels.push_back(l);
	}
	l = gather_around_label(left, width, height);
	if (l != -1){
		valid_labels.push_back(l);
	}

	//valid_labelsのゼロのカウントとラベルの種類、最小値を計算
	int zero_count = 0;
	vector<int> labels_except_zero;
	int min_label_num = 1000;
	for (auto itr = valid_labels.begin(); itr != valid_labels.end(); ++itr){
		if (*itr == 0){
			zero_count++;
		}
		else{
			labels_except_zero.push_back(*itr);  //0以外のラベルを格納
			//ラベルの最小値計算
			if (*itr < min_label_num){
				min_label_num = *itr;
			}
		}
	}
	//ラベル割り当て
	if (zero_count == valid_labels.size()){
		*latest_label_num += 1;
		labels[point] = *latest_label_num;
	}
	else{
		labels[point] = min_label_num;
		if (many_kind_label(labels_except_zero)){
			for (int i = 0; i < labels_except_zero.size(); i++){
				if (labels_except_zero[i] != min_label_num){
					lookup_table[labels_except_zero[i]] = min_label_num;
				}
			}
		}
	}
}

//lookupテーブルからラベル番号を参照するときに用いる(ネストしているラベルに対応するため)
int reference_label(int input_label){
	int dst_label = lookup_table[input_label];
	if (input_label == dst_label){
		auto itr = label_list.begin();
		itr = find(itr, label_list.end(), dst_label);
		if (itr == label_list.end()){
			label_list.push_back(dst_label);
			label_color_list[dst_label] = get_label_color();
			/*	cout << dst_label << endl;
			cout << label_color_list[dst_label] << endl;*/
		}
	}
	else{
		dst_label = reference_label(dst_label);
	}
	return dst_label;
}

//ラベル探索の際に使用
unordered_map<int, int> index_of_labels;

//int data_size_per_cls[LABEL_KIND_NUM] = {};
//ラベリング本体
void labeling(Mat& frame, int height_min, int height_max, int width_min, int width_max){
	const int mask = 9;
	Mat gray_img, thre_img;
	cvtColor(frame, gray_img, CV_RGB2GRAY);
	threshold(gray_img, thre_img, 0, 255, THRESH_BINARY | THRESH_OTSU);
	imwrite("thre_image.png", thre_img);
	const int width = thre_img.cols;
	const int height = thre_img.rows;
	int latest_label_num = 0;                                    //もっとも新しいラベル

	//ラベリングのためのルックアップテーブルを用意
	for (int i = 0; i < LOOKUP_SIZE; i++){
		lookup_table[i] = i;
	}

	//全画素ラベル初期化
	for (int y = height_min; y <= height_max; y++){
		unsigned char* ptr = thre_img.ptr<unsigned char>(y);
		for (int x = width_min; x < width_max; x++){
			int pt = ptr[x];
			vector<int> v{ x, y };
			labels[v] = 0;
		}
	}

	//ラベリング実行
	for (int y = height_min; y < height_max; y++){
		unsigned char* ptr = thre_img.ptr<unsigned char>(y);
		for (int x = width_min; x < width_max; x++){
			int p = ptr[x];
			if (p == 255){
				assign_label(x, y, frame.cols, frame.rows, &latest_label_num);
			}
		}
	}

	//ルックアップテーブルを用いてラベルの書き換え
	for (auto itr = labels.begin(); itr != labels.end(); ++itr){
		int fixed_label = reference_label(itr->second);
		labels[itr->first] = fixed_label;
	}

	int index = 0;
	for (auto itr = label_list.begin(); itr != label_list.end(); ++itr){
		index_of_labels[*itr] = index;
		index++;
	}

	/********雑音除去のコード(時間あるとき続き実装)********/
	/*
	unordered_map<int, int> data_size_cls;
	for (auto itr = label_list.begin(); itr != label_list.end(); ++itr){
		int label = *itr;
		data_size_cls[label] = 0;
	}
	vector<int> point;
	int label, size;
	//面積が明らかに少ないラベル(雑音)の除去
	for (auto itr = labels.begin(); itr != labels.end(); ++itr){
		label = itr->second;
		if (label == 23){ cout << "true" << endl; }
		data_size_cls[label]++;
	}
	vector<int> noise_labels;
	for (auto itr = data_size_cls.begin(); itr != data_size_cls.end(); ++itr){
		size = itr->second;
		if (size < label_size_thresh){
			noise_labels.push_back(itr->first);
		}
	}
	if (noise_labels.size() != 0){
		for (auto itr = labels.begin(); itr != labels.end(); ++itr){
			label = itr->second;
			auto itr2 = find(noise_labels.begin(), noise_labels.end(), label);
			if (itr2 != noise_labels.end()){
				//見つけたとき。ノイズだったとき。
				labels[itr->first] = 0;
			}
		}
	}*/
	/******************************************************/

}

//ラベリング結果をテキストファイルに書き出す
void output_labels(int width, int height){
	try{
		if (labels.empty()){ throw "Exception: labelsが空です"; }
	}
	catch (char* e){
		cout << e;
	}
	ofstream out_labels(output_labels_filename[ID]);
	for (int y = 0; y < height; y++){
		for (int x = 0; x < width; x++){
			vector<int> point{ x, y };
			int label = labels[point];
			if (label != 0){
				out_labels << x << "," << y << "," << label << endl;
			}
		}
	}
	out_labels.close();
}

//ラベリング結果のファイルをインポートし、labelsに代入する
void import_labels(){
	ifstream input_labels_file;
	input_labels_file.open(output_labels_filename[ID]);
	if (input_labels_file.fail()){
		cout << "Exception: ファイルが見つかりません。" << endl;
		cin.get();
	}

	string str;
	int x, y, l, c;
	vector<int> p;
	while (getline(input_labels_file, str)){
		string tmp;
		istringstream stream(str);
		c = 0;
		while (getline(stream, tmp, ',')){
			if (c == 0){ x = stoi(tmp); }
			else if (c == 1){ y = stoi(tmp); }
			else{ l = stoi(tmp); }
			c++;
		}
		p = { x, y };
		labels[p] = l;
	}
}

int width_normalize(int x, int width_min, int resized_wmean ){
	return (int)(((double)x - (double)width_min) / (double)resized_wmean * 1000);
}

//逆変換
int inv_width_normalize(int normal_width, int resized_wmean, int width_min){
	return (int)(((double)resized_wmean*(double)normal_width / 1000.0) + (double)width_min);
}

int height_normalize(int y, int height_min, int resized_hmean){
	return (int)(((double)y - (double)height_min) / (double)resized_hmean * 1000);
}

//逆変換
int inv_height_normalize(int normal_height, int resized_hmean, int height_min){
	return (int)(((double)resized_hmean*(double)normal_height / 1000.0) + (double)height_min);
}

//決定する順にIDを定義
int id_orderby_dst[LABEL_KIND_NUM] = { HEAD, RIGHT_SHOULDER, NECK, LEFT_SHOULDER, RIGHT_ELBOW, LEFT_ELBOW,
RIGHT_WRIST, ANKLE, LEFT_WRIST, LEFT_KNEE, RIGHT_KNEE, LEFT_HEEL, RIGHT_HEEL };
//決定する順に名前を定義
string name_orderby_dst[LABEL_KIND_NUM] = { "頭", "右肩", "首", "左肩", "右肘", "左肘",
"右手首", "腰", "左手首", "左膝", "右膝", "左足首", "右足首" };
//決定する順に色を定義
int color_orderby_dst[LABEL_KIND_NUM] = { GREEN, YELLOW, PINK, BLUE, GREEN, YELLOW, PINK,
BLUE, GREEN, YELLOW, GREEN, YELLOW, PINK };
//indexにIDを突っ込むだけで色が返ってくる
int connection_joint_color[LABEL_KIND_NUM] = {};

void create_feature_space(int* color_feature_space, int type, int r, int g, int b){
	color_feature_space[r * 255 * 255 + g * 255 + b] = type;
}
int get_color_type(int* color_feature_space, int r, int g, int b){
	return color_feature_space[r * 255 * 255 + g * 255 + b];
}

int search_color_from_feature_space(int* color_feature_space, Point p, Vec3b color){
	const int mask = 9; //処理マスク初期値
	int r = color[2];
	int g = color[1];
	int b = color[0];

	//色特徴空間内を探索(maxk*maxk*maskの範囲)
	//	int bin[LABEL_KIND_NUM] = {};
	double min_dist = 1000000000.0;
	int min_label = 0;
	for (int tr = r - (int)(mask / 2); tr <= r + (int)(mask / 2); tr++){
		for (int tg = g - (int)(mask / 2); tg <= g + (int)(mask / 2); tg++){
			for (int tb = b - (int)(mask / 2); tb <= b + (int)(mask / 2); tb++){
				if (point_validation(tr, tg, 255, 255, 3, tb, 255)){
					continue;
				}
				else{
					if (get_color_type(color_feature_space, tr, tg, tb) != 0){
						double dist = sqrt((tr - r)*(tr - r) + (tg - g)*(tg - g) + (tb - b)*(tb - b));
						if (dist < min_dist){
							min_dist = dist;
							min_label = get_color_type(color_feature_space, tr, tg, tb);
						}
					}
				}
			}
		}
	}
	return min_label;
}

void explore_withX(Mat& frame, Label* parts[], vector<int>* sorted_labels, int* color_feature_space){
	const int phase_size = 6;
	const int phase_label[phase_size] = { 1, 3, 2, 3, 2, 2 };

	int iter_count = 0;
	for (int j = 0; j < phase_size; j++){
		vector<pair<int, int>> minX_parts_pair;
		int size = phase_label[j];
		for (int k = 0; k < size; k++){
			if (iter_count < sorted_labels->size()){
				int input_X = parts[sorted_labels->at(iter_count)]->get_minX();
				minX_parts_pair.push_back(pair<int, int>(input_X, sorted_labels->at(iter_count)));
				iter_count++;
			}else{
				break;
			}
		}
		iter_count -= size;
		sort(minX_parts_pair.begin(), minX_parts_pair.end());
		for (auto itr = minX_parts_pair.begin(); itr != minX_parts_pair.end(); ++itr){
			if (iter_count < sorted_labels->size()){
				Point sample_point = parts[sorted_labels->at(iter_count)]->get_cog().at(0);
				Vec3b sample_color = frame.at<Vec3b>(sample_point);
				int color_type = search_color_from_feature_space(color_feature_space, sample_point, sample_color);
			//	if (color_type == color_orderby_dst[iter_count]){
					parts[itr->second]->set_joint_mean(id_orderby_dst[iter_count], name_orderby_dst[iter_count]);
					parts[itr->second]->change_is_parts();
					parts[itr->second]->set_color_type(color_orderby_dst[iter_count]);
			//	}
				iter_count++;
			}
			else{
				break;
			}
		}
	}
}

void explore_withY(Label* parts[], vector<int>* sorted_labels){
	vector<pair<int, int>> minY_parts_pair;
	for (int i = 0; i < PARTS_LENGTH; i++){
		if (parts[i]->current_size() >= LABEL_SIZE_THRESH){
			int input_minY = parts[i]->get_minY();
			minY_parts_pair.push_back(pair<int, int>(input_minY, i));
		}
	}	
	sort(minY_parts_pair.begin(), minY_parts_pair.end());
	for (auto itr = minY_parts_pair.begin(); itr != minY_parts_pair.end(); ++itr){
		pair<int, int> pair = *itr;
		sorted_labels->push_back(pair.second);
	}
}

//ラベルクラスにIDを付与
void assign_joint_to_label(Mat& frame, Label* parts[], int* color_feature_space){
	vector<int> sorted_labels;
	explore_withY(parts, &sorted_labels);
	explore_withX(frame, parts, &sorted_labels, color_feature_space);
}

void check_maxY(vector<int>* labels_maxY, int label, int y){
	if (y > (*labels_maxY)[label-1]){
		(*labels_maxY)[label-1] = y;
	}
}

void check_minY(vector<int>* labels_minY, int label, int y){
	if (y < (*labels_minY)[label-1]){
		(*labels_minY)[label-1] = y;
	}
}

void check_maxX(vector<int>* labels_maxX, int label, int x){
	if (x > (*labels_maxX)[label-1]){
		(*labels_maxX)[label-1] = x;
	}
}

void check_minX(vector<int>* labels_minX, int label, int x){
	if (x < (*labels_minX)[label-1]){
		(*labels_minX)[label-1] = x;
	}
}

//Labelクラスを初期化(※リファクタリングしたいなー)
void init_label_class(Mat& frame, Label* parts[], int* color_feature_space){
	int height = frame.rows;
	int width = frame.cols;

	vector<int> labels_minY, labels_minX, labels_maxY, labels_maxX;
	vector<int>* labels_minY_ptr = &labels_minY;//labelごとの最小値と最大値を計算するために使用
	vector<int>* labels_minX_ptr = &labels_minX;
	vector<int>* labels_maxY_ptr = &labels_maxY;
	vector<int>* labels_maxX_ptr = &labels_maxX;
	vector<vector<Point>> parts_points;//ラベルごとの座標を保持するvectorを定義

	for (int i = 0; i < PARTS_LENGTH; i++){
		labels_minY_ptr->push_back(10000000000);
		labels_minX_ptr->push_back(10000000000);
		labels_maxY_ptr->push_back(0);
		labels_maxX_ptr->push_back(0);
		vector<Point> v;
		parts_points.push_back(v);
	}

	//分類
	int x, y, label;
	uchar r, g, b;
	Point p;
	Vec3b val;
	vector<int> point;
	for (auto itr = labels.begin(); itr != labels.end(); ++itr){
		point = itr->first;
		x = point[0];
		y = point[1];
		p = Point{ x, y };
		label = itr->second;
		
		//ラベルが0だったらスキップ
		if (label == 0) continue;
		
		//モデル当てはめの際に使用,各ラベルごとのx,yの最小値を計算しておく
		check_minY(labels_minY_ptr, index_of_labels[label], p.y);
		check_minX(labels_minX_ptr, index_of_labels[label], p.x);
		check_maxY(labels_maxY_ptr, index_of_labels[label], p.y);
		check_maxX(labels_maxX_ptr, index_of_labels[label], p.x);

		parts_points[index_of_labels[label]-1].push_back(p);
	}
	
    //それぞれのラベルにおける重心を求める
	vector<Point> cogs;
	int cog_x, cog_y;
	for (int i = 0; i < PARTS_LENGTH; i++){
		cog_x = ((*labels_maxX_ptr)[i] + (*labels_minX_ptr)[i]) / 2;
	    cog_y = ((*labels_maxY_ptr)[i] + (*labels_minY_ptr)[i]) / 2;
		Point cog_point{ cog_x, cog_y };
		cogs.push_back(cog_point);
	}

	//全ラベルのLabelクラスの初期化
	for (int i = 0; i < PARTS_LENGTH; i++){
		*(parts[i]) = { parts_points[i], cogs[i], (*labels_minX_ptr)[i], (*labels_minY_ptr)[i] };
		vector<Point> points = parts[i]->get_current_points();
	}

	//Labelに関節IDとnameを付与
	assign_joint_to_label(frame, parts, color_feature_space);
}

//prev_pointsとcurrent_pointsで被っている点を探索し、被っていればcurrent_pointsにセットする
void find_same_point(Label *label, Point p){
	vector<Point> prev_points = label->get_prev_points();
	auto itr = find(prev_points.begin(), prev_points.end(), p);
	if (itr != prev_points.end()){
		Point sp = *itr;
		label->set_current_points(sp);
	}
}

//ラベルごとにfind_same_pointを実行する
void search_same_points(Mat& frame, Label* parts[], int height_min, int height_max, int width_min, int width_max){
	for (int y = height_min; y < height_max; y++){
		Vec3b* ptr = frame.ptr<Vec3b>(y);
		for (int x = width_min; x < width_max; x++){
			Vec3b color = ptr[x];
			if (color[2] > 30 && color[1] > 30 && color[0] > 30){
				Point p{ x, y };
				for (int i = 0; i < PARTS_LENGTH; i++){
					if(parts[i]->get_is_parts()) find_same_point(parts[i], p);
				}
			}
		}
	}
}

//周りの点を探索する
void search_around_points_each_labels(Mat& frame, Label *label[], int* color_feature_space){
	Point cp;
	for (int i = 0; i < PARTS_LENGTH; i++){
		//partsじゃないならスキップ
		if (!label[i]->get_is_parts()){ continue; }
		
		vector<Point> current_points = label[i]->get_current_points();
		vector<Point> prev_points = label[i]->get_prev_points();
		if (current_points.size() == 0 && prev_points.size() == 0){
			cp = label[i]->get_prev_back_up();
		}
		else if (current_points.size() == 0){
			cp = prev_points[0];
		}
		else{
			cp = current_points[0];
		}
		Vec3b current;
		for (int y = cp.y - (AROUND_PIXEL_Y / 2); y < cp.y + (AROUND_PIXEL_Y / 2); y++){
			Vec3b* ptr = frame.ptr<Vec3b>(y);
			for (int x = cp.x - (AROUND_PIXEL_X / 2); x < cp.x + (AROUND_PIXEL_X / 2); x++){
				current = ptr[x];
				Point p{ x, y };
				//マーカーでない点はスキップする
				if (current[2] >= 10 && current[1] >= 10 && current[0] >= 10){
					int min_label = search_color_from_feature_space(color_feature_space, p, current);
					if (min_label == label[i]->get_color_type()){
						label[i]->set_current_points(p);
					}
				}
			}
		}
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
		int width = frame.cols;
		int height = frame.rows;
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
vector<XYRGB> data;
//画像の前処理（ノイズ除去,リサイズ,など）
void resize_and_preproc(Mat& src, int* height_min_ptr, int* height_max_ptr, int* width_min_ptr, int* width_max_ptr,
	int* resized_width_ptr, int* resized_height_ptr, int* resized_wmean_ptr, int* resized_hmean_ptr, bool first=false){
	/*************ノイズ除去*************/
	const int mask = 9;
	Mat filtered_img;
	medianBlur(src, filtered_img, mask);
	/***********画像のリサイズ************/
	int y, x;
	//const int extra_y_size = 20;
    int height_min = 1000000000;
	int height_max = 0;
	int width_min = 1000000000;
	int width_max = 0;
	for (y = 0; y < src.rows; y++){
		Vec3b* ptr = filtered_img.ptr<Vec3b>(y);
		for (x = 0; x < src.cols; x++){
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

	*height_min_ptr = height_min;
	*height_max_ptr = height_max;
	*width_min_ptr = width_min;
	*width_max_ptr = width_max;
	*resized_width_ptr = width_max - width_min;
	*resized_height_ptr = height_max - height_min;
	*resized_wmean_ptr = (width_max + width_min) / 2;
	*resized_hmean_ptr = (height_max + height_min) / 2;

	Mat resized_img(src, Rect(width_min, height_min, *resized_width_ptr, *resized_height_ptr));
	/***************クラスタリング用のデータ構築****************/
	if (first){
		XYRGB p;
		for (y = height_min; y <= height_max; y++){
			Vec3b* ptr = src.ptr<Vec3b>(y);
			for (x = width_min; x < width_max; x++){
				Vec3b c = ptr[x];
				if (c[2] > 30 && c[1] > 30 && c[0] > 30){
					//change_label_feature_space(y, c[2], c[1], c[0], true);
					//cout << height_normalize(y) << endl;
					p = { width_normalize(x, width_min, *resized_wmean_ptr), height_normalize(y, height_min, *resized_hmean_ptr) , c[2], c[1], c[0] };
					data.push_back(p);
				}
			}
		}
	}
	try{
		imwrite("resized_img.png", resized_img);
	}
	catch (runtime_error& ex){
		printf("error");
	}
}

//重複している点がないかをチェック
bool check_distinct_points(XYRGB *kCenter, XYRGB data, int count){
	for (int i = 0; i < count; i++){
		XYRGB center = kCenter[i];
		if (center.x == data.x, center.y == data.y && center.r == data.r && center.g == data.g && center.b == data.b){
			return false;
		}
		else{
			return true;
		}
	}
}

string output_color_name[COLOR_KIND] = { "green_data.dat", "yellow_data.dat", "pink_data.dat", "blue_data.dat" };
void output_sample_color(Mat frame, Label* parts[]){
	const int parts_index = 10;
	const int target = BLUE;
	Label sample_parts = *(parts[parts_index]);
	vector<Point> points = sample_parts.get_current_points();
	ofstream out_labels(output_color_name[target]);
	for (auto itr = points.begin(); itr != points.end(); ++itr){
		Point p = *itr;
		Vec3b c = frame.at<Vec3b>(p);
		int r = c[2];
		int g = c[1];
		int b = c[0];
		out_labels << r << " " << g << " " << b << endl;
	}
	out_labels.close();
}

void import_color_data(int* color_feature_space){
	for (int i = 0; i < COLOR_KIND; i++){
		ifstream input_color_file;
		input_color_file.open(output_color_name[i]);
		if (input_color_file.fail()){
			cout << "Exception: ファイルが見つかりません。" << endl;
			cin.get();
		}
		string str;
		int r, g, b, l, c;
		vector<int> p;
		while (getline(input_color_file, str)){
			string tmp;
			istringstream stream(str);
			c = 0;
			while (getline(stream, tmp, ',')){
				if (c == 0){ r = stoi(tmp); }
				else if (c == 1){ g = stoi(tmp); }
				else{ b = stoi(tmp); }
				c++;
			}
			create_feature_space(color_feature_space, i+1, r, g, b);
		}
		input_color_file.close();
	}
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

	//色教師データの構築
	int* color_feature_space = NULL;
	color_feature_space = (int *)calloc(256 * 256 * 256, sizeof(int));
	import_color_data(color_feature_space);
	
	Label parts1, parts2, parts3, parts4, parts5, parts6, parts7, parts8, parts9, parts10,
		parts11, parts12, parts13, parts14, parts15, parts16, parts17, parts18, parts19, parts20,
		parts21, parts22, parts23;
	Label* parts[PARTS_LENGTH] = { &parts1, &parts2, &parts3, &parts4, &parts5, &parts6,
		&parts7, &parts8, &parts9, &parts10, &parts11, &parts12, &parts13, &parts14, &parts15,
		&parts16, &parts17, &parts18, &parts19, &parts20, &parts21, &parts22, &parts23 };

	ofstream ofs("output_angles.txt");
	namedWindow("test");
	
	Mat dst_img, resized_img;
	int count = 0;
	while (1){
		count++;
		Mat& frame = dst_img;
		video >> frame;
		if (frame.empty() || video.get(CV_CAP_PROP_POS_AVI_RATIO) == 1){
			break;
		}
		int width = frame.cols;
		int height = frame.rows;
		int height_min, height_max, width_min, width_max; //人間の領域のx,yの最小値と最大値
		int	resized_width, resized_height, resized_hmean, resized_wmean; //トリミングした画像の幅、高さ、平均値
		//対象のフレームまではスキップ
		if (count < use_start_frame){
			continue;
		}
		else if (count == use_start_frame){
			resize_and_preproc(frame, &height_min, &height_max, &width_min, &width_max, &resized_width, &resized_height, &resized_hmean, &resized_wmean);
			labeling(frame, height_min, height_max, width_min, width_max);
		/*	for (int y = height_min; y < height_max; y++){
				Vec3b* ptr = frame.ptr<Vec3b>(y);
				for (int x = width_min; x < width_max; x++){
					vector<int> point{ x, y };
					int label = labels[point];
					if (label != 0){
						ptr[x] = label_color_list[label];
					}
				}
			}*/
			init_label_class(frame, parts, color_feature_space);
			Scalar test_colors[LABEL_KIND_NUM] = { { 0, 0, 255 }, { 0, 255, 0 }, { 255, 0, 0 }, { 0, 255, 255 },
			{ 255, 255, 0 }, { 255, 0, 255 }, { 0, 0, 125 }, { 0, 125, 0 },
			{ 125, 0, 0 }, { 0, 125, 125 }, { 125, 0, 125 }, { 125, 125, 0 } };
			
		/*	for (int m = 0; m < PARTS_LENGTH; m++){
				if (parts[m]->get_is_parts()){
					vector<Point> test_point = parts[m]->get_current_points();
					vector<Point> test_point = parts[m]->get_cog();
					test_points.push_back(test_point);
				}
			}*/

			//特定のPartsだけみたいとき
			//vector<Point> test_point = parts[0]->get_current_points();
			//test_points.push_back(test_point);
	
			
			//色のサンプルを出力したいとき
			//output_sample_color(frame, parts);
			/*
			int n = 0;
			for (auto itr = test_points.begin(); itr != test_points.end(); ++itr){
				vector<Point> test_point = *itr;
				for (auto itr2 = test_point.begin(); itr2 != test_point.end(); ++itr2){
					Point p = *itr2;
					rectangle(dst_img, p, p, test_colors[n]);
				}
				n++;
			}*/
		//	output_labels(width, height);
		/*	for (int y = 0; y < height; y++){
				Vec3b* ptr = frame.ptr<Vec3b>(y);
				for (int x = 0; x < width; x++){
					vector<int> point{ x, y };
					int label = labels[point];
					if (label != 0){
						ptr[x] = label_color_list[label];
					}
				}
			}
			break;*/
		}
		else if(count >= use_end_frame){
			//対象となるフレームが終わったらループを抜ける
			break;
		}
		else{
			if (MODE == 1){
				resize_and_preproc(frame, &height_min, &height_max, &width_min, &width_max, &resized_width, &resized_height, &resized_hmean, &resized_wmean);
				change_prev_and_current(parts);
				search_same_points(frame, parts, height_min, height_max, width_min, width_max);
				search_around_points_each_labels(frame, parts, color_feature_space);
				set_cog_each_label(parts);
			}
			else{
				break;
			}
		}
		vector<vector<Point>> test_points;

		for (int m = 0; m < PARTS_LENGTH; m++){
			if (parts[m]->get_id() == HEAD){
				//vector<Point> test_point = parts[m]->get_current_points();
				Point test_point = parts[m]->get_cog()[count - use_start_frame];
				circle(dst_img, test_point, 10, Scalar(0, 0, 255));
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