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

/*******「誰の」「何の処理か」「特徴量」を設定********/
#define ID 3                                             //0:星野, 1:秀野, 2:羽田, 3:北沢
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

string joint_names[LABEL_KIND_NUM] = {"頭", "左", "a", "b", "c", "d", "e", "f", "g", "h", "k", "l"};

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
public:
	Label(){}
	Label(vector<Point> current_points, Point first_cog, int minX, int minY)
		: current_points(current_points), minX(minX), minY(minY)
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
	int get_minY(){ return minY; };
	int get_minX(){ return minX; };
	void set_prev_back_up();
	void set_current_points(Point p);
	void set_cog(Point p);
	void calc_and_set_cog();
	void change_ptr();
	void clear_prev_points();
	void set_joint_mean(int id, string name);
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
	const int label_size_thresh = 30;
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
	int label;
	//面積が明らかに少ないラベル(雑音)の除去
	for (auto itr = labels.begin(); itr != labels.end(); ++itr){
		label = itr->second;
		data_size_cls[label]++;
	}
	for (auto itr = data_size_cls.begin(); itr != data_size_cls.end(); ++itr){
		int size = itr->second;
		if (size < label_size_thresh){
			//閾値を満たさないラベルがあればlabelsから削除(めんどい)
		}
	}
	*/
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

void explore_withX(Label* parts, int* sorted_labels){
	const int phase_size = 6;
	const int phase_label[phase_size] = { 1, 3, 2, 3, 2, 2 };

	int iter_count = 0;
	while (iter_count < LABEL_KIND_NUM){
		for (int j = 0; j < phase_size; j++){
			vector<pair<int, int>> minX_parts_pair;
			vector<pair<int, int>>::iterator it;
			int size = phase_label[j];
			for (int k = 0; k < size; k++){
				int input_X = parts[sorted_labels[iter_count]].get_minX();
				minX_parts_pair.push_back(pair<int, int>(input_X, sorted_labels[iter_count]));
				iter_count++;
			}
			iter_count -= size;
			sort(minX_parts_pair.begin(), minX_parts_pair.end());
			for (auto itr = minX_parts_pair.begin(); itr != minX_parts_pair.end(); ++itr){
				parts[itr->second].set_joint_mean(id_orderby_dst[iter_count], name_orderby_dst[iter_count]);
				iter_count++;
			}
		}
	}
}

void explore_withY(Label* parts, int* sorted_labels){
	int min = 1000000000;
	vector<pair<int, int>> minY_parts_pair;
	vector<pair<int, int>>::iterator it;
	for (int i = 0; i < LABEL_KIND_NUM; i++){
		int input_minY = parts[i].get_minY();
		minY_parts_pair.push_back(pair<int, int> (input_minY, i));
	}	
	sort(minY_parts_pair.begin(), minY_parts_pair.end());
	int index = 0;
	for (auto itr = minY_parts_pair.begin(); itr != minY_parts_pair.end(); ++itr){
		pair<int, int> pair = *itr;
		sorted_labels[index] = pair.second;
		index++;
	}
}

//ラベルクラスにIDを付与
void assign_joint_to_label(Label* parts){
	int sorted_labels[LABEL_KIND_NUM];
	explore_withY(parts, sorted_labels);
	explore_withX(parts, sorted_labels);
}

void check_maxY(vector<int>* labels_maxY, int label, int y){
	if (y > (*labels_maxY)[label]){
		(*labels_maxY)[label] = y;
	}
}

void check_minY(vector<int>* labels_minY, int label, int y){
	if (y < (*labels_minY)[label]){
		(*labels_minY)[label] = y;
	}
}

void check_maxX(vector<int>* labels_maxX, int label, int x){
	if (x > (*labels_maxX)[label]){
		(*labels_maxX)[label] = x;
	}
}

void check_minX(vector<int>* labels_minX, int label, int x){
	if (x < (*labels_minX)[label]){
		(*labels_minX)[label] = x;
	}
}

//Labelクラスを初期化(※リファクタリングしたいなー)
void init_label_class(Mat& frame, Label* parts){
	int height = frame.rows;
	int width = frame.cols;
	const int extra = 10;//とりあえずラベルの数+10個初期化
	vector<int> labels_minY, labels_minX, labels_maxY, labels_maxX;
	vector<int>* labels_minY_ptr = &labels_minY;//labelごとの最小値と最大値を計算するために使用
	vector<int>* labels_minX_ptr = &labels_minX;
	vector<int>* labels_maxY_ptr = &labels_maxY;
	vector<int>* labels_maxX_ptr = &labels_maxX;
	vector<vector<Point>> parts_points;//ラベルごとの座標を保持するvectorを定義

	for (int i = 0; i < LABEL_KIND_NUM+extra; i++){
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
	for (int i = 0; i < LABEL_KIND_NUM+extra; i++){
		cog_x = ((*labels_maxX_ptr)[i] + (*labels_minX_ptr)[i]) / 2;
	    cog_y = ((*labels_maxY_ptr)[i] + (*labels_minY_ptr)[i]) / 2;
		Point cog_point{ cog_x, cog_y };
		cogs.push_back(cog_point);
	}

	//全ラベルのLabelクラスの初期化
	for (int i = 0; i < LABEL_KIND_NUM; i++){
		parts[i] = { parts_points[i], cogs[i], (*labels_minX_ptr)[i], (*labels_minY_ptr)[i] };
	}

	assign_joint_to_label(parts);
}

//全ラベルのprev_pointsとcurrent_pointsを入れ替え,current_pointsをクリアする
void change_prev_and_current(Label* parts){
	for (int i = 0; i < LABEL_KIND_NUM; i++){
		parts[i].clear_prev_points();
		parts[i].change_ptr();
	}
}

void set_cog_each_label(Label* parts){
	for (int i = 0; i < LABEL_KIND_NUM; i++){
		parts[i].calc_and_set_cog();
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
	Label parts[LABEL_KIND_NUM] = { parts1, parts2, parts3, parts4, parts5, parts6,
		parts7, parts8, parts9, parts10, parts11, parts12 };
	//腰、左膝、右膝、左足首、右足首のLabelインスタンスを宣言

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
			init_label_class(frame, parts);
		//	output_labels(width, height);
			for (int y = 0; y < height; y++){
				Vec3b* ptr = frame.ptr<Vec3b>(y);
				for (int x = 0; x < width; x++){
					vector<int> point{ x, y };
					int label = labels[point];
					if (label != 0){
						ptr[x] = label_color_list[label];
					}
				}
			}
			break;
		}
		else if(count >= use_end_frame){
			//対象となるフレームが終わったらループを抜ける
			break;
		}
		else{
			break;
			if (MODE == 1){
				resize_and_preproc(frame, &height_min, &height_max, &width_min, &width_max, &resized_width, &resized_height, &resized_hmean, &resized_wmean);
				change_prev_and_current(parts);
//				search_points_from_image(frame, parts);
				set_cog_each_label(parts);
			}
			else{
				break;
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