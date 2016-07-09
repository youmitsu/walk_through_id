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
#define LOOKUP_SIZE 100                                  //ルックアップテーブルのデフォルトサイズ
#define LABEL_KIND_NUM 5                                 //取得したいラベルの種類数
#define AROUND_PIXEL_X 500                               //現在の座標の周りの探索する際のXの範囲
#define AROUND_PIXEL_Y 80                                //                              Yの範囲
#define ID_COUNT 4                                       //データとなる動画の数
#define COLOR_DECIDE_LENGTH 6                            //色空間を定義するのに必要な要素数 ex){rs, re, gs, ge, bs, be}の配列
#define MODE_KIND 3

/*******「誰の」「何の処理か」を設定********/
#define ID 0                                             //0:星野, 1:秀野, 2:羽田, 3:北沢
#define MODE 0                                           //0:ラベリングテストモード 1:追跡モード 2:再生モード

using namespace std;
using namespace cv;

/*************定数群(なんか怖いから配列系はconstにした)****************/
const string video_urls[ID_COUNT] = { "Hoshino.avi", "Shuno.avi", "Haneda.avi", "Kitazawa.avi" };
const int use_start_frames[ID_COUNT] = { 400, 207, 529, 1832 };
const int use_frame_nums[ID_COUNT] = { 32, 32, 32, 38 };
//ラベルごとの色空間を定義
const unsigned int ankle_color_spaces[ID_COUNT][COLOR_DECIDE_LENGTH] = { { 0, 50, 50, 255, 150, 255 },
{ 0, 50, 50, 255, 150, 255 },
{ 0, 50, 50, 255, 150, 255 },
{ 0, 50, 50, 255, 150, 255 } };      //青
const unsigned int left_knee_color_spaces[ID_COUNT][COLOR_DECIDE_LENGTH] = { { 0, 80, 150, 255, 0, 80 },
{ 0, 80, 150, 255, 0, 150 },
{ 0, 80, 150, 255, 0, 150 },
{ 0, 80, 150, 255, 0, 150 } };      //緑
const unsigned int right_knee_color_spaces[ID_COUNT][COLOR_DECIDE_LENGTH] = { { 180, 255, 170, 255, 0, 150 },
{ 180, 255, 170, 255, 0, 150 },
{ 180, 255, 170, 255, 0, 150 },
{ 180, 255, 170, 255, 0, 150 } };     //黄色
const unsigned int left_heel_color_spaces[ID_COUNT][COLOR_DECIDE_LENGTH] = { { 180, 255, 170, 255, 0, 150 },
{ 180, 255, 170, 255, 0, 150 },
{ 180, 255, 170, 255, 0, 150 },
{ 180, 255, 170, 255, 0, 150 } };      //黄色
const unsigned int right_heel_color_spaces[ID_COUNT][COLOR_DECIDE_LENGTH] = { { 100, 255, 0, 100, 100, 255 },
{ 100, 255, 0, 100, 100, 255 },
{ 100, 255, 0, 100, 100, 255 },
{ 100, 255, 0, 100, 100, 255 } };     //紫

const int labels_each_ids[ID_COUNT][LABEL_KIND_NUM] = { { 15, 25, 31, 38, 41 },
{ 21, 34, 35, 44, 45 },
{ 24, 33, 37, 41, 47 },
{ 30, 50, 48, 57, 59 } };

/************グローバル変数群***********/
string video_url;                                            //使用する動画のURL
int use_start_frame;                                         //動画から使う最初のフレーム
int use_frame_num;                                           //使用するフレーム数
int use_end_frame;                                           //動画から使う最後のフレーム
vector<unsigned int> ankle_color_space;         //腰に該当する色空間
vector<unsigned int> left_knee_color_space;     //左膝
vector<unsigned int> right_knee_color_space;    //右膝
vector<unsigned int> left_heel_color_space;     //左足首
vector<unsigned int> right_heel_color_space;    //右足首
int label_num_by_id[LABEL_KIND_NUM];                         //取得したい関節に該当するラベル番号を格納
unordered_map<int, int> lookup_table;                        //ルックアップテーブル
int latest_label_num = 0;                                    //ラベリングで使用する
int width;                                                   //画像の幅
int height;                                                  //高さ
vector<double> angles;                                       //フレームごとの関節の角度
const string output_labels_filename[ID_COUNT] = { "output_labels_hoshino.txt",  "output_labels_shuno.txt",
"output_labels_haneda.txt", "output_labels_kitazawa.txt"};

//グローバル変数の初期化
void init_config(){
	try{
		if (ID < 0 || ID >= ID_COUNT){ throw "Exception: IDが範囲外です。"; }
		if (MODE < 0 || MODE >= MODE_KIND){ throw "Exception: MODEが範囲外です。"; }
	}
	catch (char *e){
		cout << e;
	}
	video_url = video_urls[ID];
	use_start_frame = use_start_frames[ID];
	use_frame_num = use_frame_nums[ID];
	use_end_frame = use_start_frame + use_frame_num;
	//色空間初期化(このコードだめだねぇ)
	for (int i = 0; i < COLOR_DECIDE_LENGTH; i++){
		ankle_color_space.push_back(ankle_color_spaces[ID][i]);
		left_knee_color_space.push_back(left_knee_color_spaces[ID][i]);
		right_knee_color_space.push_back(right_knee_color_spaces[ID][i]);
		left_heel_color_space.push_back(left_heel_color_spaces[ID][i]);
		right_heel_color_space.push_back(right_heel_color_spaces[ID][i]);
	}
	for (int i = 0; i < LABEL_KIND_NUM; i++){ label_num_by_id[i] = labels_each_ids[ID][i]; }
}

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
	char name;
	vector<unsigned int> color_space;
	vector<Point> cog;
	vector<Point> prev_points;
	vector<Point> current_points;
	Point prev_back_up;
public:
	Label(){}
	Label(char name, vector<Point> current_points, Point first_cog, vector<unsigned int> color_space)
		: name(name), current_points(current_points), color_space(color_space)
	{
		vector<Point> pp;
		prev_points = pp;
		cog.push_back(first_cog);
	}
	char get_name() { return name; }
	vector<unsigned int> get_color_space(){ return color_space; }
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
	if (!prev_points.size() == 0){
		prev_back_up = prev_points[0];
	}
}

void Label::clear_prev_points(){
	set_prev_back_up();
	prev_points.clear();
}

unordered_map<vector<int>, int, HashVI> labels;  //key：座標、value：ラベル番号
vector<int> label_list;                          //全ラベルの一覧
unordered_map<int, Vec3b> label_color_list;      //ラベルごとの色

//座標の正当性チェック
bool point_validation(int x, int y, int width, int height){
	if (x < 0 || x > width || y < 0 || y > height) {
		return true;
	}
	else{
		return false;
	}
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
void assign_label(int x, int y, int width, int height){
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
		latest_label_num += 1;
		labels[point] = latest_label_num;
	}
	else{
		labels[point] = min_label_num;
		if (many_kind_label(labels_except_zero)){
			for (int i = 0; i < labels_except_zero.size(); i++){
				if (labels_except_zero[i] != min_label_num){
		//			cout << labels_except_zero[i] << endl;
					lookup_table[labels_except_zero[i]] = min_label_num;
				}
			}
		}
    }
}

//ランダムなRGB値を返す
Vec3b get_label_color(){
	const int MAX_VALUE = 255;
	unsigned int r = rand() % MAX_VALUE;
	unsigned int g = rand() % MAX_VALUE;
	unsigned int b = rand() % MAX_VALUE;
	return Vec3b(r, g, b);
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

//ラベリング本体
void labeling(Mat& frame){
	Mat gray_img, thre_img;
	cvtColor(frame, gray_img, CV_RGB2GRAY);
	threshold(gray_img, thre_img, 0, 255, THRESH_BINARY | THRESH_OTSU);
	const int width = thre_img.cols;
	const int height = thre_img.rows;

	//ラベリングのためのルックアップテーブルを用意
	for (int i = 0; i < LOOKUP_SIZE; i++){
		lookup_table[i] = i;
	}

	//全画素ラベル初期化
	for (int y = 0; y < height; y++){
		unsigned char* ptr = thre_img.ptr<unsigned char>(y);
		for (int x = 0; x < width; x++){
			int pt = ptr[x];
			vector<int> v{ x, y };
			labels[v] = 0;
		}
	}
	
	//ラベリング実行
	for (int y = 0; y < height; y++){
		unsigned char* ptr = thre_img.ptr<unsigned char>(y);
		for (int x = 0; x < width; x++){
			if (ptr[x] > 200){
				assign_label(x, y, width, height);
			}
		}
	}

	//ルックアップテーブルを用いてラベルの書き換え
	for (int y = 0; y < height; y++){
		unsigned char* ptr = thre_img.ptr<unsigned char>(y);
		for (int x = 0; x < width; x++){
			vector<int> v{ x, y };
			labels[v] = reference_label(labels[v]);
		}
	}
}

//ラベリング結果をテキストファイルに書き出す
void output_labels(){
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
			out_labels << label << endl;
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
	while (getline(input_labels_file, str)){
		
	}
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
//Labelクラスを初期化
void init_label_class(Mat& frame, Label *ankle_ptr, Label *left_knee_ptr,
	Label *right_knee_ptr, Label *left_heel_ptr, Label *right_heel_ptr){
	int height = frame.rows;
	int width = frame.cols;
	
	//ラベルごとの最大値の{ankle[x],ankle[y],left_knee[x],left_knee[y],x,y,...,x,y}
	int max_points[10] = {}; 
	//ラベルごとの最小値の{ankle[x],ankle[y],left_knee[x],left_knee[y],x,y,...,x,y}
	int min_points[10];
	for (int i = 0; i < 10; i++){ min_points[i] = 100000000; }

	vector<Point> ankle_point;
	vector<Point> left_knee_point;
	vector<Point> right_knee_point;
	vector<Point> left_heel_point;
	vector<Point> right_heel_point;
	for (int y = 0; y < height; y++){
		Vec3b* ptr = frame.ptr<Vec3b>(y);
		for (int x = 0; x < width; x++){
			vector<int> v{ x, y };
			if (labels[v] == label_num_by_id[0]){
				ankle_point.push_back(Point{ x, y });
				change_min_and_max_value(x, y, &max_points[0], &max_points[1],
					&min_points[0], &min_points[1]);
			}
			else if (labels[v] == label_num_by_id[1]){
				left_knee_point.push_back(Point{ x, y });
				change_min_and_max_value(x, y, &max_points[2], &max_points[3],
					&min_points[2], &min_points[3]);
			}
			else if (labels[v] == label_num_by_id[2]){
				right_knee_point.push_back(Point{ x, y });
				change_min_and_max_value(x, y, &max_points[4], &max_points[5],
					&min_points[4], &min_points[5]);
			}
			else if (labels[v] == label_num_by_id[3]){
				left_heel_point.push_back(Point{ x, y });
				change_min_and_max_value(x, y, &max_points[6], &max_points[7],
					&min_points[6], &min_points[7]);
			}
			else if (labels[v] == label_num_by_id[4]){
				right_heel_point.push_back(Point{ x, y });
				change_min_and_max_value(x, y, &max_points[8], &max_points[9],
					&min_points[8], &min_points[9]);
			}
		}
	}
	Point cogs[5];
	for (int i = 0; i < LABEL_KIND_NUM; i++){
		int x = (max_points[i*2] + min_points[i*2]) / 2;
	    int y = (max_points[i*2+1] + min_points[i*2+1]) / 2;
		Point cog_point{ x, y };
		cogs[i] = cog_point;
	}

	Label ankle('腰', ankle_point, cogs[0], ankle_color_space);
	Label left_knee('左膝', left_knee_point, cogs[1], left_knee_color_space);
	Label right_knee('右膝', right_knee_point, cogs[2], right_knee_color_space);
	Label left_heel('左首', left_heel_point, cogs[3], left_heel_color_space);
	Label right_heel('右首', right_heel_point, cogs[4], right_heel_color_space);

	*ankle_ptr = ankle;
	*left_knee_ptr = left_knee;
	*right_knee_ptr = right_knee;
	*left_heel_ptr = left_heel;
	*right_heel_ptr = right_heel;
}

//全ラベルのprev_pointsとcurrent_pointsを入れ替え,current_pointsをクリアする
void change_prev_and_current(Label *ankle, Label *left_knee, Label *right_knee,
	Label *left_heel, Label *right_heel){
	ankle->clear_prev_points();
	ankle->change_ptr();
	left_knee->clear_prev_points();
	left_knee->change_ptr();
	right_knee->clear_prev_points();
	right_knee->change_ptr();
	left_heel->clear_prev_points();
	left_heel->change_ptr();
	right_heel->clear_prev_points();
	right_heel->change_ptr();
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
void search_same_points(Mat& frame, Label *ankle, Label *left_knee,
	Label *right_knee, Label *left_heel, Label *right_heel){
	for (int y = 0; y < height; y++){
		unsigned char* ptr = frame.ptr<unsigned char>(y);
		for (int x = 0; x < width; x++){
			if (ptr[x] != 0){
				Point p{ x, y };
				find_same_point(ankle, p);
				find_same_point(left_knee, p);
				find_same_point(right_knee, p);
				find_same_point(left_heel, p);
				find_same_point(right_heel, p);
			}
		}
	}
}

//周りの点を探索する
void search_around_points_each_labels(Mat& frame, Label *label){
	Point cp;
	vector<Point> current_points = label->get_current_points();
	vector<Point> prev_points = label->get_prev_points();
	if (current_points.size() == 0 && prev_points.size() == 0){
		cp = label->get_prev_back_up();
	}
	else if (current_points.size() == 0){
		cp = prev_points[0];
	}
	else{
		cp = current_points[0];
	}
	Vec3b current_color;
	vector<unsigned int> cs = label->get_color_space();
	unsigned int rs = cs[0];
	unsigned int re = cs[1];
	unsigned int gs = cs[2];
	unsigned int ge = cs[3];
	unsigned int bs = cs[4];
	unsigned int be = cs[5];
	for (int y = cp.y - (AROUND_PIXEL_Y / 2); y < cp.y + (AROUND_PIXEL_Y / 2); y++){
		Vec3b* ptr = frame.ptr<Vec3b>(y);
		for (int x = cp.x - (AROUND_PIXEL_X / 2); x < cp.x + (AROUND_PIXEL_X / 2); x++){
			current_color = ptr[x];
			if (rs < current_color[2] && re > current_color[2] &&
				gs < current_color[1] && ge > current_color[1] &&
				bs < current_color[0] && be > current_color[0]){
				label->set_current_points(Point{ x, y });
			}
		}
	}
}

//ラベルごとに周りの点を探索する
void search_around_points(Mat& frame, Label *ankle, Label *left_knee,
    Label *right_knee, Label *left_heel, Label *right_heel){
	search_around_points_each_labels(frame, ankle);
	search_around_points_each_labels(frame, left_knee);
	search_around_points_each_labels(frame, right_knee);
	search_around_points_each_labels(frame, left_heel);
	search_around_points_each_labels(frame, right_heel);
}

//ラベルごとに重心をセットする
void set_cog_each_label(Label *ankle, Label *left_knee, Label *right_knee,
	Label *left_heel, Label *right_heel){
	ankle->calc_and_set_cog();
	left_knee->calc_and_set_cog();
	right_knee->calc_and_set_cog();
	left_heel->calc_and_set_cog();
	right_heel->calc_and_set_cog();
}


//腰と右膝、左膝から成す角度を求め、anglesにpushする
void set_angle_ankle_and_knees(Point ankle, Point right_knee, Point left_knee){
	int ankle_x = ankle.x;
	int ankle_y = ankle.y;
	int right_knee_x = right_knee.x;
	int right_knee_y = right_knee.y;
	int left_knee_x = left_knee.x;
	int left_knee_y = left_knee.y;
	int a1 = right_knee_x - ankle_x;
	int a2 = right_knee_y - ankle_y;
	int b1 = left_knee_x - ankle_x;
	int b2 = left_knee_y - ankle_y;
	float cos = ((a1*b1) + (a2*b2)) / ((sqrt((a1*a1) + (a2*a2))*sqrt((b1*b1) + (b2*b2))));
	float angle = acosf(cos);
	angles.push_back(angle);
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
	//腰、左膝、右膝、左足首、右足首のLabelインスタンスを宣言
	Label ankle;
	Label left_knee;
	Label right_knee;
	Label left_heel;
	Label right_heel;

	ofstream ofs("output_angles.txt");

	namedWindow("test");
	
	Mat dst_img;
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
			//最初のフレームでラベリングとLabelクラスを初期化
			if (MODE == 0){
				labeling(frame);
				output_labels();
				for (int i = 0; i < label_list.size(); i++){
					int label = label_list[i];
					Vec3b label_color = label_color_list[label];
					cout << label << ":" << label_color << endl;
				}
		/*		for (int y = 0; y < height; y++){
					Vec3b* ptr = frame.ptr<Vec3b>(y);
					for (int x = 0; x < width; x++){
						vector<int> point{ x, y };
						int label = labels[point];
						if (label != 0){
							ptr[x] = label_color_list[label];
						}
					}
				}*/
			}
			if (MODE == 1){
				import_labels();
				init_label_class(frame, &ankle, &left_knee, &right_knee,
					&left_heel, &right_heel);
				set_angle_ankle_and_knees(ankle.get_cog()[count - use_start_frame],
					right_knee.get_cog()[count - use_start_frame], left_knee.get_cog()[count - use_start_frame]);
			}
		}
		else if(count >= use_end_frame){
			//対象となるフレームが終わったらループを抜ける
			break;
		}
		else{
			if (MODE == 1){
				change_prev_and_current(&ankle, &left_knee, &right_knee, &left_heel, &right_heel);

				search_same_points(frame, &ankle, &left_knee, &right_knee, &left_heel, &right_heel);

				search_around_points(frame, &ankle, &left_knee, &right_knee, &left_heel, &right_heel);

				set_cog_each_label(&ankle, &left_knee, &right_knee, &left_heel, &right_heel);

				set_angle_ankle_and_knees(ankle.get_cog()[count - use_start_frame],
					right_knee.get_cog()[count - use_start_frame], left_knee.get_cog()[count - use_start_frame]);
			}
			else{
				break;
			}
		}
		
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
        
		if (MODE == 1){
			circle(dst_img, ankle.get_cog()[count - use_start_frame], 5, Scalar(0, 0, 255), -1);
			circle(dst_img, left_knee.get_cog()[count - use_start_frame], 5, Scalar(0, 255, 0), -1);
			circle(dst_img, right_knee.get_cog()[count - use_start_frame], 5, Scalar(255, 0, 0), -1);
			circle(dst_img, left_heel.get_cog()[count - use_start_frame], 5, Scalar(0, 255, 255), -1);
			circle(dst_img, right_heel.get_cog()[count - use_start_frame], 5, Scalar(255, 0, 255), -1);
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
	
	if (MODE == 1){
		for (int i = 0; i < use_frame_num; i++){
			cout << i << "フレーム目:" << angles[i] << endl;
			ofs << i << ", " << angles[i] << endl;
			circle(dst_img, ankle.get_cog()[i], 1, Scalar(0, 0, 255), -1);
			circle(dst_img, left_knee.get_cog()[i], 1, Scalar(0, 255, 100), -1);
			circle(dst_img, right_knee.get_cog()[i], 1, Scalar(255, 0, 0), -1);
			circle(dst_img, left_heel.get_cog()[i], 1, Scalar(255, 217, 0), -1);
			circle(dst_img, right_heel.get_cog()[i], 1, Scalar(255, 0, 255), -1);
		}
	}
	
	try{
		imwrite("出力結果.png", dst_img);
	}
	catch (runtime_error& ex){
		printf("failure");
		return 1;
	}
	/**********フーリエ変換とプロット***********/
	if (MODE == 1){
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
	waitKey(0);
	return 0;
}