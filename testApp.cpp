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
#define LABEL_KIND_NUM 6                                 //取得したいラベルの種類数
#define AROUND_PIXEL_X 200                               //現在の座標の周りの探索する際のXの範囲
#define AROUND_PIXEL_Y 50                                //                              Yの範囲
#define ID_COUNT 4                                       //データとなる動画の数
#define COLOR_DECIDE_LENGTH 9                            //色空間を定義するのに必要な要素数 ex){rs, re, gs, ge, bs, be}の配列
#define MODE_KIND 3
#define FEATURE_KIND 2

/*******「誰の」「何の処理か」「特徴量」を設定********/
#define ID 0                                             //0:星野, 1:秀野, 2:羽田, 3:北沢
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

const int labels_each_ids[ID_COUNT][LABEL_KIND_NUM] = { { 15, 25, 31, 38, 41, 2 },
{ 21, 37, 38, 49, 47, 1 },
{ 25, 39, 38, 48, 44, 1 },
{ 30, 50, 48, 57, 59, 1 } };

enum JOINT{
	ANKLE = 1,
	LEFT_KNEE = 2,
	RIGHT_KNEE = 3,
	LEFT_HEEL = 4,
	RIGHT_HEEL = 5,
	HEAD = 6,
};

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
vector<unsigned int> head_color_space;          //頭
int label_num_by_id[LABEL_KIND_NUM];                         //取得したい関節に該当するラベル番号を格納
unordered_map<int, int> lookup_table;                        //ルックアップテーブル
int latest_label_num = 0;                                    //ラベリングで使用する
int width;                                                   //画像の幅
int height;                                                  //高さ
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

unordered_map<vector<int>, int, HashVI> labels;  //key：座標、value：ラベル番号
vector<int> label_list;                          //全ラベルの一覧
unordered_map<int, Vec3b> label_color_list;      //ラベルごとの色

//座標の正当性チェック
bool point_validation(int x, int y, int width, int height, int z = NULL, int depth = NULL, int dimension = 2){
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
	else{
		try{
			throw("正しい次元を指定してください");
		}
		catch (char *e){
			cout << e;
		}
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

	for (int y = 0; y < height; y++){
		for (int x = 0; x < width; x++){
			p = { x, y };
			if (!labels[p]){
				labels[p] = 0;
			}
		}
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

/***********ヒストグラム求める***************/
int ankle_hist[256][256][256] = {};
int left_knee_hist[256][256][256] = {};
int right_knee_hist[256][256][256] = {};
int left_heel_hist[256][256][256] = {};
int right_heel_hist[256][256][256] = {};

void histgram(int part, Vec3b val){
	int r = val[2];
	int g = val[1];
	int b = val[0];
	switch (part){
	case 0:
		ankle_hist[r][g][b] += 1;
		break;
	case 1:
		left_knee_hist[r][g][b] += 1;
		break;
	case 2:
		right_knee_hist[r][g][b] += 1;
		break;
	case 3:
		left_heel_hist[r][g][b] += 1;
		break;
	case 4:
		right_heel_hist[r][g][b] += 1;
		break;
	default:
		cout << "おいおいちょっと待て" << endl;
		break;
	}
}

void output_histgram_data(){
	ofstream ankle("ankle_histgram.dat");
	ofstream left_knee("left_knee_histgram.dat");
	ofstream right_knee("right_knee_histgram.dat");
	ofstream left_heel("left_heel_histgram.dat");
	ofstream right_heel("right_heel_histgram.dat");
	for (int i = 0; i < 256; i++){
		for (int j = 0; j < 256; j++){
			for (int k = 0; k < 256; k++){
				if (ankle_hist[i][j][k] != 0){ ankle << i << " " << j << " " << k << endl; }
				if (left_knee_hist[i][j][k] != 0){
					left_knee << i << " " << j << " " << k << endl; 
				}
				if (right_knee_hist[i][j][k] != 0){
					right_knee << i << " " << j << " " << k << endl; 
				}
				if (left_heel_hist[i][j][k] != 0){ left_heel << i << " " << j << " " << k << endl; }
				if (right_heel_hist[i][j][k] != 0){ right_heel << i << " " << j << " " << k << endl; }
			}
		}
	}
	ankle.close();
	left_knee.close();
	right_knee.close();
	left_heel.close();
	right_heel.close();
}

/*****************色特徴空間生成*****************/
//unordered_map<Vec3b, int, HashVI> color_feature_space;  //色特徴空間本体
int color_feature_space[256][256][256];  //色特徴空間本体(ハッシュ的な役割)

void create_feature_space(int part, Vec3b val){
	int r = val[2];
	int g = val[1];
	int b = val[0];
	color_feature_space[r][g][b] = part;
}

/*****************ラベル特徴空間生成(クラスタリングに使用)*****************/

bool *label_feature_space;
/*最初のラベル割り当てに使う
  5次元空間におけるKMeansクラスタリングを実行する*/
void create_label_space(Mat& frame){
	const int height = 100;
	label_feature_space = new bool[(height+1)*128*128*128]; //[x][y][r][g][b]
	for (int y = 0; y < height; y++){
		Vec3b* ptr = frame.ptr<Vec3b>(y);
		for (int x = 0; x < width; x++){
			Vec3b color = ptr[x];
			label_feature_space[x*width+y]++;
		}
	}
}

//Labelクラスを初期化(※リファクタリングしたいなー)
void init_label_class(Mat& frame, Label *ankle_ptr, Label *left_knee_ptr,
	Label *right_knee_ptr, Label *left_heel_ptr, Label *right_heel_ptr, Label *head_ptr){
	int height = frame.rows;
	int width = frame.cols;

	create_label_space(frame);
	
	//ラベルごとの最大値の{ankle[x],ankle[y],left_knee[x],left_knee[y],x,y,...,x,y}
	int max_points[12] = {}; 
	//ラベルごとの最小値の{ankle[x],ankle[y],left_knee[x],left_knee[y],x,y,...,x,y}
	int min_points[12];
	for (int i = 0; i < 10; i++){ min_points[i] = 100000000; }

	vector<Point> ankle_point;
	vector<Point> left_knee_point;
	vector<Point> right_knee_point;
	vector<Point> left_heel_point;
	vector<Point> right_heel_point;
	vector<Point> head_point;
	for (int y = 0; y < height; y++){
		Vec3b* ptr = frame.ptr<Vec3b>(y);
		for (int x = 0; x < width; x++){
			vector<int> v{ x, y };
			Vec3b val = ptr[x];
			int r = val[2];
			int g = val[1];
			int b = val[0];
			if (labels[v] == label_num_by_id[0]){
				if (HIST == 1){
					histgram(0, val);
				}
				if (COLOR == 1){
					create_feature_space(ANKLE, val);
				}
				ankle_point.push_back(Point{ x, y });
				change_min_and_max_value(x, y, &max_points[0], &max_points[1],
						&min_points[0], &min_points[1]);
			}
			else if (labels[v] == label_num_by_id[1]){
				if (HIST == 1){
					histgram(1, val);
				}
				if (COLOR == 1){
					create_feature_space(LEFT_KNEE, val);
				}
				left_knee_point.push_back(Point{ x, y });
				change_min_and_max_value(x, y, &max_points[2], &max_points[3],
						&min_points[2], &min_points[3]);
			}
			else if (labels[v] == label_num_by_id[2]){
				if (HIST == 1){
					histgram(2, val);
				}
				if (COLOR == 1){
					create_feature_space(RIGHT_KNEE, val);
				}
				right_knee_point.push_back(Point{ x, y });
				change_min_and_max_value(x, y, &max_points[4], &max_points[5],
						&min_points[4], &min_points[5]);
			}
			else if (labels[v] == label_num_by_id[3]){
				if (HIST == 1){
					histgram(3, val);
				}
				if (COLOR == 1){
					create_feature_space(LEFT_HEEL, val);
				}
				left_heel_point.push_back(Point{ x, y });
				change_min_and_max_value(x, y, &max_points[6], &max_points[7],
						&min_points[6], &min_points[7]);
			}
			else if (labels[v] == label_num_by_id[4]){
				if (HIST == 1){
					histgram(4, val);
				}
				if (COLOR == 1){
					create_feature_space(RIGHT_HEEL, val);
				}
				right_heel_point.push_back(Point{ x, y });
				change_min_and_max_value(x, y, &max_points[8], &max_points[9],
						&min_points[8], &min_points[9]);
			}
			else if (labels[v] == label_num_by_id[5]){
				if (HIST == 1){
					//histgram(5, val);
				}
				if (COLOR == 1){
					//create_feature_space(HEAD, val);
				}
				head_point.push_back(Point{ x, y });
				change_min_and_max_value(x, y, &max_points[10], &max_points[11],
						&min_points[10], &min_points[11]);
			}
		}
	}

	Point cogs[LABEL_KIND_NUM];
	for (int i = 0; i < LABEL_KIND_NUM; i++){
		int x = (max_points[i*2] + min_points[i*2]) / 2;
	    int y = (max_points[i*2+1] + min_points[i*2+1]) / 2;
		Point cog_point{ x, y };
		cogs[i] = cog_point;
	}

	Label ankle(ANKLE, "腰", ankle_point, cogs[0]);
	Label left_knee(LEFT_KNEE, "左膝", left_knee_point, cogs[1]);
	Label right_knee(RIGHT_KNEE, "右膝", right_knee_point, cogs[2]);
	Label left_heel(LEFT_HEEL ,"左首", left_heel_point, cogs[3]);
	Label right_heel(RIGHT_HEEL, "右首", right_heel_point, cogs[4]);
	Label head(HEAD, "頭", head_point, cogs[5]);

	*ankle_ptr = ankle;
	*left_knee_ptr = left_knee;
	*right_knee_ptr = right_knee;
	*left_heel_ptr = left_heel;
	*right_heel_ptr = right_heel;
	*head_ptr = head;
}

//全ラベルのprev_pointsとcurrent_pointsを入れ替え,current_pointsをクリアする
void change_prev_and_current(Label *ankle, Label *left_knee, Label *right_knee,
	Label *left_heel, Label *right_heel, Label *head){
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
	head->clear_prev_points();
	head->change_ptr();
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
	Label *right_knee, Label *left_heel, Label *right_heel, Label *head){
	for (int y = 0; y < height; y++){
		Vec3b* ptr = frame.ptr<Vec3b>(y);
		for (int x = 0; x < width; x++){
			Vec3b color = ptr[x];
			if (color[2] > 30 && color[1] > 30 && color[0] > 30){
				Point p{ x, y };
				find_same_point(ankle, p);
				find_same_point(left_knee, p);
				find_same_point(right_knee, p);
				find_same_point(left_heel, p);
				find_same_point(right_heel, p);
				find_same_point(head, p);
			}
		}
	}
}

int bin[LABEL_KIND_NUM] = {};   //プロットした点の周辺の点のラベルを格納(ヒストグラム)
void search_color_from_feature_space(Point p, Vec3b color, Label* label){
	const int mask = 9; //処理マスク初期値
	int joint = label->get_id();
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
				if (point_validation(tr, tg, 255, 255, tb, 255, 3)){
					continue;
				}
				else{
					if (color_feature_space[tr][tg][tb] != 0){
						double dist = sqrt((tr - r)*(tr - r) + (tg - g)*(tg - g) + (tb - b)*(tb - b));
						if (dist < min_dist){
							min_dist = dist;
							min_label = color_feature_space[tr][tg][tb];
						}
//						bin[color_feature_space[tr][tg][tb]] += 1;
					}
				}
			}
		}
	}

	//最頻値計算
/*	int max_label = 0;
	for (int i = 1; i <= LABEL_KIND_NUM; i++){
		if (bin[i] > max_label){
			max_label = bin[i];
		}	
	}
	*/

	//与えられたラベルと最頻値がマッチするかどうか
	//マッチすればそのラベルの点と判断し、しなければスルー
/*	if (joint == max_label){
		label->set_current_points(p);
	}*/
	if (joint == min_label){
		label->set_current_points(p);
	}
}

//明らかに外れている点を除去
void remove_marker_noise(Mat& frame, Label* label){
	const int around_pixel_x = 5;
	const int around_pixel_y = 5;
	const int thresh = 20;
	vector<Point>& cps = label->get_current_points();
	for (auto itr = cps.begin(); itr != cps.end(); ++itr){
		Point cp = *itr;
		int blank_pixel_count = 0;
		for (int y = cp.y - around_pixel_y; y <= cp.y + around_pixel_y; y++){
			for (int x = cp.x - around_pixel_x; x <= cp.x + around_pixel_x; x++){
				vector<int> p{ x, y };
				if (cp.x == x && cp.y == y){ continue; }//同じ点ならスキップ
				Vec3b frame_p = frame.at<Vec3b>(y, x);
				if (frame_p[2] <= 10 && frame_p[1] <= 10 && frame_p[0] <= 10){
					blank_pixel_count++;
				}
			}
		}
		//すっからかんの点がthreshより多かったらその点を削除
		if (blank_pixel_count >= thresh){
			itr = cps.erase(itr);
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
	Vec3b current;
	for (int y = cp.y - (AROUND_PIXEL_Y / 2); y < cp.y + (AROUND_PIXEL_Y / 2); y++){
		Vec3b* ptr = frame.ptr<Vec3b>(y);
		for (int x = cp.x - (AROUND_PIXEL_X / 2); x < cp.x + (AROUND_PIXEL_X / 2); x++){
			current = ptr[x];
			Point p{ x, y };
			//マーカーでない点はスキップする
			if (current[2] >= 10 && current[1] >= 10 && current[0] >= 10){
				search_color_from_feature_space(p, current, label);
			}
		}
	}
}

//ラベルごとに周りの点を探索する
void search_around_points(Mat& frame, Label *ankle, Label *left_knee,
    Label *right_knee, Label *left_heel, Label *right_heel, Label *head){
	search_around_points_each_labels(frame, ankle);
	search_around_points_each_labels(frame, left_knee);
	search_around_points_each_labels(frame, right_knee);
	search_around_points_each_labels(frame, left_heel);
	search_around_points_each_labels(frame, right_heel);
	search_around_points_each_labels(frame, head);
}

//ラベルごとに重心をセットする
void set_cog_each_label(Label *ankle, Label *left_knee, Label *right_knee,
	Label *left_heel, Label *right_heel, Label *head){
	ankle->calc_and_set_cog();
	left_knee->calc_and_set_cog();
	right_knee->calc_and_set_cog();
	left_heel->calc_and_set_cog();
	right_heel->calc_and_set_cog();
	head->calc_and_set_cog();
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
void evaluate_angle_ankle_and_knees(Point ankle, Point right_knee, Point left_knee){
	evaluate_angle(ankle, right_knee, left_knee);
}

void evaluate_front_knee_angle(Point left_knee, Point ankle, Point left_heel){
	evaluate_angle(left_knee, ankle, left_heel);
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
	Label head;

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
			}
			if (MODE == 1){
			//	import_labels();
				init_label_class(frame, &ankle, &left_knee, &right_knee,
					&left_heel, &right_heel, &head);

				//ヒストグラムの出力。終わったらbreak
				if (HIST == 1){
					output_histgram_data();
					break;
				}

				switch (FEATURE){
				case 0:
					evaluate_angle_ankle_and_knees(ankle.get_cog()[count - use_start_frame],
						right_knee.get_cog()[count - use_start_frame], left_knee.get_cog()[count - use_start_frame]);
				case 1:
					evaluate_front_knee_angle(right_knee.get_cog()[count - use_start_frame],
						ankle.get_cog()[count - use_start_frame], right_heel.get_cog()[count - use_start_frame]);
				default:
					break;
				}
			/*	circle(dst_img, ankle.get_cog()[count - use_start_frame], 10, Scalar(255, 0, 0), -1);
				circle(dst_img, left_knee.get_cog()[count - use_start_frame], 5, Scalar(255, 255, 255), -1);
				circle(dst_img, left_knee.get_cog()[count - use_start_frame], 10, Scalar(0, 255, 0), -1);
				circle(dst_img, left_heel.get_cog()[count - use_start_frame], 10, Scalar(0, 0, 255), -1);
				circle(dst_img, right_heel.get_cog()[count - use_start_frame], 5, Scalar(255, 0, 255), -1);*/
			}
		}
		else if(count >= use_end_frame){
			//対象となるフレームが終わったらループを抜ける
			break;
		}
		else{
			if (MODE == 1){
				change_prev_and_current(&ankle, &left_knee, &right_knee, &left_heel, &right_heel, &head);

				search_same_points(frame, &ankle, &left_knee, &right_knee, &left_heel, &right_heel, &head);

				search_around_points(frame, &ankle, &left_knee, &right_knee, &left_heel, &right_heel, &head);

				set_cog_each_label(&ankle, &left_knee, &right_knee, &left_heel, &right_heel, &head);
				switch (FEATURE){
				case 0:
					evaluate_angle_ankle_and_knees(ankle.get_cog()[count - use_start_frame],
						right_knee.get_cog()[count - use_start_frame], left_knee.get_cog()[count - use_start_frame]);
				case 1:
					evaluate_front_knee_angle(right_knee.get_cog()[count - use_start_frame],
						ankle.get_cog()[count - use_start_frame], right_heel.get_cog()[count - use_start_frame]);
				default:
					break;
				}
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
			circle(dst_img, ankle.get_cog()[count - use_start_frame], 5, Scalar(0, 0, 255), -1);
			circle(dst_img, left_knee.get_cog()[count - use_start_frame], 5, Scalar(0, 255, 0), -1);
			circle(dst_img, right_knee.get_cog()[count - use_start_frame], 5, Scalar(255, 0, 0), -1);
			circle(dst_img, left_heel.get_cog()[count - use_start_frame], 5, Scalar(0, 255, 255), -1);
			circle(dst_img, right_heel.get_cog()[count - use_start_frame], 5, Scalar(255, 0, 255), -1);
			circle(dst_img, head.get_cog()[count - use_start_frame], 5, Scalar(255, 255, 0), -1);
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
	if (MODE == 1 && HIST == 0){
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