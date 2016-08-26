// testApp.cpp : �R���\�[�� �A�v���P�[�V�����̃G���g�� �|�C���g���`���܂��B
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
#define LOOKUP_SIZE 100                                  //���b�N�A�b�v�e�[�u���̃f�t�H���g�T�C�Y
#define LABEL_KIND_NUM 12                                 //�擾���������x���̎�ސ�
#define AROUND_PIXEL_X 200                               //���݂̍��W�̎���̒T������ۂ�X�͈̔�
#define AROUND_PIXEL_Y 50                                //                              Y�͈̔�
#define ID_COUNT 4                                       //�f�[�^�ƂȂ铮��̐�
#define COLOR_DECIDE_LENGTH 9                            //�F��Ԃ��`����̂ɕK�v�ȗv�f�� ex){rs, re, gs, ge, bs, be}�̔z��
#define MODE_KIND 3
#define FEATURE_KIND 2

/*******�u�N�́v�u���̏������v�u�����ʁv��ݒ�********/
#define ID 0                                             //0:����, 1:�G��, 2:�H�c, 3:�k��
#define MODE 1                                           //0:���x�����O���[�h 1:�ǐՃ��[�h 2:�Đ����[�h
#define FEATURE 0                                        //0:�҂̊p�x�A1:�G�̊p�x
#define HIST 1                                           //�q�X�g�O�����o��
#define COLOR 0                                          //�F������Ԑ���

using namespace std;
using namespace cv;

/*************�萔�Q(�Ȃ񂩕|������z��n��const�ɂ���)****************/
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

/************�O���[�o���ϐ��Q***********/
string video_url;                                            //�g�p���铮���URL
int use_start_frame;                                         //���悩��g���ŏ��̃t���[��
int use_frame_num;                                           //�g�p����t���[����
int use_end_frame;                                           //���悩��g���Ō�̃t���[��
int label_num_by_id[LABEL_KIND_NUM];                         //�擾�������֐߂ɊY�����郉�x���ԍ����i�[
unordered_map<int, int> lookup_table;                        //���b�N�A�b�v�e�[�u��
int latest_label_num = 0;                                    //���x�����O�Ŏg�p����
int width;                                                   //�摜�̕�
int height;                                                  //����
vector<double> angles;                                       //�t���[�����Ƃ̊֐߂̊p�x
const string output_labels_filename[ID_COUNT] = { "output_labels_hoshino.txt",  "output_labels_shuno.txt",
"output_labels_haneda.txt", "output_labels_kitazawa.txt" };
int height_min, height_max, width_min, width_max;
int	resized_width, resized_height, resized_mean;

//�O���[�o���ϐ��̏�����
void init_config(){
	try{
		if (ID < 0 || ID >= ID_COUNT){ throw "Exception: ID���͈͊O�ł��B"; }
		if (MODE < 0 || MODE >= MODE_KIND){ throw "Exception: MODE���͈͊O�ł��B"; }
		if (FEATURE < 0 || FEATURE >= FEATURE_KIND){ throw "Exception: FEATURE���͈͊O�ł��B"; }
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

//vector���L�[�Ƃ���n�b�V���}�b�v���g�p���邽�߂̃N���X
class HashVI{
public:
	size_t operator()(const vector<int> &x) const {
		const int C = 997;      // �f��
		size_t t = 0;
		for (int i = 0; i != x.size(); ++i) {
			t = t * C + x[i];
		}
		return t;
	}
};

/*********************************************************************************
*                                                                                *
*  Label�N���X                                                                   *
*  private:                                                                      *
*    char name : ���x���̖��O                                                    *
*    Point cog : ���x���̏d�S�ʒu                                                *
*    prev_points : �O�̃t���[���̍��W                                            *
*    current_points : ���݂̃t���[���̍��W                                       *
*  public:                                                                       *
*    �f�t�H���g�R���X�g���N�^                                                    *
*    �R���X�g���N�^                                                              *
*      �����F���O, ���݂̍��W, �ŏ����x���̏d�S, ���x���̐F���                  *
*      ����F                                                                    *
*	   �@�E�����̒l�������o�ϐ��ɃZ�b�g                                          *
*		 �E���prev_points��������                                               *
*     �����o�֐�                                                                 *
*	   �Eget_name():���O��Ԃ�                                                   *
*	   �Eget_color_space():���x���̐F��Ԏ擾                                    *
*	   �Eget_current_points():���݂̍��W�擾                                     *
*	   �Eget_prev_points():1�t���[���O�̍��W�擾                                 *
*	   �Eget_cog():�d�S�̈ꗗ�擾                                                *
*	   �Eset_current_points(Point p):p��current_points��push_back����            *
*	   �Eset_cog(Point p):p��cog��push_back����                                  *
*	   �Ecalc_and_set_cog():current_points�̍��W����d�S���v�Z���Acog�ɃZ�b�g����*
*	   �Echange_ptr():current_points���N���A���Aprev_points�Ɉڂ��B              *
*	   �Eclear_prev_points():prev_points���N���A����                             *
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

unordered_map<vector<int>, int, HashVI> labels;  //key�F���W�Avalue�F���x���ԍ�
vector<int> label_list;                          //�S���x���̈ꗗ
unordered_map<int, Vec3b> label_color_list;      //���x�����Ƃ̐F

//���W�̐������`�F�b�N
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
			throw("�������������w�肵�Ă�������");
		}
		catch (char *e){
			cout << e;
		}
	}
}

//�����_����RGB�l��Ԃ�
Scalar get_label_color(){
	const int MAX_VALUE = 255;
	unsigned int r = rand() % MAX_VALUE;
	unsigned int g = rand() % MAX_VALUE;
	unsigned int b = rand() % MAX_VALUE;
	return Scalar(r, g, b);
}

//�ő�l�ƍŏ��l�����߂�
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

//�w�肵���f�[�^�_���}�b�v���ɑ��݂��邩
bool label_exist (vector<int> yrgb){
	auto itr = labels.find(yrgb);
	if (itr != labels.end()){
		return true;
	}
	else{
		return false;
	}
}

//���t�t������
int search_label_with_supervised_classification(vector<int> yrgb){
	const int mask = 9;
	int y, r, g, b, ty, tr, tg, tb;
	y = yrgb[0];
	r = yrgb[1];
	g = yrgb[2];
	b = yrgb[3];
	
	double dist;
	double min_dist = 1000000000.0;
	int min_label = -1;

	for (ty = y - (int)(mask / 2); ty <= y + (int)(mask / 2); ty++){
		for (tr = r - (int)(mask / 2); tr <= r + (int)(mask / 2); tr++){
			for (tg = g - (int)(mask / 2); tg <= g + (int)(mask / 2); tg++){
				for (tb = b - (int)(mask / 2); tb <= b + (int)(mask / 2); tb++){
					if (!label_exist(yrgb)){
						continue;
					}
					else{
						double dist = sqrt((ty - y)*(ty - y) + (tr - r)*(tr - r) + (tg - g)*(tg - g) + (tb - b)*(tb - b));
						if (dist < min_dist){
							min_dist = dist;
							min_label = labels[yrgb];
						}
					}
				}
			}
		}
	}
	return min_label;
}

//y�l��resized_height�ɂ���Đ��K������
int height_normalize(int y){
	return (y - height_min) / resized_mean;
}

//�摜���烉�x����T������
void search_points_from_image(Mat& frame, Label* parts[]){
	int r, g, b, y, label;
	const int rgb_thresh = 10;
	Point p;
	Vec3b val;
	vector<int> yrgb;
	for (int y = 0; y < height; y++){
		Vec3b* ptr = frame.ptr<Vec3b>(y);
		for (int x = 0; x < width; x++){
			if (y < height_min || y > height_max || width_min < x || width_max > x){
				continue;
			}
			p = { x, y };
			val = ptr[x];

			y = height_normalize(y);
			r = val[2];
			g = val[1];
			b = val[0];
			yrgb = { y, r, g, b };

			if (r >= rgb_thresh && g >= rgb_thresh && b >= rgb_thresh){
				label = search_label_with_supervised_classification(yrgb);
				parts[label]->set_current_points(p);
			}
		}
	}
}

//Label�N���X��������(�����t�@�N�^�����O�������ȁ[)
void init_label_class(Mat& frame, Label* parts[]){
	int height = frame.rows;
	int width = frame.cols;
	int i;
	//���x�����Ƃ̍ő�l��{ankle[x],ankle[y],left_knee[x],left_knee[y],x,y,...,x,y}
	int max_points[LABEL_KIND_NUM*2] = {}; 
	//���x�����Ƃ̍ŏ��l��{ankle[x],ankle[y],left_knee[x],left_knee[y],x,y,...,x,y}
	int min_points[LABEL_KIND_NUM*2];
	for (int i = 0; i < LABEL_KIND_NUM*2; i++){ min_points[i] = 100000000; }

	//���x�����Ƃ̍��W��ێ�����vector���`
	vector<Point> parts_points[LABEL_KIND_NUM];
	for (i = 0; i < LABEL_KIND_NUM; i++){
		vector<Point> v;
		parts_points[i] = v;
	}

	//����
	int r, g, b, y, label;
	Point p;
	Vec3b val;
	vector<int> yrgb;
	for (int y = 0; y < height; y++){
		Vec3b* ptr = frame.ptr<Vec3b>(y);
		for (int x = 0; x < width; x++){
			if (y < height_min || y > height_max || width_min < x || width_max > x){
				continue;
			}
			p = { x, y };
			val = ptr[x];

			y = height_normalize(y);
			r = val[2];
			g = val[1];
			b = val[0];
			yrgb = { y, r, g, b };

			if (r >= rgb_thresh && g >= rgb_thresh && b >= rgb_thresh){
				label = search_label_with_supervised_classification(yrgb);
				parts_points[label].push_back(Point{ x, y });
				change_min_and_max_value(x, y, &max_points[label * 2], &max_points[label * 2 + 1], &min_points[label * 2], &min_points[label * 2 + 1]);
			}
		}
	}

    //���ꂼ��̃��x���ɂ�����d�S�����߂�
	Point cogs[LABEL_KIND_NUM];
	int cog_x, cog_y;
	for (i = 0; i < LABEL_KIND_NUM; i++){
		cog_x = (max_points[i*2] + min_points[i*2]) / 2;
	    cog_y = (max_points[i*2+1] + min_points[i*2+1]) / 2;
		Point cog_point{ cog_x, cog_y };
		cogs[i] = cog_point;
	}

	//�S���x����Label�N���X�̏�����
	for (i = 0; i < LABEL_KIND_NUM; i++){
		*parts[i] = { ANKLE, "��", parts_points[i], cogs[i] };
	}
}

//�S���x����prev_points��current_points�����ւ�,current_points���N���A����
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

//3�_��^����ꂽ�Ƃ��Ɋp�x�����߂�
//c:�p�x�̊�_�Aa,b:����ȊO
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

//���ƉE�G�A���G���琬���p�x�����߁Aangles��push����
void evaluate_angle_ankle_and_knees(Point* parts){
//	evaluate_angle(ankle, right_knee, left_knee);
}

void evaluate_front_knee_angle(Point* parts){
//	evaluate_angle(left_knee, ankle, left_heel);
}

//�����̓���Đ��̂��߂̃��\�b�h
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
		//�Ώۂ̃t���[���܂ł̓X�L�b�v
		if (count < use_start_frame){
			continue;
		}
		cout << count << endl;
		imshow("test", frame);
		waitKey(30);
	}
}
vector<YRGB> data;
//�摜�̑O�����i�m�C�Y����,���T�C�Y,�Ȃǁj
Mat resize_and_preproc(Mat& src, bool first=false){
	/*************�m�C�Y����*************/
	const int mask = 9;
	Mat filtered_img;
	medianBlur(src, filtered_img, mask);
	/***********�摜�̃��T�C�Y************/
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
	/***************�N���X�^�����O�p�̃f�[�^�\�z****************/
	if (first){
		YRGB p;
		for (y = height_min; y <= height_max; y++){
			Vec3b* ptr = src.ptr<Vec3b>(y);
			for (x = width_min; x < width_max; x++){
				Vec3b c = ptr[x];
				if (c[2] > 10 && c[1] > 10 && c[0] > 10){
					//change_label_feature_space(y, c[2], c[1], c[0], true);
					p = { y - resized_mean, c[2], c[1], c[0] };
					data.push_back(p);
				}
			}
		}
	}
	return resized_img;
}

void k_means_clustering(){
	if (data.empty()){
		cout << "data���Ȃ���" << endl;
	}
	const int k = 12;   //���x���̐�
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
	vector<int> yrgb; //clsLabel�̂��߂�ނ𓾂��i�ق�Ƃ�struct�ł�肽���j

	for (int i = 0; i < k; i++){
		randIndex = (int)rand() % (nData + 1);
		kCenter[i].y = (int)data[randIndex].y;
		kCenter[i].r = (int)data[randIndex].r;
		kCenter[i].g = (int)data[randIndex].g;
		kCenter[i].b = (int)data[randIndex].b;
	}

	//�N���X�^�Z���^���ω����Ȃ��Ȃ�܂ŌJ��Ԃ�
	int iterCount = 0;
	do{
		//�N���X�^���蓖��
		changed = false;
		for (int i = 0; i < k; i++){
			clsCount[i] = 0;
			total[i].y = 0;
			total[i].r = 0;
			total[i].g = 0;
			total[i].b = 0;
		}
		labels.clear();
		//�e�f�[�^�_�ƃN���X�^�Z���^�ԂƂ̋������v�Z
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
						minIndex = j;
					}
				}
				else{
					minIndex = j;
					break;
				}
			}
			yrgb = { dy, dr, dg, db };
			labels[yrgb] = minIndex;
			total[minIndex].y += dy;
			total[minIndex].r += dr;
			total[minIndex].g += dg;
			total[minIndex].b += db;
			clsCount[minIndex]++;
		}
		//�V�����N���X�^�Z���^�𓾂�
		//�N���X�^���̕��ϒl�̎Z�o
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
		//�V�����N���X�^�Z���^�������_�ł��邩�ǂ����̔���
		if (countMatch == k){
			changed = false;
		}
		else{
			changed = true;
		}
		iterCount++;
	} while (changed);
	//�������̉��
	data.clear();
	data.shrink_to_fit();
}

int _tmain(int argc, _TCHAR* argv[])
{
	init_config();
	//�t�@�C���o�͗p�̃t�@�C������`
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
	const int video_size = video.get(CV_CAP_PROP_FRAME_COUNT);  //�r�f�I�̃t���[����

	switch (MODE){
	case 2:
		play(video);
		break;
	default:
		break;
	}
	//���A���G�A�E�G�A������A�E�����Label�C���X�^���X��錾
	Label* parts[LABEL_KIND_NUM];
	for (int i = 0; i < LABEL_KIND_NUM; i++){
		Label* ls;
		Label l;
		ls = &l;
		parts[i] = ls;
	}

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
		//�Ώۂ̃t���[���܂ł̓X�L�b�v
		if (count < use_start_frame){
			continue;
		}
		else if (count == use_start_frame){
			resized_img = resize_and_preproc(frame, true);
			k_means_clustering();
			init_label_class(frame, parts);
	/*		switch (FEATURE){
			case 0:
				evaluate_angle_ankle_and_knees(ankle.get_cog()[count - use_start_frame],
					right_knee.get_cog()[count - use_start_frame], left_knee.get_cog()[count - use_start_frame]);
			case 1:
				evaluate_front_knee_angle(right_knee.get_cog()[count - use_start_frame],
					ankle.get_cog()[count - use_start_frame], right_heel.get_cog()[count - use_start_frame]);
			default:
				break;
			}*/
		/*	circle(dst_img, ankle.get_cog()[count - use_start_frame], 10, Scalar(255, 0, 0), -1);
			circle(dst_img, left_knee.get_cog()[count - use_start_frame], 5, Scalar(255, 255, 255), -1);
			circle(dst_img, left_knee.get_cog()[count - use_start_frame], 10, Scalar(0, 255, 0), -1);
			circle(dst_img, left_heel.get_cog()[count - use_start_frame], 10, Scalar(0, 0, 255), -1);
			circle(dst_img, right_heel.get_cog()[count - use_start_frame], 5, Scalar(255, 0, 255), -1);*/
		}
		else if(count >= use_end_frame){
			//�ΏۂƂȂ�t���[�����I������烋�[�v�𔲂���
			break;
		}
		else{
			if (MODE == 1){
				resized_img = resize_and_preproc(frame);
				change_prev_and_current(parts);
				search_points_from_image(frame, parts);
				set_cog_each_label(parts);
	/*			switch (FEATURE){
				case 0:
					evaluate_angle_ankle_and_knees(ankle.get_cog()[count - use_start_frame],
						right_knee.get_cog()[count - use_start_frame], left_knee.get_cog()[count - use_start_frame]);
				case 1:
					evaluate_front_knee_angle(right_knee.get_cog()[count - use_start_frame],
						ankle.get_cog()[count - use_start_frame], right_heel.get_cog()[count - use_start_frame]);
				default:
					break;
				}*/
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
				cout << parts[i]->get_cog()[count - use_start_frame].x << ", " << parts[i]->get_cog()[count - use_start_frame].y << endl;
				circle(dst_img, parts[i]->get_cog()[count - use_start_frame], 5, get_label_color(), -1);
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
			cout << i << "�t���[����:" << angles[i] << endl;
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
		imwrite("�o�͌���.png", dst_img);
	}
	catch (runtime_error& ex){
		printf("failure");
		return 1;
	}
	/**********�t�[���G�ϊ��ƃv���b�g***********/
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

	namedWindow("���x�����O����");
	imshow("���x�����O����", dst_img);
	cout << "�v���O�����̏I��" << endl;
	cin.get();
	waitKey(0);
	return 0;
}