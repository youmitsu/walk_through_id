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
#define LABEL_KIND_NUM 13                                 //�擾���������x���̎�ސ�
#define AROUND_PIXEL_X 200                               //���݂̍��W�̎���̒T������ۂ�X�͈̔�
#define AROUND_PIXEL_Y 50                                //                              Y�͈̔�
#define ID_COUNT 4                                       //�f�[�^�ƂȂ铮��̐�
#define COLOR_DECIDE_LENGTH 9                            //�F��Ԃ��`����̂ɕK�v�ȗv�f�� ex){rs, re, gs, ge, bs, be}�̔z��
#define MODE_KIND 3
#define FEATURE_KIND 2
#define EXTRA 10
#define PARTS_LENGTH LABEL_KIND_NUM+EXTRA
#define LABEL_SIZE_THRESH 20
#define COLOR_KIND 4

/*******�u�N�́v�u���̏������v�u�����ʁv��ݒ�********/
#define ID 1                                             //0:����, 1:�G��, 2:�H�c, 3:�k��
#define MODE 1                                           //0:���x�����O���[�h 1:�ǐՃ��[�h 2:�Đ����[�h
#define FEATURE 0                                        //0:�҂̊p�x�A1:�G�̊p�x
#define HIST 1                                           //�q�X�g�O�����o��
#define COLOR 0                                          //�F������Ԑ���

using namespace std;
using namespace cv;

/*************�萔�Q(�Ȃ񂩕|������z��n��const�ɂ���)****************/
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

/************�O���[�o���ϐ��Q***********/
string video_url;                                            //�g�p���铮���URL
int use_start_frame;                                         //���悩��g���ŏ��̃t���[��
int use_frame_num;                                           //�g�p����t���[����
int use_end_frame;                                           //���悩��g���Ō�̃t���[��
int label_num_by_id[LABEL_KIND_NUM];                         //�擾�������֐߂ɊY�����郉�x���ԍ����i�[
unordered_map<int, int> lookup_table;                        //���b�N�A�b�v�e�[�u��
vector<double> angles;                                       //�t���[�����Ƃ̊֐߂̊p�x
const string output_labels_filename[ID_COUNT] = { "output_labels_hoshino.txt",  "output_labels_shuno.txt",
"output_labels_haneda.txt", "output_labels_kitazawa.txt" };

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

struct XYRGB{
	int x;
    int y;
	int r;
	int g;
	int b;
};

//�֐߂̃��f�����`
/************����**************
   1:��
   2:��
   3:����
   4:�E�I
   5:�E���
   6:���I
   7:�����
   8:��
   9:���G
   10:�E�G
   11:������
   12:�E����
*******************************/
/*
XYRGB joint_position_models[LABEL_KIND_NUM] = { { 112, 28, 120, 230, 150 }, { 125, 88, 179, 155, 202 }, { 156, 131, 56, 121, 174 },
{ 86, 215, 131, 250, 201 }, { 8, 260, 147, 127, 178 }, { 160, 240, 255, 255, 161 }, { 173, 368, 255, 255, 166 }, { 106, 279, 76, 153, 200 }, { 57, 491, 215, 238, 137 },
{ 130, 495, 133, 243, 179 }, { 50, 610, 215, 238, 137 }, { 120, 610, 147, 127, 178 } };*/

Point joint_position_models[LABEL_KIND_NUM] = { {67, 32 }, { 58, 129 }, { 65, 188 }, { 46, 151 }, { 36, 270 },
{ 3, 353 }, { 57, 329 }, { 83, 304 }, { 82, 460 }, { 29, 573 }, { 70, 573 }, { 15, 750 }, {84, 734} };

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

//���Ӎ��W�̃��x�����擾
int gather_around_label(vector<int> point, int width, int height){
	if (point_validation(point[0], point[1], width, height)){
		return -1;
	}
	else{
		return labels[point];
	}
}

//�����̃��x����ނ̑��ݗL���̔���
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

//��f�ɐV���ȃ��x�������蓖�Ă�
void assign_label(int x, int y, int width, int height ,int* latest_label_num){
	int l; //���x���̈ꎞ����p
	/********�ϐ��錾*********************************************
	* point: ���ړ_                                              *
	* leftup: ����̍��W                                         *
	* up: �^��̍��W                                             *
	* rightup: �E��̍��W                                        *
	* left: �^���̍��W                                           *
	* valid_labels: �s���ȍ��W(-1,1)�Ȃǂ��܂܂�Ă��Ȃ����x���Q *
	**************************************************************/
	vector<int> point{ x, y }, leftup{ x - 1, y - 1 }, up{ x, y - 1 },
		rightup{ x + 1, y - 1 }, left{ x - 1, y }, valid_labels;

	//valid_labels�ɕs���ȍ��W�ȊO����
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

	//valid_labels�̃[���̃J�E���g�ƃ��x���̎�ށA�ŏ��l���v�Z
	int zero_count = 0;
	vector<int> labels_except_zero;
	int min_label_num = 1000;
	for (auto itr = valid_labels.begin(); itr != valid_labels.end(); ++itr){
		if (*itr == 0){
			zero_count++;
		}
		else{
			labels_except_zero.push_back(*itr);  //0�ȊO�̃��x�����i�[
			//���x���̍ŏ��l�v�Z
			if (*itr < min_label_num){
				min_label_num = *itr;
			}
		}
	}
	//���x�����蓖��
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

//lookup�e�[�u�����烉�x���ԍ����Q�Ƃ���Ƃ��ɗp����(�l�X�g���Ă��郉�x���ɑΉ����邽��)
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

//���x���T���̍ۂɎg�p
unordered_map<int, int> index_of_labels;

//int data_size_per_cls[LABEL_KIND_NUM] = {};
//���x�����O�{��
void labeling(Mat& frame, int height_min, int height_max, int width_min, int width_max){
	const int mask = 9;
	Mat gray_img, thre_img;
	cvtColor(frame, gray_img, CV_RGB2GRAY);
	threshold(gray_img, thre_img, 0, 255, THRESH_BINARY | THRESH_OTSU);
	imwrite("thre_image.png", thre_img);
	const int width = thre_img.cols;
	const int height = thre_img.rows;
	int latest_label_num = 0;                                    //�����Ƃ��V�������x��

	//���x�����O�̂��߂̃��b�N�A�b�v�e�[�u����p��
	for (int i = 0; i < LOOKUP_SIZE; i++){
		lookup_table[i] = i;
	}

	//�S��f���x��������
	for (int y = height_min; y <= height_max; y++){
		unsigned char* ptr = thre_img.ptr<unsigned char>(y);
		for (int x = width_min; x < width_max; x++){
			int pt = ptr[x];
			vector<int> v{ x, y };
			labels[v] = 0;
		}
	}

	//���x�����O���s
	for (int y = height_min; y < height_max; y++){
		unsigned char* ptr = thre_img.ptr<unsigned char>(y);
		for (int x = width_min; x < width_max; x++){
			int p = ptr[x];
			if (p == 255){
				assign_label(x, y, frame.cols, frame.rows, &latest_label_num);
			}
		}
	}

	//���b�N�A�b�v�e�[�u����p���ă��x���̏�������
	for (auto itr = labels.begin(); itr != labels.end(); ++itr){
		int fixed_label = reference_label(itr->second);
		labels[itr->first] = fixed_label;
	}

	int index = 0;
	for (auto itr = label_list.begin(); itr != label_list.end(); ++itr){
		index_of_labels[*itr] = index;
		index++;
	}

	/********�G�������̃R�[�h(���Ԃ���Ƃ���������)********/
	/*
	unordered_map<int, int> data_size_cls;
	for (auto itr = label_list.begin(); itr != label_list.end(); ++itr){
		int label = *itr;
		data_size_cls[label] = 0;
	}
	vector<int> point;
	int label, size;
	//�ʐς����炩�ɏ��Ȃ����x��(�G��)�̏���
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
				//�������Ƃ��B�m�C�Y�������Ƃ��B
				labels[itr->first] = 0;
			}
		}
	}*/
	/******************************************************/

}

//���x�����O���ʂ��e�L�X�g�t�@�C���ɏ����o��
void output_labels(int width, int height){
	try{
		if (labels.empty()){ throw "Exception: labels����ł�"; }
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

//���x�����O���ʂ̃t�@�C�����C���|�[�g���Alabels�ɑ������
void import_labels(){
	ifstream input_labels_file;
	input_labels_file.open(output_labels_filename[ID]);
	if (input_labels_file.fail()){
		cout << "Exception: �t�@�C����������܂���B" << endl;
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

//�t�ϊ�
int inv_width_normalize(int normal_width, int resized_wmean, int width_min){
	return (int)(((double)resized_wmean*(double)normal_width / 1000.0) + (double)width_min);
}

int height_normalize(int y, int height_min, int resized_hmean){
	return (int)(((double)y - (double)height_min) / (double)resized_hmean * 1000);
}

//�t�ϊ�
int inv_height_normalize(int normal_height, int resized_hmean, int height_min){
	return (int)(((double)resized_hmean*(double)normal_height / 1000.0) + (double)height_min);
}

//���肷�鏇��ID���`
int id_orderby_dst[LABEL_KIND_NUM] = { HEAD, RIGHT_SHOULDER, NECK, LEFT_SHOULDER, RIGHT_ELBOW, LEFT_ELBOW,
RIGHT_WRIST, ANKLE, LEFT_WRIST, LEFT_KNEE, RIGHT_KNEE, LEFT_HEEL, RIGHT_HEEL };
//���肷�鏇�ɖ��O���`
string name_orderby_dst[LABEL_KIND_NUM] = { "��", "�E��", "��", "����", "�E�I", "���I",
"�E���", "��", "�����", "���G", "�E�G", "������", "�E����" };
//���肷�鏇�ɐF���`
int color_orderby_dst[LABEL_KIND_NUM] = { GREEN, YELLOW, PINK, BLUE, GREEN, YELLOW, PINK,
BLUE, GREEN, YELLOW, GREEN, YELLOW, PINK };
//index��ID��˂����ނ����ŐF���Ԃ��Ă���
int connection_joint_color[LABEL_KIND_NUM] = {};

void create_feature_space(int* color_feature_space, int type, int r, int g, int b){
	color_feature_space[r * 255 * 255 + g * 255 + b] = type;
}
int get_color_type(int* color_feature_space, int r, int g, int b){
	return color_feature_space[r * 255 * 255 + g * 255 + b];
}

int search_color_from_feature_space(int* color_feature_space, Point p, Vec3b color){
	const int mask = 9; //�����}�X�N�����l
	int r = color[2];
	int g = color[1];
	int b = color[0];

	//�F������ԓ���T��(maxk*maxk*mask�͈̔�)
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

//���x���N���X��ID��t�^
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

//Label�N���X��������(�����t�@�N�^�����O�������ȁ[)
void init_label_class(Mat& frame, Label* parts[], int* color_feature_space){
	int height = frame.rows;
	int width = frame.cols;

	vector<int> labels_minY, labels_minX, labels_maxY, labels_maxX;
	vector<int>* labels_minY_ptr = &labels_minY;//label���Ƃ̍ŏ��l�ƍő�l���v�Z���邽�߂Ɏg�p
	vector<int>* labels_minX_ptr = &labels_minX;
	vector<int>* labels_maxY_ptr = &labels_maxY;
	vector<int>* labels_maxX_ptr = &labels_maxX;
	vector<vector<Point>> parts_points;//���x�����Ƃ̍��W��ێ�����vector���`

	for (int i = 0; i < PARTS_LENGTH; i++){
		labels_minY_ptr->push_back(10000000000);
		labels_minX_ptr->push_back(10000000000);
		labels_maxY_ptr->push_back(0);
		labels_maxX_ptr->push_back(0);
		vector<Point> v;
		parts_points.push_back(v);
	}

	//����
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
		
		//���x����0��������X�L�b�v
		if (label == 0) continue;
		
		//���f�����Ă͂߂̍ۂɎg�p,�e���x�����Ƃ�x,y�̍ŏ��l���v�Z���Ă���
		check_minY(labels_minY_ptr, index_of_labels[label], p.y);
		check_minX(labels_minX_ptr, index_of_labels[label], p.x);
		check_maxY(labels_maxY_ptr, index_of_labels[label], p.y);
		check_maxX(labels_maxX_ptr, index_of_labels[label], p.x);

		parts_points[index_of_labels[label]-1].push_back(p);
	}
	
    //���ꂼ��̃��x���ɂ�����d�S�����߂�
	vector<Point> cogs;
	int cog_x, cog_y;
	for (int i = 0; i < PARTS_LENGTH; i++){
		cog_x = ((*labels_maxX_ptr)[i] + (*labels_minX_ptr)[i]) / 2;
	    cog_y = ((*labels_maxY_ptr)[i] + (*labels_minY_ptr)[i]) / 2;
		Point cog_point{ cog_x, cog_y };
		cogs.push_back(cog_point);
	}

	//�S���x����Label�N���X�̏�����
	for (int i = 0; i < PARTS_LENGTH; i++){
		*(parts[i]) = { parts_points[i], cogs[i], (*labels_minX_ptr)[i], (*labels_minY_ptr)[i] };
		vector<Point> points = parts[i]->get_current_points();
	}

	//Label�Ɋ֐�ID��name��t�^
	assign_joint_to_label(frame, parts, color_feature_space);
}

//prev_points��current_points�Ŕ���Ă���_��T�����A����Ă����current_points�ɃZ�b�g����
void find_same_point(Label *label, Point p){
	vector<Point> prev_points = label->get_prev_points();
	auto itr = find(prev_points.begin(), prev_points.end(), p);
	if (itr != prev_points.end()){
		Point sp = *itr;
		label->set_current_points(sp);
	}
}

//���x�����Ƃ�find_same_point�����s����
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

//����̓_��T������
void search_around_points_each_labels(Mat& frame, Label *label[], int* color_feature_space){
	Point cp;
	for (int i = 0; i < PARTS_LENGTH; i++){
		//parts����Ȃ��Ȃ�X�L�b�v
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
				//�}�[�J�[�łȂ��_�̓X�L�b�v����
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
		int width = frame.cols;
		int height = frame.rows;
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
vector<XYRGB> data;
//�摜�̑O�����i�m�C�Y����,���T�C�Y,�Ȃǁj
void resize_and_preproc(Mat& src, int* height_min_ptr, int* height_max_ptr, int* width_min_ptr, int* width_max_ptr,
	int* resized_width_ptr, int* resized_height_ptr, int* resized_wmean_ptr, int* resized_hmean_ptr, bool first=false){
	/*************�m�C�Y����*************/
	const int mask = 9;
	Mat filtered_img;
	medianBlur(src, filtered_img, mask);
	/***********�摜�̃��T�C�Y************/
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
	/***************�N���X�^�����O�p�̃f�[�^�\�z****************/
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

//�d�����Ă���_���Ȃ������`�F�b�N
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
			cout << "Exception: �t�@�C����������܂���B" << endl;
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

	//�F���t�f�[�^�̍\�z
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
		int height_min, height_max, width_min, width_max; //�l�Ԃ̗̈��x,y�̍ŏ��l�ƍő�l
		int	resized_width, resized_height, resized_hmean, resized_wmean; //�g���~���O�����摜�̕��A�����A���ϒl
		//�Ώۂ̃t���[���܂ł̓X�L�b�v
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

			//�����Parts�����݂����Ƃ�
			//vector<Point> test_point = parts[0]->get_current_points();
			//test_points.push_back(test_point);
	
			
			//�F�̃T���v�����o�͂������Ƃ�
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
			//�ΏۂƂȂ�t���[�����I������烋�[�v�𔲂���
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