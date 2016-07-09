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
#define LOOKUP_SIZE 100                                  //���b�N�A�b�v�e�[�u���̃f�t�H���g�T�C�Y
#define LABEL_KIND_NUM 5                                 //�擾���������x���̎�ސ�
#define AROUND_PIXEL_X 500                               //���݂̍��W�̎���̒T������ۂ�X�͈̔�
#define AROUND_PIXEL_Y 80                                //                              Y�͈̔�
#define ID_COUNT 4                                       //�f�[�^�ƂȂ铮��̐�
#define COLOR_DECIDE_LENGTH 6                            //�F��Ԃ��`����̂ɕK�v�ȗv�f�� ex){rs, re, gs, ge, bs, be}�̔z��
#define MODE_KIND 3

/*******�u�N�́v�u���̏������v��ݒ�********/
#define ID 0                                             //0:����, 1:�G��, 2:�H�c, 3:�k��
#define MODE 0                                           //0:���x�����O�e�X�g���[�h 1:�ǐՃ��[�h 2:�Đ����[�h

using namespace std;
using namespace cv;

/*************�萔�Q(�Ȃ񂩕|������z��n��const�ɂ���)****************/
const string video_urls[ID_COUNT] = { "Hoshino.avi", "Shuno.avi", "Haneda.avi", "Kitazawa.avi" };
const int use_start_frames[ID_COUNT] = { 400, 207, 529, 1832 };
const int use_frame_nums[ID_COUNT] = { 32, 32, 32, 38 };
//���x�����Ƃ̐F��Ԃ��`
const unsigned int ankle_color_spaces[ID_COUNT][COLOR_DECIDE_LENGTH] = { { 0, 50, 50, 255, 150, 255 },
{ 0, 50, 50, 255, 150, 255 },
{ 0, 50, 50, 255, 150, 255 },
{ 0, 50, 50, 255, 150, 255 } };      //��
const unsigned int left_knee_color_spaces[ID_COUNT][COLOR_DECIDE_LENGTH] = { { 0, 80, 150, 255, 0, 80 },
{ 0, 80, 150, 255, 0, 150 },
{ 0, 80, 150, 255, 0, 150 },
{ 0, 80, 150, 255, 0, 150 } };      //��
const unsigned int right_knee_color_spaces[ID_COUNT][COLOR_DECIDE_LENGTH] = { { 180, 255, 170, 255, 0, 150 },
{ 180, 255, 170, 255, 0, 150 },
{ 180, 255, 170, 255, 0, 150 },
{ 180, 255, 170, 255, 0, 150 } };     //���F
const unsigned int left_heel_color_spaces[ID_COUNT][COLOR_DECIDE_LENGTH] = { { 180, 255, 170, 255, 0, 150 },
{ 180, 255, 170, 255, 0, 150 },
{ 180, 255, 170, 255, 0, 150 },
{ 180, 255, 170, 255, 0, 150 } };      //���F
const unsigned int right_heel_color_spaces[ID_COUNT][COLOR_DECIDE_LENGTH] = { { 100, 255, 0, 100, 100, 255 },
{ 100, 255, 0, 100, 100, 255 },
{ 100, 255, 0, 100, 100, 255 },
{ 100, 255, 0, 100, 100, 255 } };     //��

const int labels_each_ids[ID_COUNT][LABEL_KIND_NUM] = { { 15, 25, 31, 38, 41 },
{ 21, 34, 35, 44, 45 },
{ 24, 33, 37, 41, 47 },
{ 30, 50, 48, 57, 59 } };

/************�O���[�o���ϐ��Q***********/
string video_url;                                            //�g�p���铮���URL
int use_start_frame;                                         //���悩��g���ŏ��̃t���[��
int use_frame_num;                                           //�g�p����t���[����
int use_end_frame;                                           //���悩��g���Ō�̃t���[��
vector<unsigned int> ankle_color_space;         //���ɊY������F���
vector<unsigned int> left_knee_color_space;     //���G
vector<unsigned int> right_knee_color_space;    //�E�G
vector<unsigned int> left_heel_color_space;     //������
vector<unsigned int> right_heel_color_space;    //�E����
int label_num_by_id[LABEL_KIND_NUM];                         //�擾�������֐߂ɊY�����郉�x���ԍ����i�[
unordered_map<int, int> lookup_table;                        //���b�N�A�b�v�e�[�u��
int latest_label_num = 0;                                    //���x�����O�Ŏg�p����
int width;                                                   //�摜�̕�
int height;                                                  //����
vector<double> angles;                                       //�t���[�����Ƃ̊֐߂̊p�x
const string output_labels_filename[ID_COUNT] = { "output_labels_hoshino.txt",  "output_labels_shuno.txt",
"output_labels_haneda.txt", "output_labels_kitazawa.txt"};

//�O���[�o���ϐ��̏�����
void init_config(){
	try{
		if (ID < 0 || ID >= ID_COUNT){ throw "Exception: ID���͈͊O�ł��B"; }
		if (MODE < 0 || MODE >= MODE_KIND){ throw "Exception: MODE���͈͊O�ł��B"; }
	}
	catch (char *e){
		cout << e;
	}
	video_url = video_urls[ID];
	use_start_frame = use_start_frames[ID];
	use_frame_num = use_frame_nums[ID];
	use_end_frame = use_start_frame + use_frame_num;
	//�F��ԏ�����(���̃R�[�h���߂��˂�)
	for (int i = 0; i < COLOR_DECIDE_LENGTH; i++){
		ankle_color_space.push_back(ankle_color_spaces[ID][i]);
		left_knee_color_space.push_back(left_knee_color_spaces[ID][i]);
		right_knee_color_space.push_back(right_knee_color_spaces[ID][i]);
		left_heel_color_space.push_back(left_heel_color_spaces[ID][i]);
		right_heel_color_space.push_back(right_heel_color_spaces[ID][i]);
	}
	for (int i = 0; i < LABEL_KIND_NUM; i++){ label_num_by_id[i] = labels_each_ids[ID][i]; }
}

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

unordered_map<vector<int>, int, HashVI> labels;  //key�F���W�Avalue�F���x���ԍ�
vector<int> label_list;                          //�S���x���̈ꗗ
unordered_map<int, Vec3b> label_color_list;      //���x�����Ƃ̐F

//���W�̐������`�F�b�N
bool point_validation(int x, int y, int width, int height){
	if (x < 0 || x > width || y < 0 || y > height) {
		return true;
	}
	else{
		return false;
	}
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
void assign_label(int x, int y, int width, int height){
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

//�����_����RGB�l��Ԃ�
Vec3b get_label_color(){
	const int MAX_VALUE = 255;
	unsigned int r = rand() % MAX_VALUE;
	unsigned int g = rand() % MAX_VALUE;
	unsigned int b = rand() % MAX_VALUE;
	return Vec3b(r, g, b);
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

//���x�����O�{��
void labeling(Mat& frame){
	Mat gray_img, thre_img;
	cvtColor(frame, gray_img, CV_RGB2GRAY);
	threshold(gray_img, thre_img, 0, 255, THRESH_BINARY | THRESH_OTSU);
	const int width = thre_img.cols;
	const int height = thre_img.rows;

	//���x�����O�̂��߂̃��b�N�A�b�v�e�[�u����p��
	for (int i = 0; i < LOOKUP_SIZE; i++){
		lookup_table[i] = i;
	}

	//�S��f���x��������
	for (int y = 0; y < height; y++){
		unsigned char* ptr = thre_img.ptr<unsigned char>(y);
		for (int x = 0; x < width; x++){
			int pt = ptr[x];
			vector<int> v{ x, y };
			labels[v] = 0;
		}
	}
	
	//���x�����O���s
	for (int y = 0; y < height; y++){
		unsigned char* ptr = thre_img.ptr<unsigned char>(y);
		for (int x = 0; x < width; x++){
			if (ptr[x] > 200){
				assign_label(x, y, width, height);
			}
		}
	}

	//���b�N�A�b�v�e�[�u����p���ă��x���̏�������
	for (int y = 0; y < height; y++){
		unsigned char* ptr = thre_img.ptr<unsigned char>(y);
		for (int x = 0; x < width; x++){
			vector<int> v{ x, y };
			labels[v] = reference_label(labels[v]);
		}
	}
}

//���x�����O���ʂ��e�L�X�g�t�@�C���ɏ����o��
void output_labels(){
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
			out_labels << label << endl;
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
	while (getline(input_labels_file, str)){
		
	}
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
//Label�N���X��������
void init_label_class(Mat& frame, Label *ankle_ptr, Label *left_knee_ptr,
	Label *right_knee_ptr, Label *left_heel_ptr, Label *right_heel_ptr){
	int height = frame.rows;
	int width = frame.cols;
	
	//���x�����Ƃ̍ő�l��{ankle[x],ankle[y],left_knee[x],left_knee[y],x,y,...,x,y}
	int max_points[10] = {}; 
	//���x�����Ƃ̍ŏ��l��{ankle[x],ankle[y],left_knee[x],left_knee[y],x,y,...,x,y}
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

	Label ankle('��', ankle_point, cogs[0], ankle_color_space);
	Label left_knee('���G', left_knee_point, cogs[1], left_knee_color_space);
	Label right_knee('�E�G', right_knee_point, cogs[2], right_knee_color_space);
	Label left_heel('����', left_heel_point, cogs[3], left_heel_color_space);
	Label right_heel('�E��', right_heel_point, cogs[4], right_heel_color_space);

	*ankle_ptr = ankle;
	*left_knee_ptr = left_knee;
	*right_knee_ptr = right_knee;
	*left_heel_ptr = left_heel;
	*right_heel_ptr = right_heel;
}

//�S���x����prev_points��current_points�����ւ�,current_points���N���A����
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

//����̓_��T������
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

//���x�����ƂɎ���̓_��T������
void search_around_points(Mat& frame, Label *ankle, Label *left_knee,
    Label *right_knee, Label *left_heel, Label *right_heel){
	search_around_points_each_labels(frame, ankle);
	search_around_points_each_labels(frame, left_knee);
	search_around_points_each_labels(frame, right_knee);
	search_around_points_each_labels(frame, left_heel);
	search_around_points_each_labels(frame, right_heel);
}

//���x�����Ƃɏd�S���Z�b�g����
void set_cog_each_label(Label *ankle, Label *left_knee, Label *right_knee,
	Label *left_heel, Label *right_heel){
	ankle->calc_and_set_cog();
	left_knee->calc_and_set_cog();
	right_knee->calc_and_set_cog();
	left_heel->calc_and_set_cog();
	right_heel->calc_and_set_cog();
}


//���ƉE�G�A���G���琬���p�x�����߁Aangles��push����
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
		//�Ώۂ̃t���[���܂ł̓X�L�b�v
		if (count < use_start_frame){
			continue;
		}
		else if (count == use_start_frame){
			//�ŏ��̃t���[���Ń��x�����O��Label�N���X��������
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
			//�ΏۂƂȂ�t���[�����I������烋�[�v�𔲂���
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
			cout << i << "�t���[����:" << angles[i] << endl;
			ofs << i << ", " << angles[i] << endl;
			circle(dst_img, ankle.get_cog()[i], 1, Scalar(0, 0, 255), -1);
			circle(dst_img, left_knee.get_cog()[i], 1, Scalar(0, 255, 100), -1);
			circle(dst_img, right_knee.get_cog()[i], 1, Scalar(255, 0, 0), -1);
			circle(dst_img, left_heel.get_cog()[i], 1, Scalar(255, 217, 0), -1);
			circle(dst_img, right_heel.get_cog()[i], 1, Scalar(255, 0, 255), -1);
		}
	}
	
	try{
		imwrite("�o�͌���.png", dst_img);
	}
	catch (runtime_error& ex){
		printf("failure");
		return 1;
	}
	/**********�t�[���G�ϊ��ƃv���b�g***********/
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

	namedWindow("���x�����O����");
	imshow("���x�����O����", dst_img);
	waitKey(0);
	return 0;
}