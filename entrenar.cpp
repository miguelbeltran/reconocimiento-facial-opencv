#include "opencv2/core/core.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <time.h>

using namespace cv;
using namespace std;

string cascadeName = "haarcascade_frontalface_alt.xml";
CascadeClassifier cascade;

/*
	Lee el conjunto de imágenes que usaremos para entrenar el modelo y las etiqueta por carpetas.
*/
static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ';');

/*
	Detección y recorte de una cara humana.
	Recibe una imagen en la cual intentaremos detectar una cara.
	Recibe un clasificador que usará para la detección.
	Devuelve 'true' si detecta una cara, 'false' si no.
*/
bool detect(Mat& img, CascadeClassifier& cascade, Mat& imRecortada);

int main(int argc, const char *argv[]){

    if (argc != 2) {
        cout << "Uso: " << argv[0] << " <csv.ext> " << endl;
        exit(1);
    }

    if(!cascade.load(cascadeName)) {
        cerr << "ERROR: No se puede cargar el clasificador." << endl;
        exit(1);
    }

    string fn_csv = string(argv[1]);
    vector<Mat> images;
    vector<int> labels;
	
    try {
        read_csv(fn_csv, images, labels);
    } catch (cv::Exception& e) {
        cerr << "Error opening file \"" << fn_csv << "\". Reason: " << e.msg << endl;
        exit(1);
    }

    Ptr<FaceRecognizer> modelEigen = createEigenFaceRecognizer(80, 10500);
    Ptr<FaceRecognizer> modelFisher = createFisherFaceRecognizer(2, 1500);    
	modelEigen->train(images, labels);
	modelFisher->train(images, labels);
	modelEigen->save("bdEigen.xml");
	modelFisher->save("bdFisher.xml");

	cout << "Entrenamiento de la BD efectuado con " << images.size() << " imágenes." << endl;
	return 0;
}

/*
	Lee el conjunto de imágenes que usaremos para entrenar el modelo y las etiqueta.
*/
static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator) {
    
    std::ifstream file(filename.c_str(), ifstream::in);
    if (!file){
        string error_message = "No valid input file was given, please check the given filename.";
        CV_Error(CV_StsBadArg, error_message);
    }
    string line, path, classlabel;
    while (getline(file, line)){
        stringstream liness(line);
        getline(liness, path, separator);
        getline(liness, classlabel);
        if(!path.empty() && !classlabel.empty()) {
        	Mat im = imread(path, CV_LOAD_IMAGE_COLOR);
            Mat imRecortada;
            if (detect(im, cascade, imRecortada)){
                Mat grayRecortada(cvRound(imRecortada.rows), cvRound(imRecortada.cols), CV_8UC1);
                cvtColor(imRecortada, grayRecortada, CV_BGR2GRAY);
                Mat grayRecortadaStandard(280, 240, CV_8UC1);
                resize(grayRecortada, grayRecortadaStandard, grayRecortadaStandard.size(), 0, 0, INTER_LINEAR);
                equalizeHist(grayRecortadaStandard, grayRecortadaStandard);
            	images.push_back(grayRecortadaStandard);
	            labels.push_back(atoi(classlabel.c_str()));
            }
        }
    }
}


/*
	Detección y recorte de una cara humana.
	Recibe una imagen en la cual intentaremos detectar una cara.
	Recibe un clasificador que usará para la detección.
	Devuelve 'true' si detecta una cara, 'false' si no.
*/
bool detect(Mat& img, CascadeClassifier& cascade, Mat& imRecortada) {
    vector<Rect> faces;
    Mat gray, smallImg(cvRound (img.rows), cvRound(img.cols), CV_8UC1);
    cvtColor(img, gray, CV_BGR2GRAY);
    resize(gray, smallImg, smallImg.size(), 0, 0, INTER_LINEAR);
    equalizeHist(smallImg, smallImg);
    cascade.detectMultiScale(smallImg, faces, 1.1, 1, 0|CV_HAAR_FIND_BIGGEST_OBJECT|CV_HAAR_DO_ROUGH_SEARCH, Size(250, 250));
    
    if(faces.size() == 0){
        return false;
    } else {
        for(vector<Rect>::const_iterator r = faces.begin(); r != faces.end(); r++){
            Point center;
            int radius;
            center.x = cvRound((r->x + r->width*0.5));
            center.y = cvRound((r->y + r->height*0.5));
            radius = cvRound((r->width + r->height)*0.25);
            
	        // Recortando la cara, despreciando el resto de la imagen.
			Point x1((center.x - radius + 20), (center.y - radius));
			Point x2((center.x + radius - 20), (center.y + radius));
			Rect myROI(x1.x, x1.y, (x2.x-x1.x), (x2.y-x1.y));        
            imRecortada = img(myROI);
        }
        return true;
    }
}
