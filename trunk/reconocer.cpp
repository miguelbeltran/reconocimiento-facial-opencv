#include "opencv2/contrib/contrib.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <iterator>
#include <stdio.h>

using namespace std;
using namespace cv;

/*
	Detección y recorte de una cara humana.
	Recibe una imagen en la cual intentaremos detectar una cara.
	Recibe un clasificador que usará para la detección.
	Devuelve 'true' si detecta una cara, 'false' si no.
*/
void detect(Mat& img, CascadeClassifier& cascade);

/*
	Reconocimiento Facial.
	Recibe la imagen de una cara a reconocer.
	Devuelve 0 si coincide la cara con alguna de la BD.
	Devuelve -1 si no reconoce ninguna persona.
*/
int recognize(Mat& img);

string cascadeName = "haarcascade_frontalface_alt.xml";
const double umbralFijoEigen = 10500;
const double umbralFijoFisher = 1500;

Ptr<FaceRecognizer> modelEigen = createEigenFaceRecognizer(80, umbralFijoEigen);
Ptr<FaceRecognizer> modelFisher = createFisherFaceRecognizer(2, umbralFijoFisher);

int main( int argc, const char** argv ) {
    Mat frame, frameCopy;
    CascadeClassifier cascade;
    if(!cascade.load(cascadeName)) {
        cerr << "ERROR: No se puede cargar el clasificador" << endl;
        return -1;
    }

    CvCapture* capture = cvCaptureFromCAM(0);
    if(!capture){
        cerr << "ERROR: La captura a través de cámara no funciona" << endl;
        return -1;
    }

    modelEigen->load("bdEigen.xml");
    modelFisher->load("bdFisher.xml");
    cvNamedWindow("ventana", 1);

    if(capture) {
        for(;;) {
            IplImage* iplImg = cvQueryFrame( capture );
            frame = iplImg;
            if (frame.empty())
                break;
            if(iplImg->origin == IPL_ORIGIN_TL)
                frame.copyTo( frameCopy );
            else
                flip(frame, frameCopy, 0);
                
            detect(frameCopy, cascade);
            	
            if( waitKey( 10 ) >= 0 )
                goto _cleanup_;
        }
        waitKey(0);
        _cleanup_:
            cvReleaseCapture( &capture );
    }

    cvDestroyWindow("ventana");
    return 0;
}

/*
	Detección y recorte de una cara humana.
	Recibe una imagen en la cual intentaremos detectar una cara.
	Recibe un clasificador que usará para la detección.
	Devuelve 'true' si detecta una cara, 'false' si no.
*/
void detect(Mat& img, CascadeClassifier& cascade) {

    double t = 0;
    int result_recognition;
    vector<Rect> faces;
    Mat gray, smallImg(cvRound (img.rows), cvRound(img.cols), CV_8UC1);
    cvtColor(img, gray, CV_BGR2GRAY);
    resize(gray, smallImg, smallImg.size(), 0, 0, INTER_LINEAR);
    equalizeHist(smallImg, smallImg);
    t = (double)cvGetTickCount();
    cascade.detectMultiScale(gray, faces, 1.1, 1, 0|CV_HAAR_FIND_BIGGEST_OBJECT|CV_HAAR_DO_ROUGH_SEARCH, Size(250, 250));
    t = (double)cvGetTickCount() - t;
    printf("Tiempo en detectar : %g ms\n", t/((double)cvGetTickFrequency()*1000.));

    for(vector<Rect>::const_iterator r = faces.begin(); r != faces.end(); r++){
        Point center;
        int radius;
        center.x = cvRound(r->x + r->width*0.5);
        center.y = cvRound(r->y + r->height*0.5);
        radius = cvRound((r->width + r->height)*0.25);

        // Recortando la cara, despreciando el resto de la imagen.
        Point x1((center.x - radius + 20), (center.y - radius));
        Point x2((center.x + radius - 20), (center.y + radius));
        Rect myROI(x1.x, x1.y, (x2.x-x1.x), (x2.y-x1.y));
        Mat aux = img(myROI);
        Mat imagenRecortada = smallImg(myROI);
		
        result_recognition = recognize(imagenRecortada);
        
        if(result_recognition == -1){
            circle(img, center, radius, CV_RGB(255, 0, 0), 3, 8, 0);
    	} else if(result_recognition == 0) {
            circle(img, center, radius, CV_RGB(0, 0, 255), 3, 8, 0);
		}
	}
    cv::imshow("ventana", img);
}

/*
	Reconocimiento Facial.
	Recibe la imagen de una cara a reconocer.
	Devuelve un entero que representa a la persona reconocida en la Base de Datos.
	Devuelve -1 si no coincide con ninguna cara de la Base de Datos.
*/
int recognize(Mat& img){
    double t = 0;
    Mat testSample(280, 240, CV_32F);
    resize(img, testSample, testSample.size(), 0, 0, INTER_LINEAR);
    
    int resultadoFisher, resultadoEigen, resultado;
    resultado = -1;
    t = (double)cvGetTickCount();

    double umbralEigen, umbralFisher;
	modelEigen->predict(testSample, resultadoEigen, umbralEigen);
	modelFisher->predict(testSample, resultadoFisher, umbralFisher);
	
    cout << "Umbral Eigen : " << umbralEigen << endl;
    cout << "Umbral Fisher : " << umbralFisher << endl;
    cout << "Resultado Eigen : " << resultadoEigen << endl;
    cout << "Resultado Fisher : " << resultadoFisher << endl;
    
    t = (double)cvGetTickCount() - t;
    printf( "Tiempo en reconocer : %g ms\n", t/((double)cvGetTickFrequency()*1000.) );
    if(resultadoEigen != -1 || resultadoFisher != -1){
    	resultado = 0;
	}
    return resultado;
}

