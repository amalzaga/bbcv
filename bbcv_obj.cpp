/************************************************************************************************
*
*  bbcv_obj.cpp - Beaglebone webcam object detection using opencv libraries.
*
*                 Example of using: - HSV Threshold (inRange)
*                                   - findContours
*                                   - bound Rectangle
*                                   - minEnclosureCircle
*                                   - Text drawing
*
*                 compile with: g++ bbcv_obj.cpp -o bbcv_obj `pkg-config --cflags --libs opencv`
*
*
*   Written by: Amador Alzaga - amadoralzaga.com
*
*************************************************************************************************/

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <stdio.h>

using namespace cv;
using namespace std;

RNG rng(12345);   //for random number generation


//function that draws a cross signaling center of contours
void centerCross(Mat img, Point centro, int lenght, Scalar color)
{
   line(img, Point(centro.x - lenght,centro.y), Point(centro.x + lenght,centro.y), color, 1, 8);
   line(img, Point(centro.x,centro.y - lenght), Point(centro.x,centro.y + lenght), color, 1, 8);
}


int main(int argc,char* argv[])
  {

   VideoCapture cap(0); // instance of VideoCapture object, opening camera 1

   if(!cap.isOpened())
    {
    cout << "Error opening camera.\n";
    return(-1);
    }

   bool asucess;
   asucess = cap.set(CV_CAP_PROP_FRAME_WIDTH,352);       //set frame size. if size is too large is possible
   asucess = cap.set(CV_CAP_PROP_FRAME_HEIGHT,288);      // to find some selected timeout errors at runtime

   double dWidth = cap.get(CV_CAP_PROP_FRAME_WIDTH);
   double dHeight = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
   
   cout << "Frame size: " << dWidth << " x " << dHeight << endl; // outputting frame size at console


   Mat frame,imgHSV,imgout,imggray,imgtmp;
 //  Mat imgLines = Mat::zeros(frame.size(),CV_8UC3);
   bool bsucess;

   vector<vector<Point> > contours;          //Data structures needed for contours storage
   vector<Vec4i> hierarchy;

   int fontface = FONT_HERSHEY_PLAIN;      // text font params
   double fontscale = 1;
   int thickness = 1;


   namedWindow("control",0);               // drawing control window
   moveWindow("control",780,20);

   int iLowH = 0;                         // and initializing HSV threshold params
   int iHighH = 18;
   int iLowS = 80;
   int iHighS = 255;
   int iLowV = 0;
   int iHighV = 255;

   cvCreateTrackbar("LowH","control",&iLowH,179);    //which can be modified at runtime
   cvCreateTrackbar("HighH","control",&iHighH,179);
   cvCreateTrackbar("LowS","control",&iLowS,255);    // using trackbars on control window
   cvCreateTrackbar("HighS","control",&iHighS,255);
   cvCreateTrackbar("LowV","control",&iLowV,255);
   cvCreateTrackbar("HighV","control",&iHighV,255);   

   namedWindow("camera");                            // draw windows for original
   moveWindow("camera",20,20);
   namedWindow("processed");                         // and processed images
   moveWindow("processed",400,20);
   
   
   while(1){                                       // main loop
   bsucess = cap.read(frame);
   if(frame.empty())
   {
   cout << "unable to read frame" << endl;
   return(-1);
   }
   cvtColor(frame,imgHSV,CV_BGR2HSV);           // convert image to HSV color space.

   inRange(imgHSV,Scalar(iLowH,iLowS,iLowV),Scalar(iHighH,iHighS,iHighV),imgout); //Threshold image
   
                                                                          // to obtain color recognition
   erode(imgout,imgout,getStructuringElement(MORPH_ELLIPSE,Size(3,3)));
   dilate(imgout,imgout,getStructuringElement(MORPH_ELLIPSE,Size(3,3))); // apply morphological transformations
   dilate(imgout,imgout,getStructuringElement(MORPH_ELLIPSE,Size(3,3)));  
   erode(imgout,imgout,getStructuringElement(MORPH_ELLIPSE,Size(3,3)));  // to clean image
    
   cvtColor(imgout,imgtmp,CV_GRAY2BGR);    // convert to grayscale cause findContours needs this format
   
   blur(imgout,imggray,Size(5,5));         // apply blur filter to reduce noise

   findContours(imggray,contours,hierarchy,CV_RETR_TREE,CV_CHAIN_APPROX_SIMPLE,Point(0,0));
   
   vector<vector<Point> > contours_poly(contours.size());
   vector<Rect> boundRect(contours.size());
   vector<Point2f> center(contours.size());
   vector<float> radius(contours.size());  

   for(int i =0;i<contours.size();i++)       // once found contours. Approx, find bound rectangle
   {
    approxPolyDP( Mat(contours[i]),contours_poly[i],3,true);   
    boundRect[i] = boundingRect( Mat(contours_poly[i]));
    minEnclosingCircle((Mat)contours_poly[i],center[i],radius[i]);  // and min enclosing circle
    Scalar color = Scalar(rng.uniform(0,255),rng.uniform(0,255),rng.uniform(0,255)); 
    drawContours(imgtmp,contours,i,color,1,8,hierarchy,0,Point());
    color = Scalar(0,255,200);    
    rectangle(frame,boundRect[i].tl(),boundRect[i].br(),color,1,8,0); // and draws it on original image
    color = Scalar(255,0,0);
    circle(frame,center[i],(int)radius[i],color,1,8,0);
    color = Scalar(0,255,0);
    centerCross(frame,center[i],5,color);

    if((int)radius[i]>20)  // adds text coordinates to objects contained in big circles
    {
     char texto[16];
     sprintf(texto,"x=%d",(int)center[i].x);
     putText(frame,texto,Point(boundRect[i].br().x,boundRect[i].br().y - 12),fontface,fontscale,color,thickness,8);
     sprintf(texto,"y=%d",(int)center[i].y);
     putText(frame,texto,boundRect[i].br(),fontface,fontscale,color,thickness,8);


    }

   }

   
  
   imshow("camera",frame);
   imshow("processed",imgtmp);


   if(waitKey(20) == 27) break;
   }

   return(0);
   }
