using UnityEngine;
using System.Collections;
using UnityEngine.UI;
using System.IO;
using OpenCVForUnity;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System;


public class operation : MonoBehaviour
{

    // Use this for initialization
    public RawImage rawimage;
    Texture2D tex;
    WebCamTexture webcamTexture = null;
    Vector3 initPosition;

    private string name1 = "/storage/emulated/0/Pictures/Screenshots/gray.png";
    //private string name1="D:\\gray.png";
    //private string name1 = "/storage/emulated/0/DCIM/Screenshots/gray.png";

    public AudioClip DNA_voice, HEART_voice, BRAIN_voice;
    int zoomIn_DNA, zoomIn_HEART, zoomIn_BRAIN = 0;
    public GameObject DNA, HEART, BRAIN;


    DescriptorExtractor extractor=null;
    FeatureDetector detector=null;

    string _xml;
    float response = 0;

    public float svm(Mat imgMat)                                                            //SVM Classification
    {
        extractor = DescriptorExtractor.create(DescriptorExtractor.AKAZE);                  //Object creation for feature extraction

        detector = FeatureDetector.create(FeatureDetector.AKAZE);                           //Object creation for feature detection

        Scalar a;
        Mat test = new Mat(1, 1, CvType.CV_32FC1);
        Mat sampleMat = new Mat(250, 250, CvType.CV_8UC3);
        MatOfKeyPoint kp2 = new MatOfKeyPoint();
        Mat desc2 = new Mat();

        Imgproc.resize(imgMat, sampleMat, sampleMat.size());
        Imgproc.cvtColor(sampleMat, imgMat, Imgproc.COLOR_RGB2GRAY);
        detector.detect(imgMat, kp2);
        extractor.compute(imgMat, kp2, desc2);
        a = Core.mean(desc2);                                                              //Feature of complete image
        test.put(0, 0, a.val[0]);

        sampleMat.release();
        kp2.release();
        desc2.release();
        detector.Dispose();
        extractor.Dispose();

        SVM svm = SVM.create();                                                            //Creation of SVM classification object   
        _xml = Utils.getFilePath("svm_trained_2k.xml");                             //Loading the SVM classification model
        svm = SVM.load(_xml);

        response = svm.predict(test);                                                //Prediction                                                          
        test.release();
        svm.Dispose();
        return response;                                                                   //Returning obtained class label            
    }

    Mat readFile(String filename)
    {
        Texture2D texture2d = new Texture2D(webcamTexture.width, webcamTexture.height, TextureFormat.RGB24, false);
        byte[] fileData;
        fileData = File.ReadAllBytes(filename);
        texture2d.LoadImage(fileData);
        Mat img = new Mat(texture2d.height, texture2d.width, CvType.CV_8UC1);
        Utils.texture2DToMat(texture2d, img);
        return img;
    }
    void writeFile(String filename, Mat img)
    {
        Texture2D texture = new Texture2D(img.cols(), img.rows(), TextureFormat.RGBA32, false);
        Utils.matToTexture2D(img, texture);
        byte[] bytes = texture.EncodeToPNG();
        File.WriteAllBytes(filename, bytes);
    }

    void Start()
    {
        webcamTexture = new WebCamTexture();
        webcamTexture.Play();                                                               //Trigerring of Smartphone Camera
        GetComponent<AudioSource>().Stop();
    }

    // Update is called once per frame
    void Update()
    {
        tex = new Texture2D(webcamTexture.width, webcamTexture.height, TextureFormat.RGB24, false);
        tex.SetPixels(webcamTexture.GetPixels());
        tex.Apply();
        byte[] bytes = tex.EncodeToPNG();
        File.WriteAllBytes(name1, bytes);
    }
    void LateUpdate()
    {
        Mat f8UC3 = new Mat(tex.height, tex.width, CvType.CV_8UC3);
        Utils.texture2DToMat(tex, f8UC3);

        Imgproc.cvtColor(f8UC3, f8UC3, Imgproc.COLOR_RGB2YCrCb);

        Core.inRange(f8UC3, new Scalar(0, 140, 77), new Scalar(255, 176, 127), f8UC3);      //Image Segmentation
        Imgproc.medianBlur(f8UC3, f8UC3, 5);
        Mat element = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(2 * 5 + 1, 2 * 5 + 1), new Point(5, 5));
        Imgproc.dilate(f8UC3, f8UC3, element);

        float res = svm(f8UC3);

        rawimage.GetComponent<RawImage>().texture =tex;

        if (res == 1)
        {
            zoomIn();
        }
        else if (res == 2)
        {
            rotate();
        }

    }
    void zoomIn()
    {

        if (DNA.GetComponent<Renderer>().isVisible)
        {
            if (!GetComponent<AudioSource>().isPlaying)
            {
                GetComponent<AudioSource>().clip = DNA_voice;
                GetComponent<AudioSource>().Play();

            }
            else if (GetComponent<AudioSource>().clip == HEART_voice || GetComponent<AudioSource>().clip == BRAIN_voice)
            {
                GetComponent<AudioSource>().Stop();
                GetComponent<AudioSource>().clip = DNA_voice;
                GetComponent<AudioSource>().Play();
            }

            if (zoomIn_HEART > 0 || zoomIn_BRAIN > 0)
            {
                Camera.main.transform.position = initPosition;
                zoomIn_HEART = 0;
                zoomIn_BRAIN = 0;
            }

            if (zoomIn_DNA < 35)
            {
                float step = 90 * Time.deltaTime;
                Camera.main.transform.position = Vector3.MoveTowards(Camera.main.transform.position, DNA.transform.position, step);
                zoomIn_DNA++;

            }
            if (zoomIn_DNA >= 35)
            {
                zoomIn_DNA = 0;
                Camera.main.transform.position = initPosition;
            }


        }
        else if (HEART.GetComponent<Renderer>().isVisible)
        {
            if (!GetComponent<AudioSource>().isPlaying)
            {
                GetComponent<AudioSource>().clip = HEART_voice;
                GetComponent<AudioSource>().Play();

            }
            else if (GetComponent<AudioSource>().clip == BRAIN_voice || GetComponent<AudioSource>().clip == DNA_voice)
            {
                GetComponent<AudioSource>().Stop();
                GetComponent<AudioSource>().clip = HEART_voice;
                GetComponent<AudioSource>().Play();
            }
            if (zoomIn_DNA > 0 || zoomIn_BRAIN > 0)
            {
                Camera.main.transform.position = initPosition;
                zoomIn_DNA = 0;
                zoomIn_BRAIN = 0;
            }
            if (zoomIn_HEART < 35)
            {
                float step = 90 * Time.deltaTime;
                Camera.main.transform.position = Vector3.MoveTowards(Camera.main.transform.position, HEART.transform.position, step);
                zoomIn_HEART++;

            }
            if (zoomIn_HEART >= 35)
            {
                zoomIn_HEART = 0;
                Camera.main.transform.position = initPosition;
            }
        }

        else if (BRAIN.GetComponent<Renderer>().isVisible)
        {
            if (!GetComponent<AudioSource>().isPlaying)
            {
                GetComponent<AudioSource>().clip = BRAIN_voice;
                GetComponent<AudioSource>().Play();

            }
            else if (GetComponent<AudioSource>().clip == HEART_voice || GetComponent<AudioSource>().clip == DNA_voice)
            {
                GetComponent<AudioSource>().Stop();
                GetComponent<AudioSource>().clip = BRAIN_voice;
                GetComponent<AudioSource>().Play();
            }
            if (zoomIn_DNA > 0 || zoomIn_HEART > 0)
            {
                Camera.main.transform.position = initPosition;
                zoomIn_DNA = 0;
                zoomIn_HEART = 0;
            }
            if (zoomIn_BRAIN < 35)
            {
                float step = 90 * Time.deltaTime;
                Camera.main.transform.position = Vector3.MoveTowards(Camera.main.transform.position, BRAIN.transform.position, step);
                zoomIn_BRAIN++;
            }
            if (zoomIn_BRAIN >= 35)
            {
                zoomIn_BRAIN = 0;
                Camera.main.transform.position = initPosition;
            }
        }
    }

    void rotate()
    {
        if (DNA.GetComponent<Renderer>().isVisible)
        {
            if (!GetComponent<AudioSource>().isPlaying)
            {
                GetComponent<AudioSource>().clip = DNA_voice;
                GetComponent<AudioSource>().Play();
            }
            else if (GetComponent<AudioSource>().clip == HEART_voice || GetComponent<AudioSource>().clip == BRAIN_voice)
            {
                GetComponent<AudioSource>().Stop();
                GetComponent<AudioSource>().clip = DNA_voice;
                GetComponent<AudioSource>().Play();
            }
            DNA.transform.Rotate(new Vector3(0, 70, 0) * Time.deltaTime);
        }

        else if (HEART.GetComponent<Renderer>().isVisible)
        {
            if (!GetComponent<AudioSource>().isPlaying)
            {
                GetComponent<AudioSource>().clip = HEART_voice;
                GetComponent<AudioSource>().Play();
            }
            else if (GetComponent<AudioSource>().clip == BRAIN_voice || GetComponent<AudioSource>().clip == DNA_voice)
            {
                GetComponent<AudioSource>().Stop();
                GetComponent<AudioSource>().clip = HEART_voice;
                GetComponent<AudioSource>().Play();
            }
            HEART.transform.Rotate(new Vector3(0, 70, 0) * Time.deltaTime);
        }

        else if (BRAIN.GetComponent<Renderer>().isVisible)
        {
            if (!GetComponent<AudioSource>().isPlaying)
            {
                GetComponent<AudioSource>().clip = BRAIN_voice;
                GetComponent<AudioSource>().Play();
            }
            else if (GetComponent<AudioSource>().clip == HEART_voice || GetComponent<AudioSource>().clip == DNA_voice)
            {
                GetComponent<AudioSource>().Stop();
                GetComponent<AudioSource>().clip = BRAIN_voice;
                GetComponent<AudioSource>().Play();
            }
            BRAIN.transform.Rotate(new Vector3(0, 70, 0) * Time.deltaTime);
        }
    }
}