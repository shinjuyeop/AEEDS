/*
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "utils.h"
using namespace cv;
using namespace std;

// ORB settings
int ORB_MAX_KPTS = 1500;
float ORB_SCALE_FACTOR = 1.2;
int ORB_PYRAMID_LEVELS = 4;
float ORB_EDGE_THRESHOLD = 31.0;
int ORB_FIRST_PYRAMID_LEVEL = 0;
int ORB_WTA_K = 2;
int ORB_PATCH_SIZE = 31;
// Some image matching options
float MIN_H_ERROR = 2.50f; // Maximum error in pixels to accept an inlier
float DRATIO = 0.80f;

void main() { // standard main signature
	Mat img1, img1_32, img2, img2_32;
	string img_path1, img_path2, homography_path;
	double t1 = 0.0, t2 = 0.0;
	vector<KeyPoint> kpts1_orb, kpts2_orb;
	vector<Point2f> matches_orb, inliers_orb;
	vector<vector<DMatch>> dmatches_orb;
	Mat desc1_orb, desc2_orb;
	int nmatches_orb = 0, ninliers_orb = 0, noutliers_orb = 0;
	int nkpts1_orb = 0, nkpts2_orb = 0;
	float ratio_orb = 0.0;
	double torb = 0.0; // Create the L2 and L1 matchers
	Ptr<DescriptorMatcher> matcher_l2 = DescriptorMatcher::create("BruteForce");
	Ptr<DescriptorMatcher> matcher_l1 = DescriptorMatcher::create("BruteForce-Hamming");

	img1 = imread("C:/Users/shinj/Desktop/3-2/AEEDS/Data/test1.jpg", 0);
	img2 = imread("C:/Users/shinj/Desktop/3-2/AEEDS/Data/test2.jpg", 0);

	// Convert the images to float
	img1.convertTo(img1_32, CV_32F, 1.0 / 255.0, 0);
	img2.convertTo(img2_32, CV_32F, 1.0 / 255.0, 0);

	// Color images for results visualization
	Mat img1_rgb_orb = Mat(Size(img1.cols, img1.rows), CV_8UC3);
	Mat img2_rgb_orb = Mat(Size(img2.cols, img1.rows), CV_8UC3);
	Mat img_com_orb = Mat(Size(img1.cols * 2, img1.rows), CV_8UC3);

	// ORB Features
	Ptr<ORB> orb = ORB::create(ORB_MAX_KPTS, ORB_SCALE_FACTOR, ORB_PYRAMID_LEVELS,
		ORB_EDGE_THRESHOLD, ORB_FIRST_PYRAMID_LEVEL, ORB_WTA_K, ORB::HARRIS_SCORE,
		ORB_PATCH_SIZE);
	t1 = getTickCount();
	orb->detectAndCompute(img1, noArray(), kpts1_orb, desc1_orb, false);
	orb->detectAndCompute(img2, noArray(), kpts2_orb, desc2_orb, false);
	matcher_l1->knnMatch(desc1_orb, desc2_orb, dmatches_orb, 2);
	matches2points_nndr(kpts1_orb, kpts2_orb, dmatches_orb, matches_orb, DRATIO);
	compute_inliers_ransac(matches_orb, inliers_orb, MIN_H_ERROR, false);
	nkpts1_orb = kpts1_orb.size();
	nkpts2_orb = kpts2_orb.size();
	nmatches_orb = matches_orb.size() / 2;
	ninliers_orb = inliers_orb.size() / 2;
	noutliers_orb = nmatches_orb - ninliers_orb;
	ratio_orb = 100.0 * (float)(ninliers_orb) / (float)(nmatches_orb);
	t2 = cv::getTickCount();
	torb = 1000.0 * (t2 - t1) / cv::getTickFrequency();

	cvtColor(img1, img1_rgb_orb, cv::COLOR_GRAY2BGR);
	cvtColor(img2, img2_rgb_orb, cv::COLOR_GRAY2BGR);
	draw_keypoints(img1_rgb_orb, kpts1_orb);
	draw_keypoints(img2_rgb_orb, kpts2_orb);
	draw_inliers(img1_rgb_orb, img2_rgb_orb, img_com_orb, inliers_orb, 0);
	cv::imshow("ORB", img_com_orb);
	cv::waitKey(0);
	cout << "ORB Results" << endl;
	cout << "**************************************" << endl;
	cout << "Number of Keypoints Image 1: " << nkpts1_orb << endl;
	cout << "Number of Keypoints Image 2: " << nkpts2_orb << endl;
	cout << "Number of Matches: " << nmatches_orb << endl;
	cout << "Number of Inliers: " << ninliers_orb << endl;
	cout << "Number of Outliers: " << noutliers_orb << endl;
	cout << "Inliers Ratio: " << ratio_orb << endl;
	cout << "ORB Features Extraction Time (ms): " << torb << endl; cout << endl;

}
*/

// ORB Reference (left) vs Webcam (right)
// Width-fit to 640x460 with letterbox (top/bottom black bars if needed)
// Build: OpenCV 4.x, C++17+
// Needs: utils.h / utils.cpp  (matches2points_nndr, compute_inliers_ransac, draw_inliers, draw_keypoints)

#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include "utils.h"

using namespace cv;
using namespace std;

// width 기준으로 비율 유지 리사이즈 후,
// 세로가 부족하면 위/아래 검정 패딩(레터박스), 넘치면 중앙 크롭
static Mat resize_width_fit_letterbox(const Mat& src, const Size& target) {
    CV_Assert(!src.empty());
    const double scale = static_cast<double>(target.width) / src.cols;
    const int new_h = cvRound(src.rows * scale);

    Mat resized;
    resize(src, resized, Size(target.width, new_h), 0, 0, INTER_AREA);

    if (resized.rows == target.height) return resized;

    if (resized.rows < target.height) { // letterbox (top/bottom)
        Mat canvas(target.height, target.width, src.type(), Scalar::all(0));
        const int y = (target.height - resized.rows) / 2;
        resized.copyTo(canvas(Rect(0, y, resized.cols, resized.rows)));
        return canvas;
    }
    else { // (rare) crop center vertically
        const int y = (resized.rows - target.height) / 2;
        return resized(Rect(0, y, target.width, target.height)).clone();
    }
}

int main() {
    // ---- Parameters ----
    const int   ORB_MAX_KPTS = 1000; // 1500
    const float ORB_SCALE_FACTOR = 1.2f;
    const int   ORB_PYRAMID_LEVELS = 4;
    const float ORB_EDGE_THRESHOLD = 31.0f;
    const int   ORB_FIRST_LEVEL = 0;
    const int   ORB_WTA_K = 2;
    const int   ORB_PATCH_SIZE = 31;
    const float MIN_H_ERROR = 1.5f;   // px (2.0)
    const float DRATIO = 0.65f;  // Lowe ratio (0.75)

    // ---- Reference image ----
    const string REF_PATH = "C:/Users/shinj/Desktop/3-2/AEEDS/Data/feature_matching.jpg";
    Mat ref_gray = imread(REF_PATH, IMREAD_GRAYSCALE);
    if (ref_gray.empty()) { cerr << "Failed to load: " << REF_PATH << endl; return -1; }

    // ---- Webcam ----
    VideoCapture cap(0);
    if (!cap.isOpened()) { cerr << "Failed to open webcam\n"; return -1; }

    // Optionally force camera size (comment out if not needed)
    // cap.set(CAP_PROP_FRAME_WIDTH,  640);
    // cap.set(CAP_PROP_FRAME_HEIGHT, 460);

    // Grab first frame to determine target size (expected 640x460)
    Mat first;
    for (int i = 0; i < 30 && first.empty(); ++i) cap >> first;
    if (first.empty()) { cerr << "No webcam frame\n"; return -1; }

    Mat first_gray; cvtColor(first, first_gray, COLOR_BGR2GRAY);
    const Size targetSize(first_gray.cols, first_gray.rows); // e.g., 640x460
    cout << "Webcam size: " << targetSize.width << "x" << targetSize.height << endl;

    // ---- Prepare reference image: width-fit + letterbox to target size ----
    Mat ref_adj = resize_width_fit_letterbox(ref_gray, targetSize);
    Mat ref_rgb; cvtColor(ref_adj, ref_rgb, COLOR_GRAY2BGR);

    // ---- ORB + matcher ----
    Ptr<ORB> orb = ORB::create(ORB_MAX_KPTS, ORB_SCALE_FACTOR, ORB_PYRAMID_LEVELS,
        ORB_EDGE_THRESHOLD, ORB_FIRST_LEVEL, ORB_WTA_K,
        ORB::HARRIS_SCORE, ORB_PATCH_SIZE);
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");

    // Precompute reference features (once)
    vector<KeyPoint> kpts_ref;
    Mat desc_ref;
    orb->detectAndCompute(ref_adj, noArray(), kpts_ref, desc_ref);

    // ---- Loop ----
    Mat frame, frame_gray;
    while (true) {
        cap >> frame;
        if (frame.empty()) break;
        cvtColor(frame, frame_gray, COLOR_BGR2GRAY);

        // If webcam size changes, re-letterbox reference to match
        if (frame_gray.size() != ref_adj.size()) {
            ref_adj = resize_width_fit_letterbox(ref_gray, frame_gray.size());
            cvtColor(ref_adj, ref_rgb, COLOR_GRAY2BGR);
            orb->detectAndCompute(ref_adj, noArray(), kpts_ref, desc_ref);
        }

        // Compute frame features
        vector<KeyPoint> kpts_frm;
        Mat desc_frm;
        orb->detectAndCompute(frame_gray, noArray(), kpts_frm, desc_frm);

        // Guards
        if (desc_ref.empty() || desc_frm.empty()) {
            Mat canvas(Size(ref_rgb.cols + frame.cols, ref_rgb.rows), CV_8UC3, Scalar(0, 0, 0));
            ref_rgb.copyTo(canvas(Rect(0, 0, ref_rgb.cols, ref_rgb.rows)));
            Mat frame_rgb; cvtColor(frame_gray, frame_rgb, COLOR_GRAY2BGR);
            frame_rgb.copyTo(canvas(Rect(ref_rgb.cols, 0, frame_rgb.cols, frame_rgb.rows)));
            imshow("ORB Matching (Ref vs Webcam)", canvas);
            if (waitKey(1) == 27) break;
            continue;
        }

        // KNN (k=2) + keep only entries with 2 neighbors
        vector<vector<DMatch>> knn;
        matcher->knnMatch(desc_ref, desc_frm, knn, 2);
        knn.erase(remove_if(knn.begin(), knn.end(),
            [](const vector<DMatch>& v) { return v.size() < 2; }),
            knn.end());
        if (knn.empty()) { if (waitKey(1) == 27) break; continue; }

        // Lowe ratio test -> matched point coords
        vector<Point2f> matches_pts, inliers_pts;
        matches2points_nndr(kpts_ref, kpts_frm, knn, matches_pts, DRATIO);

        // RANSAC inliers (need >= 4 pairs)
        if (matches_pts.size() >= 8)
            compute_inliers_ransac(matches_pts, inliers_pts, MIN_H_ERROR, false);
        else
            inliers_pts.clear();

        // Compose side-by-side canvas (same height/width by construction)
        Mat frame_rgb; cvtColor(frame_gray, frame_rgb, COLOR_GRAY2BGR);
        Mat canvas(Size(ref_rgb.cols + frame_rgb.cols, ref_rgb.rows), CV_8UC3, Scalar(0, 0, 0));

        // (Optional) comment these if 원이 너무 많으면
        // draw_keypoints(ref_rgb, kpts_ref);
        // draw_keypoints(frame_rgb, kpts_frm);

        if (!inliers_pts.empty())
            draw_inliers(ref_rgb, frame_rgb, canvas, inliers_pts, 0);
        else {
            ref_rgb.copyTo(canvas(Rect(0, 0, ref_rgb.cols, ref_rgb.rows)));
            frame_rgb.copyTo(canvas(Rect(ref_rgb.cols, 0, frame_rgb.cols, frame_rgb.rows)));
        }

        putText(canvas,
            format("matches: %d  inliers: %d",
                (int)matches_pts.size() / 2, (int)inliers_pts.size() / 2),
            Point(12, 28), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 0), 2, LINE_AA);

        imshow("ORB Matching (Ref vs Webcam)", canvas);
        if (waitKey(1) == 27) break; // ESC
    }

    return 0;
}


