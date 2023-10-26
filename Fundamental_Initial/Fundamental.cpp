// Imagine++ project
// Project:  Fundamental
// Author:   Pascal Monasse

#include "./Imagine/Features.h"
#include <Imagine/Graphics.h>
#include <Imagine/LinAlg.h>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <random>
#include <algorithm>

using namespace Imagine;
using namespace std;

static const float BETA = 0.01f; // Probability of failure

struct Match {
    float x1, y1, x2, y2;
};

// Display SIFT points and fill vector of point correspondences
void algoSIFT(Image<Color,2> I1, Image<Color,2> I2,
              vector<Match>& matches) {
    // Find interest points
    SIFTDetector D;
    D.setFirstOctave(-1);
    Array<SIFTDetector::Feature> feats1 = D.run(I1);
    drawFeatures(feats1, Coords<2>(0,0));
    cout << "Im1: " << feats1.size() << flush;
    Array<SIFTDetector::Feature> feats2 = D.run(I2);
    drawFeatures(feats2, Coords<2>(I1.width(),0));
    cout << " Im2: " << feats2.size() << flush;

    const double MAX_DISTANCE = 100.0*100.0;
    for(size_t i=0; i < feats1.size(); i++) {
        SIFTDetector::Feature f1=feats1[i];
        for(size_t j=0; j < feats2.size(); j++) {
            double d = squaredDist(f1.desc, feats2[j].desc);
            if(d < MAX_DISTANCE) {
                Match m;
                m.x1 = f1.pos.x();
                m.y1 = f1.pos.y();
                m.x2 = feats2[j].pos.x();
                m.y2 = feats2[j].pos.y();
                matches.push_back(m);
            }
        }
    }
}




vector<Match> sample8Matches(const vector<Match>& matches) {
    vector<Match> sampledMatches;
    const int numMatches = matches.size();

    // If we have less than 8 matches, we return what we have
    if (numMatches < 8) {
        return matches;
    }

    // Create an array of indices from 0 to numMatches - 1
    vector<int> indices(numMatches);
    for (int i = 0; i < numMatches; i++) {
        indices[i] = i;
    }

    // Shuffle the indices randomly
    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(indices.begin(), indices.end(), gen);

    // Take the first 8 shuffled indices
    for (int i = 0; i < 8; i++) {
        sampledMatches.push_back(matches[indices[i]]);
    }

    return sampledMatches;
}

void NormalizePoints(vector<Match>&subsetMatches, vector<Match>&normalizedPoints){

    // Step 1: Compute the centroid of the points
    float cx1 = 0.0f;
    float cy1 = 0.0f;
    float cx2 = 0.0f;
    float cy2 = 0.0f;
    int nbPoints = subsetMatches.size();

    for (const Match& point : subsetMatches) {
        cx1 += point.x1;
        cy1 += point.y1;
        cx2 += point.x2;
        cy2 += point.y2;
    }

    cx1 /= nbPoints;
    cy1 /= nbPoints;
    cx2 /= nbPoints;
    cy2 /= nbPoints;

    // Step 2: Compute the average distance from the centroid
    float averageDistance1 = 0.0f;
    float averageDistance2 = 0.0f;


    for (const Match& point : subsetMatches) {
        float dx1 = point.x1 - cx1;
        float dy1 = point.y1 - cy1;
        float dx2 = point.x2 - cx2;
        float dy2 = point.y2 - cy2;
        averageDistance1 += std::sqrt(dx1 * dx1 + dy1 * dy1);
        averageDistance2 += std::sqrt(dx2 * dx2 + dy2 * dy2);
    }

    averageDistance1 /= nbPoints;
    averageDistance2 /= nbPoints;


    // Step 3: Normalize the points
    normalizedPoints.clear();

    for (const Match& point : subsetMatches) {
        Match normalizedPoint;
        normalizedPoint.x1 = (point.x1 - cx1) / averageDistance1;
        normalizedPoint.y1 = (point.y1 - cy1) / averageDistance1;
        normalizedPoint.x2 = (point.x2 - cx2) / averageDistance2;
        normalizedPoint.y2 = (point.y2 - cy2) / averageDistance2;
        normalizedPoints.push_back(normalizedPoint);
    }
}


FMatrix<float, 3, 3> estimateF(const vector<Match>& subsetMatches, bool useRefinement=false) {
    // Create a normalization matrix N
    FMatrix<float, 3, 3> N;
    N(0, 0) = 1e-3;
    N(1, 1) = 1e-3;
    N(2, 2) = 1;

    FloatPoint3 point1, point2;
    FMatrix<float, 9, 9> A;

    // Determine the number of equations to use
    int numEquations = 8 + int(useRefinement);
    for (int i = 0; i < numEquations; i++) {
        // Extract points from matches
        point1[0] = subsetMatches[i].x1;
        point1[1] = subsetMatches[i].y1;
        point1[2] = 1;
        point2[0] = subsetMatches[i].x2;
        point2[1] = subsetMatches[i].y2;
        point2[2] = 1;

        // Normalize the points
        point1 = N * point1;
        point2 = N * point2;

        // Extract coordinates for the linear system
        float x1 = point1[0];
        float y1 = point1[1];
        float x2 = point2[0];
        float y2 = point2[1];

        // Fill the coefficient matrix A
        A(i, 0) = x1 * x2;
        A(i, 1) = x1 * y2;
        A(i, 2) = x1;
        A(i, 3) = y1 * x2;
        A(i, 4) = y1 * y2;
        A(i, 5) = y1;
        A(i, 6) = x2;
        A(i, 7) = y2;
        A(i, 8) = 1;
    }

    // If no refinement is required, add an extra equation
    if (!useRefinement) {
        for (int col = 0; col < 9; col++) {
            A(8, col) = 0;
        }
    }

    // Perform SVD on matrix A
    FMatrix<float, 9, 9> U, Vt;
    FVector<float, 9> E;

    svd(A, U, E, Vt);

    // Extract the fundamental matrix F
    FVector<float, 9> lastEigenVector = Vt.getRow(8);
    FMatrix<float, 3, 3> F;

    for (int row = 0; row < 3; row++) {
        for (int col = 0; col < 3; col++) {
            F(row, col) = lastEigenVector[3 * row + col];
        }
    }

    // Enforce the rank-2 constraint
    FVector<float, 3> E2;
    FMatrix<float, 3, 3> U2, Vt2;

    svd(F, U2, E2, Vt2);
    E2[2] = 0;
    F = U2 * Diagonal(E2) * Vt2;

    // Denormalize F
    F = transpose(N) * F * N;

    return F;
}

int updateNiter(int bestInliersSize,  int matchesSize){
    float den= log(1- pow( (float)bestInliersSize/(float)matchesSize, 8));

    if(den < 0){
        return ceil(log(BETA)/den);
    }
    else{
        return -1;
    }
}



// RANSAC algorithm to compute F from point matches (8-point algorithm)
// Parameter matches is filtered to keep only inliers as output.
FMatrix<float,3,3> computeF(vector<Match>& matches) {
    const float distMax = 1.5f; // Pixel error for inlier/outlier discrimination
    int Niter=100000; // Adjusted dynamically
    FMatrix<float,3,3> bestF;
    vector<int> bestInliers;
    // --------------- TODO ------------
    // DO NOT FORGET NORMALIZATION OF POINTS
    int iter = 0;
    while (iter<Niter){        // Randomly select 8 matches
        vector<Match> subsetMatches;
        // Fill subsetMatches with 8 randomly selected matches from 'matches'
        subsetMatches = sample8Matches(matches);
        // Estimate F from the 8 matches
        FMatrix<float, 3, 3> estimatedF = estimateF(subsetMatches);

        // Count inliers for the current estimated F
        vector<int> inliers;
        FloatPoint3 pt_1, pt_2;
        FVector<float,3> pt;

        FMatrix<float,3,3> F_T = transpose(estimatedF);
        for (size_t i = 0; i < matches.size(); i++) {
            // Calculate the epipolar error for each match using estimatedF
            // You may want to use the normalized points for this calculation
            pt_1[0] = matches[i].x1;
            pt_1[1] = matches[i].y1;
            pt_1[2] = 1;
            pt_2[0] = matches[i].x2;
            pt_2[1] = matches[i].y2;
            pt_2[2] = 1;

            pt[0] = matches[i].x1;
            pt[1] = matches[i].y1;
            pt[2] = 1;

            float error = abs(((F_T*pt)[0] * matches[i].x2) + ((F_T*pt)[1] * matches[i].y2) + (F_T*pt)[2]) / sqrt(pow((F_T*pt)[0],2) + pow((F_T*pt)[1],2));

            // If the error is less than distMax, we consider it an inlier
            if(error < distMax){
                inliers.push_back(i);
            }
        }


        // If the number of inliers is greater than the best found so far, update bestF and bestInliers
        if (inliers.size() > bestInliers.size()) {
            bestF = estimatedF;
            bestInliers = inliers;

            // Adjust Niter dynamically based on the number of inliers
            int Niter_tmp;
            Niter_tmp = updateNiter(bestInliers.size(), matches.size());
            if (Niter_tmp != -1){
                Niter = Niter_tmp;
            }
        }
        iter += 1;
    }

    // Updating matches with inliers only
    vector<Match> all=matches;
    matches.clear();
    for(size_t i=0; i<bestInliers.size(); i++)
        matches.push_back(all[bestInliers[i]]);
    return bestF;
}

void displayEpipolar(Image<Color> I1, Image<Color> I2,
                     const FMatrix<float,3,3>& F) {
    while(true) {
        int x,y;
        if(getMouse(x,y) == 3)
            break;
        // --------------- TODO ------------
        Color c = Color(rand() % 256, rand() % 256, rand() % 256);

        fillCircle(x, y, 4, c, false);

        DoublePoint3 point;
        point[0] = x; point[1] = y; point[2] = 1;

        FVector<float, 3> line_epip;

        IntPoint2 pt_left, pt_right;

        int I1_width = I1.width();

        // right
        if(x > I1_width){
            point[0] -= I1_width;
            line_epip = F*point;

            pt_left[0] = 0; pt_right[0] = I1_width;
        }
        // left
        else if (x <= I1_width){
            line_epip = transpose(F)*point;

            pt_left[0] = I1_width; pt_right[0] = I1_width*2;
        }
        pt_left[1] = -(line_epip[2])/line_epip[1];
        pt_right[1] = -(line_epip[2] + line_epip[0]*I1_width)/line_epip[1];

        // shows associated epipolar line in other image
        drawLine(pt_left, pt_right, c, 3);
    }
}
int main(int argc, char* argv[])
{
    srand((unsigned int)time(0));

    const char* s1 = argc>1? argv[1]: srcPath("im1.jpg");
    const char* s2 = argc>2? argv[2]: srcPath("im2.jpg");

    // Load and display images
    Image<Color,2> I1, I2;
    if( ! load(I1, s1) ||
        ! load(I2, s2) ) {
        cerr<< "Unable to load images" << endl;
        return 1;
    }
    int w = I1.width();
    openWindow(2*w, I1.height());
    display(I1,0,0);
    display(I2,w,0);

    vector<Match> matches;
    algoSIFT(I1, I2, matches);
    const int n = (int)matches.size();
    cout << " matches: " << n << endl;
    drawString(100,20,std::to_string(n)+ " matches",RED);
    click();
    
    FMatrix<float,3,3> F = computeF(matches);
    cout << "F="<< endl << F;

    // Redisplay with matches
    display(I1,0,0);
    display(I2,w,0);
    for(size_t i=0; i<matches.size(); i++) {
        Color c(rand()%256,rand()%256,rand()%256);
        fillCircle(matches[i].x1+0, matches[i].y1, 2, c);
        fillCircle(matches[i].x2+w, matches[i].y2, 2, c);        
    }
    drawString(100, 20, to_string(matches.size())+"/"+to_string(n)+" inliers", RED);
    click();

    // Redisplay without SIFT points
    display(I1,0,0);
    display(I2,w,0);
    displayEpipolar(I1, I2, F);

    endGraphics();
    return 0;
}
