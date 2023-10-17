// Imagine++ project
// Project:  Panorama
// Author:   Pascal Monasse
// Date:     2013/10/08

#include <Imagine/Graphics.h>
#include <Imagine/Images.h>
#include <Imagine/LinAlg.h>
#include <vector>
#include <sstream>
using namespace Imagine;
using namespace std;

// Record clicks in two images, until right button click
int getClicks(Window w1, Window w2,
               vector<IntPoint2>& pts1, vector<IntPoint2>& pts2) {
    // ------------- TODO/A completer ----------
    Window win;
    int sub_win;
    IntPoint2 cliked_pt;
    bool done = false;
    int nb_pts = 0;
    Color c;
    char r, g, b;
    Event ev;

    while(!done){
        getEvent(-1, ev);

        if((ev.type== EVT_KEY_ON) && (ev.key == KEY_RETURN)){
            done = true;
            return 0;
        }
        else if (ev.type == EVT_BUT_ON){
            //cout<<"button = "<<button<< endl;
            cliked_pt = ev.pix;


            if((nb_pts % 2) == 0){
                r = rand() % 256; g = rand() % 256; b = rand() % 256;
                c = Color(r, g, b);
            }
            setActiveWindow(ev.win);
            fillCircle(cliked_pt, 4, c, false);

            if(ev.win == w1){
                pts1.push_back(cliked_pt);
                cout << "Active window: 1" << endl;
            }
            else if(ev.win == w2){
                pts2.push_back(cliked_pt);
                cout << "Active window: 2" << endl;
            }
            cout << "Coordinates of the new point " << nb_pts << ": " << cliked_pt << endl;
            cout << endl;

            nb_pts += 1;
        }
    }
    return 0;
}


// Return homography compatible with point matches
Matrix<float> getHomography(const vector<IntPoint2>& pts1,
                            const vector<IntPoint2>& pts2) {
    size_t n = min(pts1.size(), pts2.size());
    if(n<4) {
        cout << "Not enough correspondences: " << n << endl;
        return Matrix<float>::Identity(3);
    }
    Matrix<double> A(2*n,8);
    Vector<double> B(2*n);
    // ------------- TODO/A completer ----------

    double xi, xi_p, yi, yi_p;
    for (int i=0; i<n; i++){
        xi = pts1[i][0];
        yi = pts1[i][1];

        xi_p = pts2[i][0];
        yi_p = pts2[i][1];

        A(2*i,0)= xi;
        A(2*i,1)= yi;
        A(2*i,2)= 1;
        A(2*i,3)= 0;
        A(2*i,4)= 0;
        A(2*i,5)= 0;
        A(2*i,6)= -xi_p*xi;
        A(2*i,7)= -xi_p*yi;

        A(2*i+1,0)=0;
        A(2*i+1,1)=0;
        A(2*i+1,2)=0;
        A(2*i+1,3)=xi;
        A(2*i+1,4)=yi;
        A(2*i+1,5)=1;
        A(2*i+1,6)=-yi_p*xi;
        A(2*i+1,7)=-yi_p*yi;

        B[2*i] = xi_p;
        B[2*i + 1] = yi_p;
    }


    B = linSolve(A, B);
    Matrix<float> H(3, 3);
    H(0,0)=B[0]; H(0,1)=B[1]; H(0,2)=B[2];
    H(1,0)=B[3]; H(1,1)=B[4]; H(1,2)=B[5];
    H(2,0)=B[6]; H(2,1)=B[7]; H(2,2)=1;

    // Sanity check
    for(size_t i=0; i<n; i++) {
        float v1[]={(float)pts1[i].x(), (float)pts1[i].y(), 1.0f};
        float v2[]={(float)pts2[i].x(), (float)pts2[i].y(), 1.0f};
        Vector<float> x1(v1,3);
        Vector<float> x2(v2,3);
        x1 = H*x1;
        cout << x1[1]*x2[2]-x1[2]*x2[1] << ' '
             << x1[2]*x2[0]-x1[0]*x2[2] << ' '
             << x1[0]*x2[1]-x1[1]*x2[0] << endl;
    }
    return H;
}

// Grow rectangle of corners (x0,y0) and (x1,y1) to include (x,y)
void growTo(float& x0, float& y0, float& x1, float& y1, float x, float y) {
    if(x<x0) x0=x;
    if(x>x1) x1=x;
    if(y<y0) y0=y;
    if(y>y1) y1=y;
}

// Panorama construction
void panorama(const Image<Color,2>& I1, const Image<Color,2>& I2,
              Matrix<float> H) {
    // Initialize variables to define the boundaries of the resulting panorama image
    float x0=0, y0=0, x1=I2.width(), y1=I2.height();

    // Loop through four corners of I2 and transform them using the homography matrix H
    Vector<float> v(3);
    v[0]=0; v[1]=0; v[2]=1;
    v=H*v; v/=v[2];
    growTo(x0, y0, x1, y1, v[0], v[1]);

    v[0]=I2.width(); v[1]=0; v[2]=1;
    v=H*v; v/=v[2];
    growTo(x0, y0, x1, y1, v[0], v[1]);

    v[0]=I2.width(); v[1]=I2.height(); v[2]=1;
    v=H*v; v/=v[2];
    growTo(x0, y0, x1, y1, v[0], v[1]);

    v[0]=0; v[1]=I2.height(); v[2]=1;
    v=H*v; v/=v[2];
    growTo(x0, y0, x1, y1, v[0], v[1]);

    // Output the boundaries of the panorama image
    cout << "x0 x1 y0 y1=" << x0 << ' ' << x1 << ' ' << y0 << ' ' << y1 << endl;

    // Create the output panorama image
    Image<Color> I(int(x1-x0), int(y1-y0));
    setActiveWindow( openWindow(I.width(), I.height()) );
    I.fill(WHITE);

    // ------------- TODO/A completer ----------

    // Calculate the inverse of the homography matrix
    Matrix<float> H_inv = inverse(H);
    Vector<float> p(3), p_prec(3);

    // Loop through the pixels of the output image
    for(int x = 0; x < I.width(); x++){
        for(int y = 0; y < I.height(); y++){
            // Transform the current pixel's position to the system of I2
            p[0] = x + x0; p[1] = y + y0; p[2] = 1;

            // Check if the transformed pixel is inside I2
            bool inside_I2 = (p[0] > 0 && p[1] > 0 && p[0] < I2.width() && p[1] < I2.height());

            if(inside_I2){
                // Copy the color from I2 to the output image I
                I(x, y) = I2(p[0], p[1]);
            }
            p_prec = p;

            // Transform the current pixel's position to the system of I1 using the inverse homography
            p = H_inv * p; p /= p[2];

            // Check if the transformed pixel is inside I1
            bool inside_I1 = (p[0] > 0 && p[1] > 0 && p[0] < I1.width() && p[1] < I1.height());

            if(inside_I1){
                // Use interpolation to get a color from I1 and copy it to the output image I
                I(x, y) = I1.interpolate(p[0], p[1]);
            }

            // Check if the pixel is inside both I1 and I2 (overlap)
            bool overlap = (inside_I1 && inside_I2);

            if(overlap){
                // Average the colors of the corresponding points
                I(x, y)[0] = (I2(p_prec[0], p[1])[0] + I1.interpolate(p[0], p[1]).r()) / 2;
                I(x, y)[1] = (I2(p_prec[0], p[1])[1] + I1.interpolate(p[0], p[1]).g()) / 2;
                I(x, y)[2] = (I2(p_prec[0], p[1])[2] + I1.interpolate(p[0], p[1]).b()) / 2;
            }
        }
    }

    // Display the resulting panorama image
    display(I, 0, 0);

}



// Main function
int main(int argc, char* argv[]) {
    const char* s1 = argc>1? argv[1]: srcPath("image0006.jpg");
    const char* s2 = argc>2? argv[2]: srcPath("image0007.jpg");

    // Load and display images
    Image<Color> I1, I2;
    if( ! load(I1, s1) ||
        ! load(I2, s2) ) {
        cerr<< "Unable to load the images" << endl;
        return 1;
    }
    Window w1 = openWindow(I1.width(), I1.height(), s1);
    display(I1,0,0);
    Window w2 = openWindow(I2.width(), I2.height(), s2);
    setActiveWindow(w2);
    display(I2,0,0);

    // Get user's clicks in images
    vector<IntPoint2> pts1, pts2;
    getClicks(w1, w2, pts1, pts2);

    vector<IntPoint2>::const_iterator it;
    cout << "pts1="<<endl;
    for(it=pts1.begin(); it != pts1.end(); it++)
        cout << *it << endl;
    cout << "pts2="<<endl;
    for(it=pts2.begin(); it != pts2.end(); it++)
        cout << *it << endl;

    // Compute homography
    Matrix<float> H = getHomography(pts1, pts2);
    cout << "H=" << H/H(2,2);

    // Apply homography
    panorama(I1, I2, H);

    endGraphics();
    return 0;
}
