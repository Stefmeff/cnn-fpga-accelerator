#include "conv2d.h"
#include "kernels.h"   // conv2d() reference  (../../src)
#include "tensor.h"    // Tensor, padTensor   (../../utils)

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>


struct Case { int Cin, Cout, H, W; const char *name; };

/**
 * @brief compares the output of DUT with reference implementation
 * 
 * @param dut output from conv2d_hls()
 * @param ref output from conv2d() reference implementation
 * @param Cout number of output channels
 * @param H height of the output
 * @param W width of the output
 * @param name name of the test case
 * @return int number of mismatches
 */
static int compare(const float *dut, Tensor &ref, int Cout, int H, int W,
                   const char *name)
{
    const float ABS_TOL = 1e-3f;
    const float REL_TOL = 1e-3f;
    const float *r = ref.data[0][0];
    const int   n = Cout * H * W;

    //compare elementwise, count mismatches and report diff
    int fails = 0;
    float max_abs = 0.0f;
    for (int i = 0; i < n; i++) {
        float out_dut = dut[i];
        float out_ref = r[i];
        float diff = fabsf(out_dut - out_ref);
        if (diff > max_abs) max_abs = diff;
        float rel = diff / (fabsf(out_ref) + 1e-6f);
        if (diff > ABS_TOL && rel > REL_TOL) {
            if (fails < 10) {
                int oc = i / (H * W);
                int oy = (i / W) % H;
                int ox = i % W;
                printf("  MISMATCH [%2d][%2d][%2d]: dut=%.6f ref=%.6f (abs=%.6f)\n",
                       oc, oy, ox, out_dut, out_ref, diff);
            }
            fails++;
        }
    }
    printf("[%s] %s  (max abs err = %.6g, %d/%d mismatches)\n",
           fails == 0 ? "PASS" : "FAIL", name, max_abs, fails, n);
    return fails;
}

/**
 * @brief run test case: generate random input, run reference and DUT, compare results
 * 
 * @param c test case parameters
 * @return int number of mismatches
 */
static int run_case(const Case &c)
{
    printf("Case %s : Cin=%d Cout=%d %dx%d\n", c.name, c.Cin, c.Cout, c.H, c.W);

    //generate randomn input feature map
    Tensor X(c.Cin, c.H, c.W);
    X.randomize(-4.0f, 4.0f);

    //generate random weights
    Tensor *Wt = new Tensor[c.Cout]();          
    for (int oc = 0; oc < c.Cout; oc++) {
        Wt[oc].allocate(c.Cin, KSIZE, KSIZE);
        Wt[oc].randomize(-1.0f, 1.0f);
    }

    //generate random bias
    Tensor B(1, 1, c.Cout);
    B.randomize(-1.0f, 1.0f);

    //Add padding and run reference
    Tensor Zref(c.Cout, c.H, c.W);
    Tensor *Xp = padTensor(&X, 1);
    conv2d(Xp, Wt, &B, &Zref);
    delete Xp;

    
    const int xn = c.Cin * c.H * c.W;
    const int wn = c.Cout * c.Cin * 9;
    const int zn = c.Cout * c.H * c.W;

    float *x_flat = X.data[0][0];               // already contiguous
    float *b_flat = B.data[0][0];
    float *w_flat = new float[wn];
    for (int oc = 0; oc < c.Cout; oc++)
        memcpy(w_flat + oc * c.Cin * 9, Wt[oc].data[0][0], sizeof(float) * c.Cin * 9);
    float *z_dut = new float[zn]();

    (void)xn;

    //run dut
    conv2d_hls(x_flat, w_flat, b_flat, z_dut, c.Cin, c.Cout, c.H, c.W);

    //compare results
    int fails = compare(z_dut, Zref, c.Cout, c.H, c.W, c.name);

    delete[] w_flat;
    delete[] z_dut;
    delete[] Wt;
    return fails;
}

int main()
{
    srand(1234);

    //Test cases representative of the layers
    Case cases[] = {
        {  3, 16, 32, 32, "L0  3->16 @32" },
        { 16, 16, 32, 32, "L2 16->16 @32" },
        { 16, 32, 16, 16, "L9 16->32 @16" },
        { 32, 32, 16, 16, "L11 32->32 @16" },
        { 32, 64,  8,  8, "L16 32->64 @8" },
        { 64, 64,  8,  8, "L18 64->64 @8" },
    };

    //run different cases
    int total_fails = 0;
    for (const Case &c : cases)
        total_fails += run_case(c);

        
    printf("\n==== %s : %d total mismatches ====\n",
           total_fails == 0 ? "ALL PASS" : "FAILURES", total_fails);
    return total_fails == 0 ? 0 : 1;
}
