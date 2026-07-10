#include "conv2d.h"
#include "kernels.h"   // conv2d() reference  (../../src)
#include "tensor.h"    // Tensor, padTensor   (../../utils)
#include "cnn.h"       // CNN, ConvLayer, genSeqConvWeightsWS  (../../src)

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

using namespace ml;


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
    // Tolerances sized for fixed-point quantization (max observed ~0.13 @ Cin=64),
    // not exactness. Loose enough that quantization passes, tight enough that a real
    // C/RTL divergence (errors of many units) still fails -> makes cosim meaningful.
    const float ABS_TOL = 0.3f;
    const float REL_TOL = 5e-2f;
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

    // Build a one-conv-layer CNN. Its ctor allocates W[Cout] (each [Cin][3][3])
    // and B(1,1,Cout) in exactly the layout genSeqConvWeightsWS/genSeqConvBias
    // expect, so we reuse the real weight-ordering path instead of duplicating it.
    std::vector<CNN_layer_struct> layer_list = { ConvLayer(c.Cin, c.Cout, c.H, c.W, 3, 1) };
    CNN net(layer_list);
    Tensor *Wt = net.layers[0].W;   // owned by net (freed in ~CNN)
    Tensor *B  = net.layers[0].B;   // owned by net (freed in ~CNN)

    //generate random input, weights and bias (single source for ref + DUT)
    Tensor X(c.Cin, c.H, c.W);
    X.randomize(-4.0f, 4.0f);
    for (int oc = 0; oc < c.Cout; oc++)
        Wt[oc].randomize(-1.0f, 1.0f);
    B->randomize(-1.0f, 1.0f);

    //Add padding and run reference
    Tensor Zref(c.Cout, c.H, c.W);
    Tensor *Xp = padTensor(&X, 1);
    conv2d(Xp, Wt, B, &Zref);
    delete Xp;

    //conv2d_core fuses ReLU on its output write -> apply the same to the reference
    float *zref_flat = Zref.data[0][0];
    for (int i = 0; i < c.Cout * c.H * c.W; i++)
        if (zref_flat[i] < 0.0f) zref_flat[i] = 0.0f;

    //Pack weights/bias into the accelerator's WS stream order (real functions)
    int wtot = 0, btot = 0;
    FLOAT *w_ws = net.genSeqConvWeightsWS(&wtot);
    FLOAT *b_ws = net.genSeqConvBias(&btot);

    const int zn = c.Cout * c.H * c.W;
    float *x_flat = X.data[0][0];               // already contiguous
    act_t *z_dut  = new act_t[zn]();            // raw fixed-point out (bit-exact in cosim)

    //run dut
    conv2d_hls(x_flat, w_ws, b_ws, z_dut, c.Cin, c.Cout, c.H, c.W);

    //convert fixed-point output to float for the reference comparison
    float *z_f = new float[zn];
    for (int i = 0; i < zn; i++) z_f[i] = (float)z_dut[i];

    //compare results
    int fails = compare(z_f, Zref, c.Cout, c.H, c.W, c.name);

    delete[] w_ws;
    delete[] b_ws;
    delete[] z_f;
    delete[] z_dut;
    // net dtor frees Wt/B
    return fails;
}

int main()
{
    srand(1234);

    Case cases[] = {
        // {  3, 16, 32, 32, "L0  3->16 @32" },
        // { 16, 16, 32, 32, "L2 16->16 @32" },
        { 16, 32, 16, 16, "L9 16->32 @16" },
        // { 32, 32, 16, 16, "L11 32->32 @16" },
        // { 32, 64,  8,  8, "L16 32->64 @8" },
        // { 64, 64,  8,  8, "L18 64->64 @8" },
    };

    //run different cases
    int total_fails = 0;
    for (const Case &c : cases)
        total_fails += run_case(c);

        
    printf("\n==== %s : %d total mismatches ====\n",
           total_fails == 0 ? "ALL PASS" : "FAILURES", total_fails);
    return total_fails == 0 ? 0 : 1;
}
