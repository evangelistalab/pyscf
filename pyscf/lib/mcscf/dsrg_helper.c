#include <stdlib.h>
#include <math.h>
// #include <omp.h>
#include "config.h"
#include "vhf/fblas.h"
#include "np_helper/np_helper.h"

#define MACHEPS 1e-9
#define TAYLOR_THRES 1e-3

double taylor_exp(double z)
{
        int n = (int)(0.5 * (15.0 / TAYLOR_THRES + 1)) + 1;
        if (n > 0) {
                double value = z;
                double tmp = z;
                for (int x = 0; x < n-1; x++) {
                        tmp *= -1.0 * z * z / (x + 2);
                        value += tmp;
                }
                return value;
        } else {
                return 0.0;
        }
}

double regularized_denominator(double x, double s)
{
        double z = sqrt(s) * x;
        if (fabs(z) <= MACHEPS) {
                return taylor_exp(z) * sqrt(s);
        } else {
                return (1. - exp(-s * x * x)) / x;
        }
}

void compute_T2_block(double *t2, double *ep, double *eq, double *er, double *es, double flow_param, int np, int nq, int nr, int ns)
{
#pragma omp parallel
{
        int p, q, r, s;
#pragma omp for schedule(dynamic, 50)
        for (p = 0; p < np; p++) {
                for (q = 0; q < nq; q++) {
                        for (r = 0; r < nr; r++) {
                                for (s = 0; s < ns; s++) {
                                        float denom = ep[p] + eq[q] - er[r] - es[s];
                                        *t2 *= regularized_denominator(denom,flow_param);
                                        t2++;
                                }
                        }
                }
        }
}
}