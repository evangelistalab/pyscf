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

void compute_T2_block(double *t2, double *ei, double *ej, double *ea, double *eb, double flow_param, int ni, int nj, int na, int nb)
{
#pragma omp parallel
{
        int i,j,a,b;
        double* pt2;
#pragma omp for schedule(dynamic, 2)
        for (i = 0; i < ni; i++) {
                for (j = 0; j < nj; j++) {
                        for (a = 0; a < na; a++) {
                                for (b = 0; b < nb; b++) {
                                        double denom = ei[i] + ej[j] - ea[a] - eb[b];
                                        pt2 = t2 + i * nj * na * nb + j * na * nb + a * nb + b;;
                                        *pt2 *= regularized_denominator(denom,flow_param);
                                }
                        }
                }
        }
}
}

void compute_T1(double *t1, double *ei, double *ea, double flow_param, int ni, int na)
{
#pragma omp parallel
{
        int i, a;
        double *pt1;
#pragma omp for schedule(dynamic, 2)
        for (i = 0; i < ni; i++) {
                for (a = 0; a < na; a++) {
                        double denom = ei[i] - ea[a];
                        pt1 = t1 + i * na + a;
                        *pt1 *= regularized_denominator(denom,flow_param);
                }
        }
}
}