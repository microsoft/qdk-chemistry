/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE.txt in the project root for
 * license information.
 */

/* OpenBLAS smoke test for the Windows ARM64 CI job.
 *
 * Guards two failure modes that are silent: the library builds, links, and
 * computes dgemm correctly in both, so without this probe a regression surfaces
 * only as the whole test suite timing out.
 *
 *   1. getarch selecting the GENERIC kernel set instead of an ARMV8 one. The
 *      caller checks the reported corename for this.
 *   2. dgetrf looping forever above the getf2 crossover, which is what GENERIC's
 *      zero-valued blocking parameters produce.
 *
 * The Fortran entry points are declared here directly rather than through a
 * vendor header: no header is needed for the f77 interface, and this is the same
 * interface blaspp itself calls.
 */
#include <stdio.h>

extern char *openblas_get_config(void);
extern char *openblas_get_corename(void);
extern int openblas_get_num_procs(void);
extern int openblas_get_num_threads(void);

extern void dtrsm_(char *, char *, char *, char *, int *, int *, double *, double *, int *, double *,
                   int *);
extern void dgetrf_(int *, int *, double *, int *, int *, int *);
extern void dgetf2_(int *, int *, double *, int *, int *, int *);
extern void dpotrf_(char *, int *, double *, int *, int *);

#define MAXN 64

static double m[MAXN * MAXN], b[MAXN * MAXN];
static int ipiv[MAXN];

static void fill_spd(int n) {
    for (int i = 0; i < n * n; ++i) m[i] = 0.0;
    for (int i = 0; i < n; ++i) m[i * n + i] = 2.0;
}

int main(void) {
    int info = 0;
    char L = 'L', N = 'N', U = 'U';
    double one = 1.0;

    printf("config   = %s\n", openblas_get_config());
    fflush(stdout);
    printf("corename = %s\n", openblas_get_corename());
    fflush(stdout);
    printf("num_procs=%d num_threads=%d\n", openblas_get_num_procs(), openblas_get_num_threads());
    fflush(stdout);

    for (int i = 0; i < MAXN * MAXN; ++i) b[i] = 1.0;

    int n8 = 8;
    fill_spd(n8);
    printf("before dtrsm n=8\n");
    fflush(stdout);
    dtrsm_(&L, &L, &N, &N, &n8, &n8, &one, m, &n8, b, &n8);
    printf("after  dtrsm\n");
    fflush(stdout);

    /* Straddle the getf2 crossover: the small sizes return through getf2, and
     * the larger ones exercise the blocked path that hangs under GENERIC. */
    int sizes[5] = {1, 2, 4, 24, 64};
    for (int s = 0; s < 5; ++s) {
        int n = sizes[s];

        fill_spd(n);
        printf("before dgetf2 n=%d\n", n);
        fflush(stdout);
        info = 0;
        dgetf2_(&n, &n, m, &n, ipiv, &info);
        printf("after  dgetf2 n=%d info=%d\n", n, info);
        fflush(stdout);

        fill_spd(n);
        printf("before dgetrf n=%d\n", n);
        fflush(stdout);
        info = 0;
        dgetrf_(&n, &n, m, &n, ipiv, &info);
        printf("after  dgetrf n=%d info=%d\n", n, info);
        fflush(stdout);

        fill_spd(n);
        printf("before dpotrf n=%d\n", n);
        fflush(stdout);
        info = 0;
        dpotrf_(&U, &n, m, &n, &info);
        printf("after  dpotrf n=%d info=%d\n", n, info);
        fflush(stdout);
    }

    printf("ALL OPENBLAS CALLS COMPLETED\n");
    fflush(stdout);
    return 0;
}
