#ifndef MMSEQS_MPI_H
#define MMSEQS_MPI_H

#ifdef HAVE_MPI
#include <mpi.h>
#endif

class MMseqsMPI {
public:
    static const int MASTER = 0;

    static bool active;
    static int rank;
    static int numProc;

    static void init(int argc, const char **argv);
    static inline bool isMaster() {
#ifdef HAVE_MPI
        return rank == MASTER;
#else
        return true;
#endif
    };
};

#include <cstdlib>
#ifdef OPENMP
#include <omp.h>
#define MMSEQS_IN_PARALLEL() (omp_in_parallel() != 0)
#else
#define MMSEQS_IN_PARALLEL() false
#endif

// if we are in an error case, do not call MPI_Finalize, it might still be in a Barrier
#ifdef HAVE_MPI
#include <cstdio>
#include <cstdlib>
#define EXIT(exitCode) do {                  \
    int __status = (exitCode);               \
    if(MMseqsMPI::active && __status == 0) { \
        MPI_Finalize();                      \
        MMseqsMPI::active = false;           \
    }                                        \
    fflush(NULL);                            \
    if (MMSEQS_IN_PARALLEL()) {              \
        _Exit(__status);                     \
    }                                        \
    exit(__status);                          \
} while(0)
#endif

#endif
