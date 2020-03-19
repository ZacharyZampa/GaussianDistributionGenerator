/*
 * Copyright (C) 2020 zampaze@miamioh.edu
 */


// includes
#include <omp.h>
#include <iostream>
#include <string>
#include <cstdlib>
#include <cmath>
#include <limits>

// namespaces
using namespace std;

// preprocessor macros
#define RUNLENGTH 20000000
#define N 100

// globals
struct drand48_data state;
// #pragma omp threadprivate(state)


const double epsilon = std::numeric_limits<double>::min();
const double pi2 = 2.0 * 3.14159265358979323846;

/**
 * Generate Gaussian random numbers with mean mu and standard deviation sigma
 */
void doMarsaglia(double *ave, double *stdDev, double mu, double sigma) {
    double sum = 0;
    double sumSquares = 0;
    int numObs = 0;

#pragma omp parallel for reduction(+: sum, sumSquares, numObs)
    for (int i = 0; i < RUNLENGTH; i += 2) {
        double v1, v2, u1, u2, g1, g2, w;
        do {
            drand48_r(&state, &u1); drand48_r(&state, &u2);
            v1 = u1 * 2 - 1;
            v2 = u2 * 2 - 1;
            w = v1 * v1 + v2 * v2;
        } while (w >= 1.0 || w == 0.0);
        double sq = sqrt(-2.0 * log(w) / w);
        g1 = v1 * sq;
        g2 = v2 * sq;

        // just g1
        numObs++;
        sum += g1 * sigma + mu;
        sumSquares += (g1 * sigma + mu) * (g1 * sigma + mu);

        // just g2
        numObs++;
        sum += g2 * sigma + mu;
        sumSquares += (g2 * sigma + mu) * (g2 * sigma + mu);
    }

    // update what the average and standard deviation is
    *ave = sum / numObs;
    *stdDev = sqrt((sumSquares / numObs) - (*ave * *ave));
}

void initRNG(size_t rngSeed) {
#pragma omp parallel shared(rngSeed)
    {
        int k = omp_get_thread_num();

        // Set the initial see and state
        srand48_r(rngSeed + 23 * k, &state);
    }
}

/*
 * Launchpad of program
 */
int main(int argc, char** argv) {
    // grab command line input
    if (argc < 2) {
        std::cout << "Must provide single unsigned int" << std::endl;
        std::exit(0);
    }
    size_t rngSeed = std::stoul(argv[1]);

    // Initialize the random number generator
    initRNG(rngSeed);

    double ave = 0.0;
    double stdDev = 0.0;

    for (int i = 0; i < N; ++i) {
        doMarsaglia(&ave, &stdDev, 42.0, 5.0);
        std::cout << "Average = " << ave << " Standard Deviation = " <<
                stdDev << std::endl;
    }


    return 0;
}
