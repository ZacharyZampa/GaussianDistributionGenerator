/*
 * Copyright (C) 2020 zampaze@miamioh.edu
 */

/*
 * File:   box-muller.cpp
 * Author: zackzampa
 *
 * Created on March 16, 2020, 4:15 PM
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
#pragma omp threadprivate(state)

const double epsilon = std::numeric_limits<double>::min(); 
const double pi2 = 2.0 * 3.14159265358979323846;


/**
 * Generate Gaussian random numbers with mean mu and standard deviation sigma
*/
void doBoxMuller(double *ave, double *stdDev, double mu, double sigma) {
    double sum = 0;
    double sumSquares = 0;
    int numObs = 0;
    
# pragma omp parallel for reduction(+: sum, sumSquares, numObs)
    for (int i = 0; i < RUNLENGTH; i += 2) {
        double u1, u2, g0, g1;
            do {
                drand48_r(&state, &u1); 
                drand48_r(&state, &u2);
            } while (u1 <= epsilon);

            g0 = sqrt(-2.0 * log(u1)) * cos(pi2 * u2);
            g1 = sqrt(-2.0 * log(u1)) * sin(pi2 * u2);

            // just g0
            numObs++; 
            sum += g0 * sigma + mu;
            sumSquares += (g0 * sigma + mu) * (g0 * sigma + mu);
            
            // just g1
            numObs++; 
            sum += g1 * sigma + mu;
            sumSquares += (g1 * sigma + mu) * (g1 * sigma + mu);
    }
    
        *ave = sum / numObs;  
        *stdDev = sqrt((sumSquares / numObs) - (*ave * *ave));
}

/**
 * Initialize the random number generation state.
 * Each thread gets a unique seed based on the original
 * provided seed.
 */
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
        doBoxMuller(&ave, &stdDev, 42.0, 5.0);
        std::cout << "Average = " << ave << " Standard Deviation = " << 
        stdDev << std::endl;
    }
    
    
    return 0;
}

