#include <iostream>
#include <cstdlib>
#include <algorithm>
#include <vector>
#include <set>
#include <iomanip>
#include <ctime>
#include <cmath>
#include <cstring>
#include <mpi.h>

//#define log

using namespace std;
const double precision = 0.00001;
int size;

bool isNull (double* matrix, int row) {
    for (int i = 0; i < size; ++i) {
        if (abs(matrix[size * row + i]) > precision) {
            return false;
        }
    }
    return true;
}

int findRank (double* matrix, int size) {
    int ans = size;
    for (int j = 0; j<size; ++j) {
        if (isNull (matrix, j)) {
            --ans;
        } else {
            break;
        }
    }
    return ans;
}

double maxInColumn (double* strip, int stripHeight, int k, int& maxIndex) {
    double max = 0;
    maxIndex = -1;

    for (int i = 0; i < stripHeight; ++i) {
        if (abs(strip[i*size + k]) > abs(max)) {
            bool suits = true;
            for (int j = 0; j < k; ++j) {
                if (abs (strip[i*size + j]) >= precision){
                    suits = false;
                    break;
                }
            }
            if (suits) {
                max = strip[i*size + k];
                maxIndex = i;
            }
        }
    }
    return max;
}

int toTrapezeMPI (double* matrix, int rank, int nProcs) {
    //first, divide matrix with nProcs strips, each for separate task

    #ifdef log
    cerr << "Sharing initial data started" << endl;
    #endif

    int stripHeight = (!rank) ? (size/nProcs + size%nProcs) : (size/nProcs);
    double* strip = new double[size * stripHeight];
    //broadcast them
    if (!rank) {
        memcpy(strip, matrix, stripHeight*size * sizeof(double));
        for (int i = 1; i < nProcs; ++i) {
            MPI_Send (&matrix[size*size/nProcs * i + size%nProcs], size*size/nProcs, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
        }
    } else {
        MPI_Recv (strip, stripHeight*size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        #ifdef log
        for (int i = 0; i < stripHeight; ++i) {
            for (int j = 0; j < size; ++j) {
                cout << strip [i*size + j] << " ";
            }
            cout << endl;
        }
        #endif
    }

    #ifdef log
    cerr << "Sharing initial data finished" << endl;
    #endif

    set<int> nullRows;          //to keep track of empty rows and not check them every time
    //every process received its working strip

    for (int k = 0; k < size - 1; ++k) {
        #ifdef log
        cerr << "Step " << k << " started" << endl;
        #endif
        int maxIndex;

        double* maxValues = new double[nProcs];
        double* maxValues_tmp = new double[nProcs];
        double* candidates = new double [size * nProcs];
        double* candidates_tmp = new double [size * nProcs];
        fill (candidates, candidates + size*nProcs, 0);
        fill (candidates_tmp, candidates_tmp + size*nProcs, 0);

        #ifdef log
        cerr << "Calling maxInColumn" << endl;
        #endif

        maxValues_tmp[rank] = maxInColumn (strip, stripHeight, k, maxIndex);

        #ifdef log
        cerr << "maxInColumn done, starting memcpy" << rank << endl;
        #endif

        if (maxIndex >= 0) {
            memcpy (&candidates_tmp[size * rank], &strip[size * maxIndex], size*sizeof(double));
        }

        #ifdef log
        cerr << "memcpy finished, allreduce started" << rank << endl;
        #endif

        //MPI_Barrier (MPI_COMM_WORLD);
        MPI_Allreduce (candidates_tmp, candidates, size * nProcs, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce (maxValues_tmp, maxValues, nProcs, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        #ifdef log
        cerr << "Allreduce finished" << rank << endl;
        #endif

        delete[] candidates_tmp;

        //now every process has a copy of array of max elements and of array of candidate rows
        //Now, choose the required index of the above mentioned arrays
        //maxIndex == -1 means that all rows are null in the strip

        if (maxIndex >= 0) {
            int chosen = 0;
            for (int i = 0; i < nProcs; ++i) {
                if (abs (maxValues[chosen]) < abs (maxValues[i])) {
                    chosen = i;
                }
            }
            for (int i = 0; i < stripHeight; ++i) {
                if (chosen == rank && maxIndex == i) {
                    continue;
                }
                if (!nullRows.count (i)) {
                    bool allNull = true;
                    double tmp = strip[size*i + k] / candidates[chosen*size + k];
                    for (int j = 0; j < size; ++j) {
                        strip[size*i + j] -= tmp * candidates[chosen*size + j];
                        if ( abs(strip[size*i + j]) > precision) {
                            allNull = false;
                        }
                    }
                    if (allNull) {
                        nullRows.insert (i);
                    }

                }

            }
        }
        delete[] candidates;
        delete[] maxValues_tmp;
        delete[] maxValues;

        #ifdef log
        cerr << "Step " << k << " finished" << endl;
        #endif
    }

    #ifdef log
    cerr << "elimination finished " << rank << endl;
    #endif

    int retval = 0;
    int empties = nullRows.size();
    MPI_Reduce (&empties, &retval, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    delete[] strip;
    return retval;
}


void fillMatrix (double* matrix, int size) {
    for (int i = 0; i < size*size; ++i) {
        matrix[i] = rand()%1000;
    }
    //used for testing the accuracy
    /*
    for (int i = 0; i < size; ++i) {
        matrix[(size-5) * size + i] = matrix[(size-3)*size + i];
        matrix[(size-7) * size + i] = matrix[(size-3)*size + i];
        matrix[(size-11) * size + i] = matrix[(size-3)*size + i];
    }*/

}

int main (int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int rank, nProcs;
    MPI_Comm_size(MPI_COMM_WORLD, &nProcs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    sscanf(argv[1], "%i", &size);
    srand(time(NULL));

    //To make testing easier and the testing results clearer, assume that matrix is N*N;
    //But the algorithm can be easily modified for non-square matrix

    double* matrix = new double [size * size];
    double time = 0;

    for (int i = 0; i < 1; ++i) {
        if (!rank) {
            fillMatrix (matrix, size);
        }

        MPI_Barrier (MPI_COMM_WORLD);
        double loc_time = MPI_Wtime ();

        #ifdef log
        cerr << "Started toTrapezeMPI" << endl;
        #endif

        int empty = toTrapezeMPI (matrix, rank, nProcs);

        #ifdef log
        cerr << "Finished toTrapezeMPI " << rank << endl;
        #endif

        MPI_Barrier (MPI_COMM_WORLD);
        time += MPI_Wtime () - loc_time;

        //no need to reassemble the matrix - just compute null rows in every block
        /*
        if (!rank) {
            cout << "Rank is " << size - empty << endl;
        }*/
    }

    delete[] matrix;
    time /= 3;

    if (!rank) {
        cout << "Size: " << size << "   Time: " << fixed << setprecision(8) << time << endl;
    }
    MPI_Finalize();
    return 0;
}
