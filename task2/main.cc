#include <iostream>
#include <cstdlib>
#include <algorithm>
#include <vector>
#include <set>
#include <iomanip>
#include <ctime>
#include <cmath>
#include <cstring>
#include <string>
#include <mpi.h>
#include <fstream>
#include <signal.h>
#include <mpi-ext.h>


//#define log

using namespace std;

#define TOKILL 10

int currRank, nWorkingProcs, nProcs;
string filename;
unsigned proc_failure = 0;
MPI_Comm main_comm;

const double precision = 0.00001;
int size;


// static void err_handler(MPI_Comm *pcomm, int *perr, ...) {
//     proc_failure = 1;
//     int err = *perr;
//     char errstr[MPI_MAX_ERROR_STRING];
//     int size, nf, len;
//     int *ranks_gc, *ranks_gf;
//     MPI_Group group_f, group_c;
//
//     MPI_Comm_size(main_comm, &size);
//     MPIX_Comm_failure_ack(main_comm);
//     MPIX_Comm_failure_get_acked(main_comm, &group_f);
//     MPI_Group_size(group_f, &nf);
//     MPI_Error_string(err, errstr, &len);
//     printf("\nRank %d / %d failed: %s. %d process(es) found dead\n", currRank, size, errstr, nf);
//
//     // shrink to a new comm w/o dead process
//     MPIX_Comm_shrink(main_comm, &main_comm);
//     ranks_gf= (int*)malloc(nf* sizeof(int));
//     ranks_gc= (int*)malloc(nf* sizeof(int));
//     MPI_Comm_group(main_comm, &group_c);
//     for(int i= 0; i< nf; i++)ranks_gf[i] = i;
//     MPI_Group_translate_ranks(group_f, nf, ranks_gf,group_c, ranks_gc);
//     MPI_Comm_rank(main_comm, &currRank);
//     filename = "chp/" + to_string(currRank) + ".bp";
//     free(ranks_gf); free(ranks_gc);
// }


static void err_handler(MPI_Comm* pcomm, int* perr, ...) {
    MPI_Comm comm= *pcomm;
    int err = *perr;
    char errstr[MPI_MAX_ERROR_STRING];
    int i, rank, size, nf, len, eclass;
    MPI_Group group_c, group_f;
    int *ranks_gc, *ranks_gf;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    MPIX_Comm_failure_ack(comm);
    MPIX_Comm_failure_get_acked(comm, &group_f);
    MPI_Group_size(group_f, &nf);
    MPI_Error_string(err, errstr, &len);
    //printf("Rank %d / %d: Notified of error %s. %d found dead: { ",rank, size, errstr, nf);
    ranks_gf= (int*)malloc(nf* sizeof(int));
    ranks_gc= (int*)malloc(nf* sizeof(int));
    MPI_Comm_group(comm, &group_c);
    for(i= 0; i< nf; i++)ranks_gf[i] = i;
    MPI_Group_translate_ranks(group_f, nf, ranks_gf,group_c, ranks_gc);
    //for(i= 0; i< nf; i++)printf("%d ", ranks_gc[i]);
    //printf("}\n");
    free(ranks_gf); free(ranks_gc);
}




void write_backup(double* pdata, size_t length, const std::string& file_path) {
    ofstream os(file_path, ios::binary | ios::out | ios::trunc);
    if ( !os.is_open() ){
        std::cout << "Failed to open " << file_path << std::endl;
    } else {
        os.write(reinterpret_cast<const char*>(pdata), std::streamsize(length*sizeof(double)));
        os.close();
    }
}

void read_backup(double* pdata, size_t length, const std::string& file_path) {
    ifstream is(file_path, ios::binary | ios::in);
    if ( !is.is_open() ) {
        std::cout << "Failed to open " << file_path << std::endl;
    } else {
        is.read(reinterpret_cast<char*>(pdata), std::streamsize(length*sizeof(double)));
        is.close();
    }
}

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

int toTrapezeMPI (double* matrix, int currRank, int nWorkingProcs) {
    //first, divide matrix with nWorkingProcs strips, each for separate task

    #ifdef log
    cerr << "Sharing initial data started" << endl;
    #endif

    // "demo"-killing one of the processes (something like this may occur anywhere, but we fail one of them here)
    if (currRank == TOKILL) {
        raise(SIGKILL);
    }

    int stripHeight = (!currRank) ? (size/nWorkingProcs + size%nWorkingProcs) : (size/nWorkingProcs);
    double* strip = new double[size * stripHeight];
    //broadcast them
    if (!currRank) {
        memcpy(strip, matrix, stripHeight*size * sizeof(double));
        for (int i = 1; i < nWorkingProcs; ++i) {
            MPI_Send (&matrix[size*size/nWorkingProcs * i + size%nWorkingProcs], size*size/nWorkingProcs, MPI_DOUBLE, i, 0, main_comm);
        }
    } else if (currRank < nWorkingProcs) {
        MPI_Recv (strip, stripHeight*size, MPI_DOUBLE, 0, 0, main_comm, MPI_STATUS_IGNORE);

        #ifdef log
        for (int i = 0; i < stripHeight; ++i) {
            for (int j = 0; j < size; ++j) {
                cout << strip [i*size + j] << " ";
            }
            cout << endl;
        }
        #endif
    }
    write_backup(strip, size * stripHeight, filename);

    #ifdef log
    cerr << "Sharing initial data finished" << endl;
    #endif

    set<int> nullRows;          //to keep track of empty rows and not check them every time
    //every process received its working strip

    for (int k = 0; k < size - 1; ++k) {
        // checkpoint Charlie

        read_backup(strip, size * stripHeight, filename);

        #ifdef log
        cerr << "Step " << k << " started" << endl;
        #endif
        int maxIndex;
        double* maxValues = new double[nWorkingProcs];
        double* maxValues_tmp = new double[nWorkingProcs];
        double* candidates = new double [size * nWorkingProcs];
        double* candidates_tmp = new double [size * nWorkingProcs];
        fill (candidates, candidates + size*nWorkingProcs, 0);
        fill (candidates_tmp, candidates_tmp + size*nWorkingProcs, 0);

        #ifdef log
        cerr << "Calling maxInColumn" << endl;
        #endif

        if (currRank < nWorkingProcs) {
            maxValues_tmp[currRank] = maxInColumn (strip, stripHeight, k, maxIndex);
        }

        #ifdef log
        cerr << "maxInColumn done, starting memcpy" << currRank << endl;
        #endif

        if (maxIndex >= 0 && currRank < nWorkingProcs) {
            memcpy (&candidates_tmp[size * currRank], &strip[size * maxIndex], size*sizeof(double));
        }

        #ifdef log
        cerr << "memcpy finished, allreduce started" << currRank << endl;
        #endif


        MPI_Allreduce (candidates_tmp, candidates, size * nWorkingProcs, MPI_DOUBLE, MPI_SUM, main_comm);
        MPI_Allreduce (maxValues_tmp, maxValues, nWorkingProcs, MPI_DOUBLE, MPI_SUM, main_comm);


        #ifdef log
        cerr << "Allreduce finished" << currRank << endl;
        #endif

        delete[] candidates_tmp;

        //now every process has a copy of array of max elements and of array of candidate rows
        //Now, choose the required index of the above mentioned arrays
        //maxIndex == -1 means that all rows are null in the strip

        if (maxIndex >= 0 && currRank < nWorkingProcs) {
            int chosen = 0;
            for (int i = 0; i < nWorkingProcs; ++i) {
                if (abs (maxValues[chosen]) < abs (maxValues[i])) {
                    chosen = i;
                }
            }
            for (int i = 0; i < stripHeight; ++i) {
                if (chosen == currRank && maxIndex == i) {
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

        write_backup(strip, size * stripHeight, filename);

        #ifdef log
        cerr << "Step " << k << " finished" << endl;
        #endif
    }

    #ifdef log
    cerr << "elimination finished " << currRank << endl;
    #endif

    int retval = 0;
    int empties = nullRows.size();
    MPI_Reduce (&empties, &retval, 1, MPI_INT, MPI_SUM, 0, main_comm);

    delete[] strip;
    return retval;
}


void fillMatrix (double* matrix, int size) {
    for (int i = 0; i < size*size; ++i) {
        matrix[i] = rand()%1000;
    }
}


int main (int argc, char *argv[]) {
    MPI_Errhandler errh;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nProcs);
    MPI_Comm_rank(MPI_COMM_WORLD, &currRank);
    nWorkingProcs = 1;
    while (nWorkingProcs <= nProcs){
        nWorkingProcs <<= 1;
    }
    nWorkingProcs >>= 1;
    sscanf(argv[1], "%i", &size);
    srand(time(NULL));

    main_comm = MPI_COMM_WORLD;

    // Error handler setup
    MPI_Comm_create_errhandler(err_handler, &errh);
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, errh);
    MPI_Barrier(MPI_COMM_WORLD);

    // backup filename
    filename = "chp/" + to_string(currRank) + ".bp";

    //To make testing easier and the testing results clearer, assume that matrix is N*N;
    //But the algorithm can be easily modified for non-square matrix

    double* matrix = new double [size * size];
    double time = 0;

    for (int i = 0; i < 1; ++i) {
        if (!currRank) {
            fillMatrix (matrix, size);
        }

        MPI_Barrier (main_comm);
        double loc_time = MPI_Wtime ();

        #ifdef log
        cerr << "Started toTrapezeMPI" << endl;
        #endif

        int empty = toTrapezeMPI (matrix, currRank, nWorkingProcs);

        #ifdef log
        cerr << "Finished toTrapezeMPI " << currRank << endl;
        #endif

        MPI_Barrier (main_comm);
        time += MPI_Wtime () - loc_time;

        //no need to reassemble the matrix - just compute null rows in every block
        /*
        if (!currRank) {
            cout << "currRank is " << size - empty << endl;
        }*/
    }

    delete[] matrix;
    //time /= 3;

    if (!currRank) {
        cout << "Size: " << size << "   Time: " << fixed << setprecision(8) << time << endl;
    }
    MPI_Finalize();
    return 0;
}
