#include <iostream>
#include <cstdio>
#include <random>
#include <map>
#include <vector>
#include "mpi.h"


int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int DIM_SIZE = 4;
    int NDIMS = 2;
    const int dims[2] = {DIM_SIZE, DIM_SIZE};
    const int periods[2] = {false, false};
    int coords[2];
    int rank;
    const int msg_len = 100;
    char data[msg_len+1];   // +1 added just to make output easier - its not needed
    data[msg_len] = '\0';

    std::vector<int> seq0 {1, 2, 3, 7, 11, 15};
    std::vector<int> seq1 {4, 8, 12, 13, 14, 15};

    MPI_Comm MATRIX;
    MPI_Cart_create(MPI_COMM_WORLD, NDIMS, dims, periods, false, &MATRIX);
    MPI_Comm_rank(MATRIX, &rank);
    MPI_Cart_coords(MATRIX, rank, 2, coords);

    MPI_Request req1, req2;

    bool used = rank == 0;

    for (const auto& i : seq0) {
        if (rank == i) {
            used = true;
            break;
        }
    }
    for (const auto& i : seq1) {
        if (rank == i || used) {
            used = true;
            break;
        }
    }

    if (used) {
        if (coords[0] == 3 && coords[1] == 3) {
            MPI_Irecv(data, msg_len/2, MPI_CHAR, 11, 0, MATRIX, &req1);
            MPI_Irecv(&data[msg_len/2], msg_len/2, MPI_CHAR, 14, 0, MATRIX, &req2);
        } else if (coords[0] != 0 || coords[1] != 0) {
            MPI_Irecv(data, msg_len/2, MPI_CHAR, MPI_ANY_SOURCE, MPI_ANY_TAG, MATRIX, &req1);
        }
    }
    MPI_Barrier(MATRIX);

    if (used) {
        if (coords[0] == 0 && coords[1] == 0) {
            std::srand(time(NULL));

            for (int i = 0; i < msg_len; ++i) {
                data[i] = 'a' + rand() % 26;
            }
            std::cout << "What sent: " << data << "\n" << std::endl;

            MPI_Irsend(data, msg_len / 2, MPI_CHAR, 1, 0, MATRIX, &req1);
            MPI_Irsend(&data[msg_len/2], msg_len / 2, MPI_CHAR, 4, 0, MATRIX, &req2);

            MPI_Wait(&req1,  MPI_STATUS_IGNORE);
            MPI_Wait(&req2,  MPI_STATUS_IGNORE);

            //std::cout << "Process 0 done" << std::endl;

        } else if (coords[0] == 3 && coords[1] == 3) {
            MPI_Wait(&req1,  MPI_STATUS_IGNORE);
            MPI_Wait(&req2,  MPI_STATUS_IGNORE);
            std::cout << "What received: " << data << "\n" << std::endl;

        } else {
            MPI_Wait(&req1,  MPI_STATUS_IGNORE);
            int nachste = 0;
            for (int i = 0; i < seq0.size(); ++i) {
                if (seq0[i] == rank) {
                    nachste = seq0[i+1];
                    break;
                }
                if (seq1[i] == rank) {
                    nachste = seq1[i+1];
                    break;
                }
            }
            MPI_Rsend(data, msg_len / 2, MPI_CHAR, nachste, 0, MATRIX);

            //std::cout << "Process " << rank << " done" << std::endl;
        }
    }

    MPI_Barrier(MATRIX);
    MPI_Finalize();
    return 0;
}
