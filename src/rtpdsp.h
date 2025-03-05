//
// Created by ryan on 3/5/25.
//

#ifndef RTPDSP_H
#define RTPDSP_H

#include <cstdio>
#include <cstdint>
#include <cuda.h>



#define min(a,b) ((a) < (b) ? (a) : (b))


#define CUDA_ERR_CHK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}


typedef struct {
    uint8_t header[12];
    uint8_t payload[160];
    uint32_t ssrc;
} rtp_packet;

#endif //RTPDSP_H
