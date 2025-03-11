//
// Created by ryan on 3/5/25.
//

#ifndef RTPDSP_H
#define RTPDSP_H

#include <cstdio>
#include <cstdint>
#include <iostream>
#include <cuda.h>
#include <cufftdx.hpp>
#include <block_io.hpp>


#define CUDA_ERR_CHK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}





#define RTP_PAYLOAD_LEN 1024
#define THREADS_PER_BLOCK 32
#define ELEMENTS_PER_THREAD RTP_PAYLOAD_LEN/THREADS_PER_BLOCK


using namespace cufftdx;

// R2C and C2R specific properties describing data layout and execution mode for
// the requested transform.
using real_fft_options = RealFFTOptions<complex_layout::natural, real_mode::normal>;

using FFT = decltype( Size<RTP_PAYLOAD_LEN>()
                      + Precision<float>()
                      + Type<fft_type::r2c>()
                      + Direction<fft_direction::forward>()
                      + ElementsPerThread<ELEMENTS_PER_THREAD>()
                      + SM<750>()
                      + real_fft_options()
                      + Block());


using complex_type = typename FFT::value_type;
using real_type    = typename complex_type::value_type; // i.e. float

typedef struct {
    uint8_t header[12];
    uint8_t payload[RTP_PAYLOAD_LEN];
    uint32_t ssrc;
} rtp_packet;

typedef struct {
    int16_t payload[RTP_PAYLOAD_LEN];
    complex_type spectrum[FFT::output_length];
    uint32_t ssrc;
} pktspectrum;

__global__ void kernel_dsp(rtp_packet *in, pktspectrum *out, int n);


#endif //RTPDSP_H
