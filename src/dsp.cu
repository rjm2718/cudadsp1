
#include "rtpdsp.h"

// // cuFFTDx header
// #include <cufftdx.hpp>
//
// using namespace cufftdx;

// FFT description:
// A 128-point single precision complex-to-complex forward FFT description
// using FFT = decltype( Size<RTP_PAYLOAD_LEN>()
//                       + Precision<float>()
//                       + Type<fft_type::r2c>()
//                       + Direction<fft_direction::forward>()
//                       // + ElementsPerThread<8>()
//                       + SM<750>()
//                       + Block());

__device__ int16_t ulaw2pcm(uint8_t u_val)
{
    int16_t t;

    /* Complement to obtain normal u-law value. */
    u_val = ~u_val;

    /*
     * Extract and bias the quantization bits. Then
     * shift up by the segment number and subtract out the bias.
     */
    t = ((u_val & QUANT_MASK) << 3) + BIAS;
    t <<= ((unsigned)u_val & SEG_MASK) >> SEG_SHIFT;

    return ((u_val & SIGN_BIT) ? (BIAS - t) : (t - BIAS));
}

__global__ void kernel_dsp(rtp_packet *in, pktspectrum *out, int N) {

    int thrdId = blockIdx.x * blockDim.x + threadIdx.x;
    int threadsPerPacket = blockDim.x;

    if (thrdId >= N*threadsPerPacket) return;

    int pktId = thrdId/threadsPerPacket;

    rtp_packet packet = in[pktId];
    pktspectrum* spectrum = &out[pktId];

    // // copy packet payload into shared memory -- this will be for this block/packet, accessible by threadsPerPacket threads
    // __shared__ float payloadBuffer[RTP_PAYLOAD_LEN];

    // register mem for samples each thread will work on
    complex_type::value_type threadData[ELEMENTS_PER_THREAD]; // i.e. float

    // shared memory for fft
    extern __shared__ __align__(alignof(float4)) complex_type shared_mem[];

    // this threads range/stride in packet.payload
    int stride = ELEMENTS_PER_THREAD;
    int offset = (thrdId%threadsPerPacket) * stride;

    if (stride != FFT::storage_size) {
        assert(false);
    }
    //printf("thrdId: %d, pktId: %d, offset: %d, stride: %d,  ssrc: %d\n", thrdId, pktId, offset, stride, packet.ssrc);

    // decode and copy to shared memory
    for (int i = offset, l=0; i < offset + stride; i++, l++) {
        threadData[l] = ulaw2pcm(packet.payload[i]);
        // payloadBuffer[i] = threadData[l];
    }
    __syncthreads();

    // // further signal processing ...
    // // dummy load
    // for (int z=0; z<100; z++) {
    //     for (int i = offset; i < offset + stride; i++) {
    //         payloadBuffer[i] *= -1;
    //     }
    // }

    // fft
    FFT().execute(threadData, shared_mem);

    // // copy back to global memory
    // for (int i = offset; i < offset + stride; i++) {
    //     spectrum->payload[i] = payloadBuffer[i];
    // }

    if (thrdId%threadsPerPacket == 0) {
        spectrum->ssrc = packet.ssrc;
        //printf("ssrc: %d  thrdId: %d ;  threadsPerPacket=%d  pktId=%d\n", spectrum->ssrc, thrdId, threadsPerPacket, pktId);
    }

}