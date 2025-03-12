
#include "rtpdsp.h"
#include "ulaw.h"

using namespace cufftdx;


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

    int threadId = threadIdx.x;
    int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
    int threadsPerPacket = blockDim.x;
    int stride = ELEMENTS_PER_THREAD;

    assert(stride==FFT::elements_per_thread);

    if (globalThreadId >= N*threadsPerPacket) return;

    int pktId = globalThreadId/threadsPerPacket;
    rtp_packet packet = in[pktId];
    pktspectrum* spectrum = &out[pktId];


    complex_type thread_data[FFT::elements_per_thread];

    // shared memory for fft.  this is local to the block, so we can use it as-is for the FFT().execute call
    extern __shared__ __align__(alignof(float4)) complex_type shared_mem[];

    const unsigned int local_fft_id = 0; // with one fft per block/packet, we can hardcode to 0
    unsigned int idx = threadId; // starting index for this thread

    for (int iteration = 0; iteration < 25; iteration++) {
        //example::io<FFT>::load(input_data, thread_data, local_fft_id);
        idx = threadId;
        for (unsigned int i = 0; i < FFT::elements_per_thread; i++) {
            if ((i * 2 * stride + threadId) < FFT::input_length) {
                thread_data[i].x = ulaw2pcm(packet.payload[idx]);
                idx += stride;
                thread_data[i].y = ulaw2pcm(packet.payload[idx]);
                idx += stride;
            }
        }

        FFT().execute(thread_data, shared_mem);

        // example::io<FFT>::store(thread_data, spectrum->spectrum, local_fft_id);
        idx = threadId;
        for (unsigned int i = 0; i < FFT::elements_per_thread; i++) {
            if ((i * stride + threadIdx.x) < FFT::output_length) {
                spectrum->spectrum[idx] = thread_data[i];
                idx += stride;
            }
        }



        if (threadId == 0) {
            spectrum->ssrc = packet.ssrc;
            //printf("ssrc: %d  thrdId: %d  globalThreadId: %d ;  threadsPerPacket=%d  pktId=%d\n", spectrum->ssrc, threadId, globalThreadId, threadsPerPacket, pktId);
        }
    }
    // __syncthreads();
}