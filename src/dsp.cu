
#include "rtpdsp.h"
#include "ulaw.h"

using namespace cufftdx;


/* Given the rtp_packet input buffer, each block will work on a single packet.  This kernel first decodes the
 * audio payload from ulaw (g711) to pcm, and then performs an FFT before copying into pktspectrum output.
 * Does this 25 times just for fun -- simulating further process here.
 */
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

    // sample data will be stored in register memory
    complex_type thread_data[FFT::elements_per_thread];

    // shared memory for fft.  this is local to the block, so we can use it as-is for the FFT().execute call
    extern __shared__ __align__(alignof(float4)) complex_type shared_mem[];

    const unsigned int local_fft_id = 0; // with one fft per block/packet, we can hardcode to 0
    unsigned int idx = threadId; // starting index for this thread

    for (int iteration = 0; iteration < 25; iteration++) {

        // decode RTP packet header for codec choice
        decode_sample_op decode;
        if (packet.header[1] == 0) {
            decode = ulaw2pcm;
        } else {
            // other codecs as implemented ...
        }

        // samples are interleaved into thread_data for processing
        // derived from: example::io<FFT>::load(input_data, thread_data, local_fft_id);
        idx = threadId;
        for (unsigned int i = 0; i < FFT::elements_per_thread; i++) {
            if ((i * 2 * stride + threadId) < FFT::input_length) {
                thread_data[i].x = decode(packet.payload[idx]);
                idx += stride;
                thread_data[i].y = decode(packet.payload[idx]);
                idx += stride;
            }
        }
        // __syncthreads();

        FFT().execute(thread_data, shared_mem);

        // de-interleave before copying results back to global memory
        idx = threadId;
        for (unsigned int i = 0; i < FFT::elements_per_thread; i++) {
            if ((i * stride + threadIdx.x) < FFT::output_length) {
                spectrum->spectrum[idx] = thread_data[i];
                idx += stride;
            }
        }


        // fill in rest of pktspectrum data structure as needed
        if (threadId == 0) { // no parallelism here
            spectrum->ssrc = packet.ssrc;
            //printf("ssrc: %d  thrdId: %d  globalThreadId: %d ;  threadsPerPacket=%d  pktId=%d\n", spectrum->ssrc, threadId, globalThreadId, threadsPerPacket, pktId);
        }
    }

}