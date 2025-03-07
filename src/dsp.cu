
#include "rtpdsp.h"


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

    // copy packet payload into shared memory -- this will be for this block/packet, accessible by threadsPerPacket threads
    // TODO adjustments for cuFFTDx
    __shared__ float payloadBuffer[RTP_PAYLOAD_LEN];

    // this threads range/stride in packet.payload
    int stride = RTP_PAYLOAD_LEN / threadsPerPacket;
    int offset = (thrdId%threadsPerPacket) * stride;
    //printf("thrdId: %d, pktId: %d, offset: %d, stride: %d,  ssrc: %d\n", thrdId, pktId, offset, stride, packet.ssrc);

    // decode and copy to shared memory
    for (int i = offset; i < offset + stride; i++) {
        payloadBuffer[i] = ulaw2pcm(packet.payload[i]);
    }
    __syncthreads();

    // further signal processing ...

    // copy back to global memory
    for (int i = offset; i < offset + stride; i++) {
        spectrum->payload[i] = payloadBuffer[i];
    }

    if (thrdId%threadsPerPacket == 0) {
        spectrum->ssrc = packet.ssrc;
        //printf("ssrc: %d  thrdId: %d ;  threadsPerPacket=%d  pktId=%d\n", spectrum->ssrc, thrdId, threadsPerPacket, pktId);
    }

}