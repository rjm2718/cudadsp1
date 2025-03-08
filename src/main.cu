#include <iostream>
#include <cmath>
#include <arpa/inet.h>

#include "rtpdsp.h"


int16_t search(int16_t val, int16_t *table, int16_t size)
{
    int16_t i;

    for (i = 0; i < size; i++) {
        if (val <= *table++)
            return (i);
    }
    return (size);
}

uint8_t pcm2ulaw(int16_t pcm_val)	/* 2's complement (16-bit range) */
{
    int16_t mask;
    int16_t seg;
    uint8_t uval;

    /* Get the sign and the magnitude of the value. */
    pcm_val = pcm_val >> 2;
    if (pcm_val < 0) {
        pcm_val = -pcm_val;
        mask = 0x7F;
    } else {
        mask = 0xFF;
    }
    if ( pcm_val > CLIP ) pcm_val = CLIP;		/* clip the magnitude */
    pcm_val += (BIAS >> 2);

    /* Convert the scaled magnitude to segment number. */
    seg = search(pcm_val, seg_uend, 8);

    /*
    * Combine the sign, segment, quantization bits;
    * and complement the code word.
    */
    if (seg >= 8)		/* out of range, return maximum value. */
        return (unsigned char) (0x7F ^ mask);
    else {
        uval = (unsigned char) (seg << 4) | ((pcm_val >> (seg + 1)) & 0xF);
        return (uval ^ mask);
    }

}


void add_sine_wave(int16_t *buffer, int length, double amplitude, double frequency, double sample_rate) {
    const double two_pi = 2.0 * M_PI;
    for (int i = 0; i < length; i++) {
        double t = static_cast<double>(i) / sample_rate; // Time for current sample
        buffer[i] += static_cast<int16_t>(amplitude * sin(two_pi * frequency * t));
    }
}

void mk_rtp_packet(rtp_packet *pkt, uint32_t ssrc) {

    // Set the header fields
    pkt->header[0] = 0x80; // Version 2, no padding, no extensions, CC=0
    pkt->header[1] = 0; // payload type ulaw

    // increment sequence number
    uint16_t *seqnum = (uint16_t*)&pkt->header[2];
    *seqnum++;

    uint32_t ssrc_network = htonl(ssrc);
    memcpy(&pkt->header[8], &ssrc_network, sizeof(uint32_t)); // Copy SSRC
    pkt->ssrc = ssrc;

    // simple sine wave composite
    int16_t buffer[RTP_PAYLOAD_LEN];
    memset(buffer, 0, sizeof(buffer));
    add_sine_wave(buffer, RTP_PAYLOAD_LEN, 6000, 440, 8000);

    double freq = 1000 + ssrc % 2000;
    add_sine_wave(buffer, RTP_PAYLOAD_LEN, 18000, freq, 8000);

    // convert to ulaw
    for (int i = 0; i < RTP_PAYLOAD_LEN; i++) {
        // printf("%d ", buffer[i]);
        pkt->payload[i] = pcm2ulaw(buffer[i]);
    }
    // printf("\n");
}

int main() {

    int PC = 1024;

    // buffer to write multiple packets to
    void* pktbuf_h = malloc(sizeof(rtp_packet) * PC);
    void* pbuf = pktbuf_h;
    for (int i = 0; i < PC; i++, pbuf += sizeof(rtp_packet)) {
        uint32_t ssrc = i;
        mk_rtp_packet((rtp_packet*)pbuf, ssrc);
    }

    // copy to device
    void* pktbuf_d;
    CUDA_ERR_CHK( cudaMalloc(&pktbuf_d, sizeof(rtp_packet) * PC) );
    CUDA_ERR_CHK( cudaMemcpy(pktbuf_d, pktbuf_h, sizeof(rtp_packet) * PC, cudaMemcpyHostToDevice) );

    // results buffer
    void* pktspcbuf_h = malloc(sizeof(pktspectrum) * PC);
    void* pktspcbuf_d;
    CUDA_ERR_CHK( cudaMalloc(&pktspcbuf_d, sizeof(pktspectrum) * PC) );


    // 1 block per packet
    int blocks = PC;


    // Timing setup
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    kernel_dsp<<<blocks, THREADS_PER_BLOCK, FFT::shared_memory_size>>>((rtp_packet*)pktbuf_d, (pktspectrum*)pktspcbuf_d, PC);

    cudaEventRecord(stop, 0);

    CUDA_ERR_CHK(cudaGetLastError()); // Check kernel launch errors
    CUDA_ERR_CHK(cudaDeviceSynchronize()); // Synchronize to capture runtime errors

    // Wait for the event to complete
    cudaEventSynchronize(stop);

    // Calculate elapsed time in milliseconds
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    std::cout << "Kernel execution time: " << elapsedTime << " ms\n";



    // get results
    CUDA_ERR_CHK( cudaMemcpy(pktspcbuf_h, pktspcbuf_d, sizeof(pktspectrum) * PC, cudaMemcpyDeviceToHost) );

    // for (int i = 0; i < PC; i++) {
    //     printf("%d\n", ((pktspectrum*)pktspcbuf_h)[i].ssrc);
    // }

    pktspectrum s10 = ((pktspectrum*)pktspcbuf_h)[10];
    printf("s10 ssrc: %d\n", s10.ssrc);
    for (int i = 0; i < RTP_PAYLOAD_LEN; i++) {
        // printf("%d ", s10.payload[i]);
    }
    // printf("\n");



    printf("ok\n");

    return 0;
}