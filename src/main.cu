#include <iostream>
#include <cmath>
#include <arpa/inet.h>
#include <chrono>

#include "rtpdsp.h"
#include "ulaw.h"


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
    add_sine_wave(buffer, RTP_PAYLOAD_LEN, 4000, 32, 4096);

    double freq = 100 + ssrc % 1000;
    add_sine_wave(buffer, RTP_PAYLOAD_LEN, 8000, freq, 4096);

    // convert to ulaw
    for (int i = 0; i < RTP_PAYLOAD_LEN; i++) {
        // printf("%d ", buffer[i]);
        pkt->payload[i] = pcm2ulaw(buffer[i]);
    }
    // printf("\n");
}

void printPktSpctrm(void* ps, int n) {
    char fn[100];
    sprintf(fn, "spectrum-%d.csv", n);
    FILE *file = fopen(fn, "w");
    if (!file) {
        perror("Failed to open file");
        return;
    }
    pktspectrum s = ((pktspectrum *) ps)[n];
    for (int i = 0; i < FFT::output_length; i++) {
        fprintf(file, "%.1f %.1f\n", s.spectrum[i].x, s.spectrum[i].y);
    }
    fclose(file);
    printf("wrote data to %s for ssrc %d: 8, %.1f\n", fn, s.ssrc, (100+n%1000)/32.0*8.0);
}

void processPackets(void* pktbuf_h, void* pktbuf_d, void* pktspcbuf_h, void* pktspcbuf_d, int n_packets, int n_iterations) {

    for (int i = 0; i < n_iterations; i++) {

        CUDA_ERR_CHK( cudaMemcpy(pktbuf_d, pktbuf_h, sizeof(rtp_packet) * n_packets, cudaMemcpyHostToDevice) );

        // Timing setup
        // cudaEvent_t start, stop;
        // cudaEventCreate(&start);
        // cudaEventCreate(&stop);
        // cudaEventRecord(start, 0);

        kernel_dsp<<<n_packets, THREADS_PER_BLOCK, FFT::shared_memory_size>>>((rtp_packet*)pktbuf_d, (pktspectrum*)pktspcbuf_d, n_packets);

        // cudaEventRecord(stop, 0);

        CUDA_ERR_CHK(cudaGetLastError()); // Check kernel launch errors

        CUDA_ERR_CHK(cudaMemcpy(pktspcbuf_h, pktspcbuf_d, sizeof(pktspectrum) * n_packets, cudaMemcpyDeviceToHost));

        // Wait for the event to complete
        // cudaEventSynchronize(stop);

        // Calculate elapsed time in milliseconds
        // float elapsedTime;
        // cudaEventElapsedTime(&elapsedTime, start, stop);
        // std::cout << "Kernel execution time: " << elapsedTime << " ms\n";
    }
    
    CUDA_ERR_CHK(cudaDeviceSynchronize());
}

int main(int argc, char** argv) {

    int n_iterations = 4;
    if (argc > 1) {
        n_iterations = std::stoi(argv[1]);
    }

    int NUM_PACKETS = 20000;

    // allocate host buffer and generate test packets
    void* pktbuf_h = malloc(sizeof(rtp_packet) * NUM_PACKETS);
    void* pbuf = pktbuf_h;
    for (int i = 0; i < NUM_PACKETS; i++, pbuf += sizeof(rtp_packet)) {
        uint32_t ssrc = i;
        mk_rtp_packet((rtp_packet*)pbuf, ssrc);
    }

    // rtp_packet p10 = ((rtp_packet*)pktbuf_h)[10];
    // for (int i = 0; i < RTP_PAYLOAD_LEN; i++) {
    //     printf("%d\n", p10.payload[i]);
    // }
    // return 0;

    void* pktbuf_d;
    CUDA_ERR_CHK( cudaMalloc(&pktbuf_d, sizeof(rtp_packet) * NUM_PACKETS) );

    // results buffer
    void* pktspcbuf_h = malloc(sizeof(pktspectrum) * NUM_PACKETS);
    void* pktspcbuf_d;
    CUDA_ERR_CHK( cudaMalloc(&pktspcbuf_d, sizeof(pktspectrum) * NUM_PACKETS) );

    assert(!FFT::requires_workspace);

    printf("num packets %d\n", NUM_PACKETS);
    printf("fft input_length %d\n", FFT::input_length);
    printf("fft output_length %d\n", FFT::output_length);
    printf("block size %d, elements per thread %d\n", ELEMENTS_PER_THREAD, THREADS_PER_BLOCK);
    printf("per-SM shared memory size %d KB\n", FFT::shared_memory_size/1024);
    printf("hostToDevice copy %d MB\n", sizeof(rtp_packet) * NUM_PACKETS/1024/1024);
    printf("deviceToHost copy %d MB\n", sizeof(pktspectrum) * NUM_PACKETS/1024/1024);

    auto start_time = std::chrono::high_resolution_clock::now();

    processPackets(pktbuf_h, pktbuf_d, pktspcbuf_h, pktspcbuf_d, NUM_PACKETS, n_iterations);

    auto end_time = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    auto us_per_packet = (double)duration / NUM_PACKETS / n_iterations * 1000.0;
    printf("\niterations: %d\n", n_iterations);
    printf("total time: %d ms (%.1f ms per iteration, %.1f μs per packet)\n", duration, (float)duration/n_iterations, us_per_packet);



    // printPktSpctrm(pktspcbuf_h, 10);
    // printPktSpctrm(pktspcbuf_h, 150);
    // printPktSpctrm(pktspcbuf_h, 450);

    return 0;
}