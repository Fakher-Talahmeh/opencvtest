
#include <mpi.h>
#include <omp.h>
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <iostream>
#include <vector>
#include <stdint.h>
#include <chrono>
#include <numeric>

using namespace std;
using namespace cv;
using namespace std::chrono;

// ChaCha20 constants
const uint32_t CONSTANTS[4] = { 0x61707865, 0x3320646e, 0x79622d32, 0x6b206574 };

uint32_t rotateLeft(uint32_t value, int shift) {
    return (value << shift) | (value >> (32 - shift));
}

void quarterRound(uint32_t& a, uint32_t& b, uint32_t& c, uint32_t& d) {
    a += b; d ^= a; d = rotateLeft(d, 16);
    c += d; b ^= c; b = rotateLeft(b, 12);
    a += b; d ^= a; d = rotateLeft(d, 8);
    c += d; b ^= c; b = rotateLeft(b, 7);
}

void chacha20Block(uint32_t state[16], uint8_t output[64]) {
    uint32_t workingState[16];
    memcpy(workingState, state, 16 * sizeof(uint32_t));

    for (int i = 0; i < 10; ++i) {
        quarterRound(workingState[0], workingState[4], workingState[8], workingState[12]);
        quarterRound(workingState[1], workingState[5], workingState[9], workingState[13]);
        quarterRound(workingState[2], workingState[6], workingState[10], workingState[14]);
        quarterRound(workingState[3], workingState[7], workingState[11], workingState[15]);

        quarterRound(workingState[0], workingState[5], workingState[10], workingState[15]);
        quarterRound(workingState[1], workingState[6], workingState[11], workingState[12]);
        quarterRound(workingState[2], workingState[7], workingState[8], workingState[13]);
        quarterRound(workingState[3], workingState[4], workingState[9], workingState[14]);
    }

    for (int i = 0; i < 16; ++i) {
        workingState[i] += state[i];
        output[i * 4 + 0] = (workingState[i] >> 0) & 0xFF;
        output[i * 4 + 1] = (workingState[i] >> 8) & 0xFF;
        output[i * 4 + 2] = (workingState[i] >> 16) & 0xFF;
        output[i * 4 + 3] = (workingState[i] >> 24) & 0xFF;
    }
}

void chacha20Encrypt(uint8_t* plaintext, uint8_t* ciphertext, size_t length,
    const uint8_t key[32], const uint8_t nonce[12], uint32_t counter) {
    uint32_t state[16];

    state[0] = CONSTANTS[0];
    state[1] = CONSTANTS[1];
    state[2] = CONSTANTS[2];
    state[3] = CONSTANTS[3];

    for (int i = 0; i < 8; ++i) {
        state[4 + i] = ((uint32_t)key[i * 4 + 0] << 0) |
            ((uint32_t)key[i * 4 + 1] << 8) |
            ((uint32_t)key[i * 4 + 2] << 16) |
            ((uint32_t)key[i * 4 + 3] << 24);
    }

    state[12] = counter;
    state[13] = ((uint32_t)nonce[0] << 0) | ((uint32_t)nonce[1] << 8) |
        ((uint32_t)nonce[2] << 16) | ((uint32_t)nonce[3] << 24);
    state[14] = ((uint32_t)nonce[4] << 0) | ((uint32_t)nonce[5] << 8) |
        ((uint32_t)nonce[6] << 16) | ((uint32_t)nonce[7] << 24);
    state[15] = ((uint32_t)nonce[8] << 0) | ((uint32_t)nonce[9] << 8) |
        ((uint32_t)nonce[10] << 16) | ((uint32_t)nonce[11] << 24);

#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < static_cast<int>(length); i += 64) {
        uint8_t block[64];
        uint32_t localState[16];
        memcpy(localState, state, sizeof(state));
        localState[12] = counter + i / 64;

        chacha20Block(localState, block);

        size_t blockSize = std::min(size_t(64), length - static_cast<size_t>(i));
        for (size_t j = 0; j < blockSize; ++j) {
            ciphertext[i + j] = plaintext[i + j] ^ block[j];
        }
    }
}
void encryptROI(Mat& frame, const Rect& roi, const uint8_t key[32],
    const uint8_t nonce[12], uint32_t counter) {

#pragma omp parallel for collapse(2) schedule(dynamic)
    for (int i = roi.y; i < roi.y + roi.height; i++) {
        for (int j = roi.x; j < roi.x + roi.width; j++) {
            if (i < frame.rows && j < frame.cols) {
                Vec3b& pixel = frame.at<Vec3b>(i, j);

                // تشفير كل قناة من قنوات البكسل
                uint8_t plaintext[3] = { pixel[0], pixel[1], pixel[2] };
                uint8_t ciphertext[3];

                // استخدام موقع البكسل في المفتاح لتجنب تشفير متماثل
                uint32_t pixelCounter = counter + (i * frame.cols + j);
                chacha20Encrypt(plaintext, ciphertext, 3, key, nonce, pixelCounter);

                pixel[0] = ciphertext[0];
                pixel[1] = ciphertext[1];
                pixel[2] = ciphertext[2];
            }
        }
    }
}

int main(int argc, char* argv[]) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    if (provided != MPI_THREAD_MULTIPLE) {
        cerr << "Warning: MPI implementation does not fully support threads\n";
    }

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    uint8_t key[32] = { 0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6,
                       0xab, 0xf7, 0xcf, 0x93, 0x44, 0x00, 0x00, 0x00,
                       0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6,
                       0xab, 0xf7, 0xcf, 0x93, 0x44, 0x00 };
    uint8_t nonce[12] = { 0x00, 0x01, 0x02, 0x03, 0x04, 0x05,
                         0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b };

    if (rank == 0) {
        VideoCapture cap("C:\\Users\\fafan\\Documents\\test.mp4");
        if (!cap.isOpened()) {
            cerr << "Error opening video file" << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
            return -1;
        }

        // إعداد VideoWriter لحفظ الفيديو المشفر
        int frame_width = cap.get(CAP_PROP_FRAME_WIDTH);
        int frame_height = cap.get(CAP_PROP_FRAME_HEIGHT);
        int fps = cap.get(CAP_PROP_FPS);
        VideoWriter video("encrypted_output.mp4",
            VideoWriter::fourcc('m', 'p', '4', 'v'),
            fps, Size(frame_width, frame_height));

        CascadeClassifier faceCascade;
        if (!faceCascade.load("C:\\Users\\fafan\\source\\repos\\opencvtest\\opencvtest\\haarcascade_frontalface_default.xml")) {
            cerr << "Error loading face cascade" << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
            return -1;
        }

        namedWindow("Original", WINDOW_AUTOSIZE);
        namedWindow("Face Encryption", WINDOW_AUTOSIZE);

        const int BATCH_SIZE = 4;
        vector<Mat> frameBuffer(BATCH_SIZE);
        vector<Mat> encryptedFrames(BATCH_SIZE);

        int frameCount = 0;
        auto startTime = high_resolution_clock::now();
        vector<double> processingTimes;

        while (true) {
            int validFrames = 0;

            // قراءة دفعة من الفريمات
#pragma omp parallel for reduction(+:validFrames)
            for (int i = 0; i < BATCH_SIZE; i++) {
                Mat frame;
#pragma omp critical
                {
                    cap >> frame;
                }
                if (!frame.empty()) {
                    frameBuffer[i] = frame.clone();
                    encryptedFrames[i] = frame.clone();
                    validFrames++;
                }
            }

            if (validFrames == 0) break;

            auto batchStart = high_resolution_clock::now();

            // معالجة كل الفريمات في الدفعة بشكل متوازي
#pragma omp parallel for schedule(dynamic)
            for (int f = 0; f < validFrames; f++) {
                Mat gray;
                cvtColor(frameBuffer[f], gray, COLOR_BGR2GRAY);
                equalizeHist(gray, gray);

                vector<Rect> faces;
                faceCascade.detectMultiScale(gray, faces, 1.1, 3, 0, Size(30, 30));

                cout << "Frame " << frameCount + f << ": Found " << faces.size() << " faces" << endl;

                int facesPerProcess = faces.size() / size;
                int remainder = faces.size() % size;

                // معالجة جزء العملية الرئيسية
#pragma omp parallel for
                for (int i = 0; i < facesPerProcess; i++) {
                    encryptROI(encryptedFrames[f], faces[i], key, nonce, frameCount + f);
                    rectangle(encryptedFrames[f], faces[i], Scalar(0, 255, 0), 2);
                }

                // توزيع العمل على العمليات الأخرى
                vector<MPI_Request> requests;
                for (int i = 1; i < size; i++) {
                    int startIdx = i * facesPerProcess + min(i, remainder);
                    int endIdx = startIdx + facesPerProcess + (i < remainder ? 1 : 0);

                    for (int j = startIdx; j < endIdx && j < faces.size(); j++) {
                        MPI_Request req1, req2, req3;
                        MPI_Isend(&f, 1, MPI_INT, i, 0, MPI_COMM_WORLD, &req1);
                        MPI_Isend(&faces[j], sizeof(Rect), MPI_BYTE, i, 1, MPI_COMM_WORLD, &req2);

                        Mat faceROI = encryptedFrames[f](faces[j]);
                        MPI_Isend(faceROI.data, faceROI.total() * faceROI.elemSize(),
                            MPI_UNSIGNED_CHAR, i, 2, MPI_COMM_WORLD, &req3);

                        requests.push_back(req1);
                        requests.push_back(req2);
                        requests.push_back(req3);
                    }
                }

                if (!requests.empty()) {
                    MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
                }

                // استقبال النتائج
                for (int i = 1; i < size; i++) {
                    int startIdx = i * facesPerProcess + min(i, remainder);
                    int endIdx = startIdx + facesPerProcess + (i < remainder ? 1 : 0);

                    for (int j = startIdx; j < endIdx && j < faces.size(); j++) {
                        Mat faceROI = encryptedFrames[f](faces[j]);
                        MPI_Recv(faceROI.data, faceROI.total() * faceROI.elemSize(),
                            MPI_UNSIGNED_CHAR, i, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        rectangle(encryptedFrames[f], faces[j], Scalar(0, 255, 0), 2);
                    }
                }
            }

            auto batchEnd = high_resolution_clock::now();
            double batchTime = duration_cast<milliseconds>(batchEnd - batchStart).count();
            processingTimes.push_back(batchTime / validFrames);

            // عرض وحفظ الفريمات المعالجة
            for (int f = 0; f < validFrames; f++) {
                imshow("Original", frameBuffer[f]);
                imshow("Face Encryption", encryptedFrames[f]);
                video.write(encryptedFrames[f]);
                frameCount++;
            }

            if (waitKey(1) == 27) break;
        }

        // Signal workers to exit
        for (int i = 1; i < size; i++) {
            int endSignal = -1;
            MPI_Send(&endSignal, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
        }

        auto endTime = high_resolution_clock::now();
        double totalTime = duration_cast<milliseconds>(endTime - startTime).count() / 1000.0;
        double avgTime = accumulate(processingTimes.begin(), processingTimes.end(), 0.0) / processingTimes.size();

        cout << "\nProcessing Statistics:\n";
        cout << "Total time: " << totalTime << " seconds\n";
        cout << "Frames processed: " << frameCount << "\n";
        cout << "Average processing time per frame: " << avgTime << " ms\n";
        cout << "Average FPS: " << frameCount / totalTime << "\n";
        cout << "Processes used: " << size << "\n";
        cout << "Threads per process: " << omp_get_max_threads() << "\n";
        cout << "Batch size: " << BATCH_SIZE << "\n";

        video.release();
        cap.release();
        destroyAllWindows();
    }
    else {
        int num_threads = (rank > size / 2) ? 4 : 1;
        omp_set_num_threads(num_threads);

        while (true) {
            int frameIndex;
            MPI_Status status;
            MPI_Recv(&frameIndex, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);

            if (frameIndex == -1) break;

            Rect face;
            MPI_Recv(&face, sizeof(Rect), MPI_BYTE, 0, 1, MPI_COMM_WORLD, &status);

            vector<uint8_t> buffer(face.width * face.height * 3);
            MPI_Recv(buffer.data(), buffer.size(), MPI_UNSIGNED_CHAR,
                0, 2, MPI_COMM_WORLD, &status);

            Mat faceROI(face.height, face.width, CV_8UC3, buffer.data());

#pragma omp parallel if(rank > size/2)
            {
#pragma omp single
                {
                    encryptROI(faceROI, Rect(0, 0, face.width, face.height), key, nonce, frameIndex);
                }
            }

            MPI_Send(faceROI.data, faceROI.total() * faceROI.elemSize(),
                MPI_UNSIGNED_CHAR, 0, frameIndex, MPI_COMM_WORLD);
        }
    }

    MPI_Finalize();
    return 0;
}
