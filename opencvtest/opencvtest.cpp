#include <mpi.h>
#include <omp.h>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <numeric>

using namespace std;
using namespace cv;
using namespace std::chrono;
using namespace cv::dnn;

// ChaCha20 constants
const uint32_t CONSTANTS[4] = { 0x61707865, 0x3320646e, 0x79622d32, 0x6b206574 };

// Rotate left function for ChaCha20
uint32_t rotateLeft(uint32_t value, int shift) {
    return (value << shift) | (value >> (32 - shift));
}

// ChaCha20 quarter round
void quarterRound(uint32_t& a, uint32_t& b, uint32_t& c, uint32_t& d) {
    a += b; d ^= a; d = rotateLeft(d, 16);
    c += d; b ^= c; b = rotateLeft(b, 12);
    a += b; d ^= a; d = rotateLeft(d, 8);
    c += d; b ^= c; b = rotateLeft(b, 7);
}

// ChaCha20 block generation
void chacha20Block(uint32_t state[16], uint8_t output[64]) {
    uint32_t workingState[16];
    memcpy(workingState, state, sizeof(workingState));

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

// ChaCha20 encryption
void chacha20Encrypt(uint8_t* plaintext, uint8_t* ciphertext, size_t length, const uint8_t key[32], const uint8_t nonce[12], uint32_t counter) {
    uint32_t state[16];

    // Initialize ChaCha20 state
    memcpy(state, CONSTANTS, 4 * sizeof(uint32_t));
    memcpy(state + 4, key, 32);
    state[12] = counter;
    memcpy(state + 13, nonce, 12);

    // Parallel encryption
#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < length; i += 64) {
        uint8_t block[64];
        uint32_t localState[16];
        memcpy(localState, state, sizeof(state));
        localState[12] = counter + i / 64;

        chacha20Block(localState, block);

        size_t blockSize = min(size_t(64), length - i);
        for (size_t j = 0; j < blockSize; ++j) {
            ciphertext[i + j] = plaintext[i + j] ^ block[j];
        }
    }
}

// Encrypt Region of Interest (ROI)
void encryptROI(Mat& frame, const Rect& roi, const uint8_t key[32], const uint8_t nonce[12], uint32_t counter) {
#pragma omp parallel for collapse(2) schedule(dynamic)
    for (int i = roi.y; i < roi.y + roi.height; i++) {
        for (int j = roi.x; j < roi.x + roi.width; j++) {
            if (i < frame.rows && j < frame.cols) {
                Vec3b& pixel = frame.at<Vec3b>(i, j);
                uint8_t plaintext[3] = { pixel[0], pixel[1], pixel[2] };
                uint8_t ciphertext[3];

                uint32_t pixelCounter = counter + (i * frame.cols + j);
                chacha20Encrypt(plaintext, ciphertext, 3, key, nonce, pixelCounter);

                pixel[0] = ciphertext[0];
                pixel[1] = ciphertext[1];
                pixel[2] = ciphertext[2];
            }
        }
    }
}

class BatchProcessor {
private:
    const int BATCH_SIZE;
    queue<Mat> frameQueue;
    mutex queueMutex;
    condition_variable condVar;
    bool finished;

public:
    BatchProcessor(int batchSize) : BATCH_SIZE(batchSize), finished(false) {}

    void addFrame(const Mat& frame) {
        unique_lock<mutex> lock(queueMutex);
        frameQueue.push(frame.clone());
        lock.unlock();
        condVar.notify_one();
    }

    vector<Mat> getNextBatch() {
        unique_lock<mutex> lock(queueMutex);
        vector<Mat> batch;

        condVar.wait(lock, [this]() {
            return frameQueue.size() >= BATCH_SIZE || (finished && !frameQueue.empty());
            });

        int batchSize = min(BATCH_SIZE, (int)frameQueue.size());
        for (int i = 0; i < batchSize; i++) {
            if (!frameQueue.empty()) {
                batch.push_back(frameQueue.front());
                frameQueue.pop();
            }
        }

        return batch;
    }

    void setFinished() {
        unique_lock<mutex> lock(queueMutex);
        finished = true;
        lock.unlock();
        condVar.notify_all();
    }

    bool isFinished() {
        unique_lock<mutex> lock(queueMutex);
        return finished && frameQueue.empty();
    }

    size_t queueSize() {
        unique_lock<mutex> lock(queueMutex);
        return frameQueue.size();
    }
};

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

    const int OPTIMAL_BATCH_SIZE = max(16, omp_get_max_threads() * 4);

    if (rank == 0) {
        const string videoPath = "C:\\Users\\fafan\\Downloads\\PEOPLE FACES HD #video #stockfootage #backgroundvideo #stockvideo #people  #faces.mp4";
        VideoCapture cap;

        try {
            if (!cap.open(videoPath)) {
                if (!cap.open(videoPath, CAP_FFMPEG)) {
                    throw runtime_error("Failed to open video file: " + videoPath);
                }
            }

            cout << "Successfully opened video file" << endl;

            int frame_width = cap.get(CAP_PROP_FRAME_WIDTH);
            int frame_height = cap.get(CAP_PROP_FRAME_HEIGHT);
            int fps = cap.get(CAP_PROP_FPS);

            cout << "Video Properties:" << endl
                << "Width: " << frame_width << endl
                << "Height: " << frame_height << endl
                << "FPS: " << fps << endl;

            VideoWriter video("encrypted_output.mp4",
                VideoWriter::fourcc('m', 'p', '4', 'v'),
                fps, Size(frame_width, frame_height));

            if (!video.isOpened()) {
                throw runtime_error("Could not create output video file");
            }

            String weightsPath = "C:\\Users\\fafan\\Downloads\\yolov4.weights";
            String configPath = "C:\\Users\\fafan\\Downloads\\yolov4 (1).cfg";
            Net net = readNetFromDarknet(configPath, weightsPath);
            if (net.empty()) {
                throw runtime_error("Could not load YOLO model");
            }

            namedWindow("Original", WINDOW_NORMAL);
            namedWindow("Face Encryption", WINDOW_NORMAL);
            resizeWindow("Original", frame_width / 2, frame_height / 2);
            resizeWindow("Face Encryption", frame_width / 2, frame_height / 2);

            BatchProcessor batchProcessor(OPTIMAL_BATCH_SIZE);
            atomic<bool> shouldExit(false);

            thread readThread([&]() {
                Mat frame;
                while (!shouldExit && cap.read(frame)) {
                    if (frame.empty()) {
                        cerr << "Warning: Empty frame received" << endl;
                        continue;
                    }
                    batchProcessor.addFrame(frame);
                }
                batchProcessor.setFinished();
                });

            vector<double> processingTimes;
            auto startTime = high_resolution_clock::now();
            int frameCount = 0;

            while (!batchProcessor.isFinished() && !shouldExit) {
                vector<Mat> batch = batchProcessor.getNextBatch();
                if (batch.empty()) continue;

                auto batchStart = high_resolution_clock::now();

                for (size_t i = 0; i < batch.size(); i++) {
                    Mat displayFrame = batch[i].clone();
                    Mat processedFrame = batch[i].clone();

                    Mat blob = blobFromImage(processedFrame, 1 / 255.0, Size(416, 416), Scalar(0, 0, 0), true, false);
                    net.setInput(blob);
                    Mat detections = net.forward();

                    vector<Rect> faces;
                    vector<float> confidences;

                    for (int j = 0; j < detections.size[1]; j++) {
                        float* data = (float*)detections.ptr(0, j);
                        float confidence = data[4];
                        if (confidence > 0.5) {
                            int x = static_cast<int>(data[0] * processedFrame.cols);
                            int y = static_cast<int>(data[1] * processedFrame.rows);
                            int width = static_cast<int>(data[2] * processedFrame.cols);
                            int height = static_cast<int>(data[3] * processedFrame.rows);
                            faces.emplace_back(x, y, width, height);
                            confidences.emplace_back(confidence);
                        }
                    }

                    if (!faces.empty()) {
                        // Process faces in main process
                        int mainProcessFaces = faces.size() / size;

#pragma omp parallel for
                        for (int j = 0; j < mainProcessFaces; j++) {
                            encryptROI(processedFrame, faces[j], key, nonce, frameCount + i);
                            rectangle(processedFrame, faces[j], Scalar(0, 255, 0), 2);
                            rectangle(displayFrame, faces[j], Scalar(0, 255, 0), 2);
                        }

                        // Distribute remaining faces to other processes
                        for (int p = 1; p < size; p++) {
                            int signal = 1;
                            MPI_Send(&signal, 1, MPI_INT, p, 0, MPI_COMM_WORLD);

                            int startIdx = mainProcessFaces + (p - 1) * (faces.size() - mainProcessFaces) / (size - 1);
                            int endIdx = mainProcessFaces + p * (faces.size() - mainProcessFaces) / (size - 1);
                            int numFaces = endIdx - startIdx;

                            if (numFaces > 0) {
                                MPI_Send(&numFaces, 1, MPI_INT, p, 1, MPI_COMM_WORLD);

                                for (int j = startIdx; j < endIdx; j++) {
                                    MPI_Send(&faces[j], sizeof(Rect), MPI_BYTE, p, 2, MPI_COMM_WORLD);
                                    Mat faceROI = processedFrame(faces[j]);
                                    int dataSize = faceROI.total() * faceROI.elemSize();
                                    MPI_Send(&dataSize, 1, MPI_INT, p, 3, MPI_COMM_WORLD);
                                    MPI_Send(faceROI.data, dataSize, MPI_UNSIGNED_CHAR, p, 4, MPI_COMM_WORLD);
                                }

                                for (int j = startIdx; j < endIdx; j++) {
                                    Mat faceROI = processedFrame(faces[j]);
                                    int dataSize = faceROI.total() * faceROI.elemSize();
                                    MPI_Recv(faceROI.data, dataSize, MPI_UNSIGNED_CHAR, p, 5,
                                        MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                                    rectangle(processedFrame, faces[j], Scalar(0, 255, 0), 2);
                                    rectangle(displayFrame, faces[j], Scalar(0, 255, 0), 2);
                                }
                            }
                        }
                    }
                    else {
                        // If no faces found, encrypt the entire frame
                        Rect fullFrame(0, 0, processedFrame.cols, processedFrame.rows);
                        encryptROI(processedFrame, fullFrame, key, nonce, frameCount + i);
                    }

                    auto currentTime = high_resolution_clock::now();
                    auto timestamp = duration_cast<milliseconds>(currentTime - startTime).count() / 1000.0;
                    double currentFPS = frameCount / timestamp;

                    string frameInfo = format("Frame: %d", frameCount + 1);
                    string faceInfo = format("Faces: %zu", faces.size());
                    string timeInfo = format("Time: %.2fs", timestamp);
                    string fpsInfo = format("FPS: %.2f", currentFPS);

                    // Display information on the original frame
                    putText(displayFrame, frameInfo, Point(30, 30),
                        FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 255, 0), 2);
                    putText(displayFrame, faceInfo, Point(30, 60),
                        FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 255, 0), 2);
                    putText(displayFrame, timeInfo, Point(30, 90),
                        FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 255, 0), 2);
                    putText(displayFrame, fpsInfo, Point(30, 120),
                        FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 255, 0), 2);

                    // Display information on the processed frame
                    putText(processedFrame, frameInfo, Point(30, 30),
                        FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 255, 0), 2);
                    putText(processedFrame, faceInfo, Point(30, 60),
                        FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 255, 0), 2);
                    putText(processedFrame, timeInfo, Point(30, 90),
                        FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 255, 0), 2);
                    putText(processedFrame, fpsInfo, Point(30, 120),
                        FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 255, 0), 2);

                    imshow("Original", displayFrame);
                    imshow("Face Encryption", processedFrame);
                    video.write(processedFrame);

                    char c = waitKey(1);
                    if (c == 27) { // ESC key
                        shouldExit = true;
                        break;
                    }

                    frameCount++;
                }

                auto batchEnd = high_resolution_clock::now();
                double batchTime = duration_cast<milliseconds>(batchEnd - batchStart).count();
                processingTimes.push_back(batchTime / batch.size());

                cout << "\rProcessing frame: " << frameCount
                    << " | FPS: " << frameCount / (duration_cast<milliseconds>(batchEnd - startTime).count() / 1000.0)
                    << flush;
            }

            shouldExit = true;
            readThread.join();

            // Signal workers to exit
            for (int p = 1; p < size; p++) {
                int signal = -1;
                MPI_Send(&signal, 1, MPI_INT, p, 0, MPI_COMM_WORLD);
            }

            auto endTime = high_resolution_clock::now();
            double totalTime = duration_cast<milliseconds>(endTime - startTime).count() / 1000.0;
            double avgTime = accumulate(processingTimes.begin(), processingTimes.end(), 0.0) /
                processingTimes.size();

            cout << "\n\nProcessing Statistics:\n"
                << "Total time: " << totalTime << " seconds\n"
                << "Frames processed: " << frameCount << "\n"
                << "Average processing time per frame: " << avgTime << " ms\n"
                << "Average FPS: " << frameCount / totalTime << "\n"
                << "Processes used: " << size << "\n"
                << "Threads per process: " << omp_get_max_threads() << "\n"
                << "Batch size: " << OPTIMAL_BATCH_SIZE << "\n";

            video.release();
            cap.release();
            destroyAllWindows();
        }
        catch (const exception& e) {
            cerr << "Error: " << e.what() << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
            return -1;
        }
    }
    else {
        // Worker process code
        while (true) {
            MPI_Status status;
            int signal;

            MPI_Recv(&signal, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
            if (signal == -1) break;

            int numFaces;
            MPI_Recv(&numFaces, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);

            for (int i = 0; i < numFaces; i++) {
                Rect face;
                MPI_Recv(&face, sizeof(Rect), MPI_BYTE, 0, 2, MPI_COMM_WORLD, &status);

                int dataSize;
                MPI_Recv(&dataSize, 1, MPI_INT, 0, 3, MPI_COMM_WORLD, &status);

                vector<uchar> buffer(dataSize);
                MPI_Recv(buffer.data(), dataSize, MPI_UNSIGNED_CHAR, 0, 4,
                    MPI_COMM_WORLD, &status);

                try {
                    Mat faceROI(face.height, face.width, CV_8UC3, buffer.data());
                    encryptROI(faceROI, Rect(0, 0, face.width, face.height),
                        key, nonce, status.MPI_TAG);

                    MPI_Send(faceROI.data, dataSize, MPI_UNSIGNED_CHAR, 0, 5, MPI_COMM_WORLD);
                }
                catch (const exception& e) {
                    cerr << "Worker error: " << e.what() << endl;
                }
            }
        }
    }

    MPI_Finalize();
    return 0;
}