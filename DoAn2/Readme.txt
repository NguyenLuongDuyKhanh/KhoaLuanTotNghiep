TensorFlow trên rasp(không cài được vì gói wrapt)
https://www.makeuseof.com/tag/image-recognition-tensorflow-raspberry-pi/
https://www.youtube.com/watch?v=ukkNek46h_8
https://medium.com/@abhizcc/installing-latest-tensor-flow-and-keras-on-raspberry-pi-aac7dbf95f2

Keras-TensorFlow Win (đã cài được và chạy ầm ầm)
+Nhớ bật Anaconda lên chạy lệnh activate keras-gpu
https://github.com/antoniosehk/keras-tensorflow-windows-installation

Cách build dataset
https://www.pyimagesearch.com/2018/06/11/how-to-build-a-custom-face-recognition-dataset/
https://www.pyimagesearch.com/2018/04/09/how-to-quickly-build-a-deep-learning-image-dataset/
https://www.pyimagesearch.com/2017/12/04/how-to-create-a-deep-learning-dataset-using-google-images/

Giải thích về CNN
https://medium.com/syncedreview/accelerating-convolutional-neural-networks-on-raspberry-pi-725600463fd0
https://towardsdatascience.com/portable-computer-vision-tensorflow-2-0-on-a-raspberry-pi-part-1-of-2-84e318798ce9
https://www.edureka.co/blog/convolutional-neural-network/ (cái này siêu hay)

Phương pháp DNN
https://www.pyimagesearch.com/2018/06/25/raspberry-pi-face-recognition/
https://github.com/nischi/MMM-Face-Reco-DNN

https://www.pyimagesearch.com/2018/06/18/face-recognition-with-opencv-python-and-deep-learning/
https://www.pyimagesearch.com/2018/09/24/opencv-face-recognition/
https://www.pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/
https://www.pyimagesearch.com/2017/04/17/real-time-facial-landmark-detection-opencv-python-dlib/

Hai cái này cùng phương pháp nhưng chưa hiểu lắm
https://github.com/paviro/MMM-Facial-Recognition
https://github.com/normyx/MMM-Facial-Recognition-OCV3

Giải thích keras và tensorflow khác nhau
https://www.pyimagesearch.com/2019/10/21/keras-vs-tf-keras-whats-the-difference-in-tensorflow-2-0/

Các kiến trúc mạng neuron phổ biến:
Deep Neural Network (DNN) fully-connected layers
Convolutional Neural Network (CNN)
Recurrent Neural Network (RNN)
https://viblo.asia/p/deep-learning-qua-kho-dung-lo-da-co-keras-LzD5dBqoZjY
https://forum.machinelearningcoban.com/t/kien-truc-cac-mang-cnn-noi-tieng-phan-1-alex-lenet-inception-vgg/2582
https://techblog.vn/gioi-thieu-ve-cac-pre-trained-models-trong-linh-vuc-computer-vision
https://medium.com/analytics-vidhya/cnns-architectures-lenet-alexnet-vgg-googlenet-resnet-and-more-666091488df5

Phân biệt convolution layer và pooling
https://topdev.vn/blog/thuat-toan-cnn-convolutional-neural-network/
https://nttuan8.com/bai-6-convolutional-neural-network/

Test CNN với MNIST dataset (Chạy tốt trên laptop và cả colab)
https://nttuan8.com/bai-7-gioi-thieu-keras-va-bai-toan-phan-loai-anh/

Ô tô tự lái
https://nttuan8.com/bai-8-o-to-tu-lai-voi-udacity-open-source/

Vô cùng quan trọng, VGG with keras
https://machinelearningmastery.com/how-to-perform-face-recognition-with-vggface2-convolutional-neural-network-in-keras/
https://medium.com/analytics-vidhya/face-recognition-with-vgg-face-in-keras-96e6bc1951d5
https://medium.com/coinmonks/paper-review-of-vggnet-1st-runner-up-of-ilsvlc-2014-image-classification-d02355543a11

Obj tracking
https://www.pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/

Train model nhận diện chó mèo với keras (chạy tốt trên laptop)
https://www.pyimagesearch.com/2018/09/10/keras-tutorial-how-to-get-started-with-keras-deep-learning-and-python/
Có 2 mạng SmallVGGNet (CNN) và simple_nn (neuron)
python train_simple_nn.py --dataset animals --model output/simple_nn.model --label-bin output/simple_nn_lb.pickle --plot output/simple_nn_plot.png
python predict.py --image images/cat.jpg --model output/simple_nn.model --label-bin output/simple_nn_lb.pickle --width 32 --height 32 --flatten 1
python train_vgg.py --dataset animals --model output/smallvggnet.model --label-bin output/smallvggnet_lb.pickle --plot output/smallvggnet_plot.png
python predict.py --image images/panda.jpg --model output/smallvggnet.model --label-bin output/smallvggnet_lb.pickle --width 64 --height 64
python predict.py --image images/dog.jpg --model output/smallvggnet.model --label-bin output/smallvggnet_lb.pickle --width 64 --height 64
Kiếm được dataset 1000 mặt thì test đưuọc với facial
Khoogn chạy được trên rasp do có keras

Bao gồm bài toán facedetect và facerecognition
Mỗi bài toán có cách giải quyết riêng:
FaceDetection dùng
+HOG
+CNN(cái này thì rasp không dùng được do không có GPU)
FaceRecognition
+DNN
Haar()