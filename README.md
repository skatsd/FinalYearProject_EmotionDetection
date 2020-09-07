In present day technology human-machine interaction is growing in demand and machine
needs to understand human gestures and emotions. If a machine can identify human emotions,
it can understand human behaviour better, thus improving the task efficiency. Emotions can
understand by text, vocal, verbal and facial expressions. Facial expressions play big role in
judging emotions of a person. It is found that limited work is done in field of real time emotion
recognition using facial images. This project presents an integrated system for human emotion
detection.

The proposed approach takes the video stream as input and produces the emotion label
corresponding to this video sequence. This output is encoded as one out of seven classes:
Anger, Disgust, Fear, Happiness, sad, surprise and Neutral. The system contains several
pipelined modules: face detection, image pre-processing, deep feature extraction, and feature
encoding. In the proposed method we use Haar cascade, feature extraction and Cascade
classifiers in face detection. This model is implemented on general convolutional neural
network (CNN) building framework for designing real-time CNNs.

We validate our models by creating a real-time vision system which accomplishes the tasks of
face detection and emotion classification simultaneously in one blended step using our
proposed CNN architecture. After presenting the details of the training procedure setup we
proceed to evaluate on standard benchmark sets. We report accuracies of 60.08% in the FER-
2013 emotion dataset. In training the model pipeline we use: several convolutional neural
networks with rectifier activation function, several maxpooling layers, flatten layer, dense
layer and final layer with seven neuron classifiers. When this model is feed video input can
recognize real time emotion of the person in the video.

In this project, we have worked on Spatial Domain by recognising 7 different types of emotions
(Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise) from videos of human faces.
Supervised Deep Learning approach was used to train the model for further analysis and
extensive testing. Finally, the model was tested against photos and videos which were
completely independent from the training and validation set. Individual emotion percentage
per video and the number of faces processed were calculated per video to draw conclusion to
our work.