# Eigenfaces-Music-Genre-Classification

**Abstract**

As in the PCA project, we can see how useful SVD is to reduce dimensions and represent the whole data. In the first part we use two data sets to again demonstrate how it can be combined together to analyze images, with wavelet, which has been shown to be very useful in analyzing images. In the second part, we also use SVD to reduce dimensions and then use unsupervised and supervised learning, s.t. Naive Bayesian, Linear Discrimination, SVM, to try to classify music genres or bands.


**Introduction and Overview**

Here we describe two examples here to illustrate how SVD can help reduce dimensions and help our computations.

**Problem Description**

**.1 Eigenfaces**

Two datasets presented here are both images of faces, while one of the set has cropped faces and another one has uncropped faces, meaning the first type of images would be centered. What we want to do here is to find the important features of the faces. This could be used for future training to separate human faces and other things.

**.2 Music Classification**

We would conduct three tests: classify three bands from three different genres; classify three bands from the same genre; classfiy genre with songs from diffrent bands. Each band or genre would have its own feature in frequencies, and we hope to extract those features and then conduct testing.

**General Approach**

**.1 Eigenfaces**

We would use Haar wavelet on all images, and then conduct SVD to extract features. $U,S,V$ would be presented later for analysis.

**.2 Music Classification**

For each test, first, we would use SVD on the music dataset itself, and then train on them using supervised learning methods, s.t. Naive Bayesian, Linear Discrimination, and SVM, and then test; second, we obtain the music spectrogram first, then conduct SVD and do the same steps. We can use to illustrate how useful spectrogram for non-stationary dataset.

**Theoretical Background**

As we learned from our textbook \cite{kutz_2013}, haar wavelet is a very powerful tool to gain the features of images, whose function is giebn as follows:

    \Psi(t)=\begin{cases} 
      1 & 0 \leq t\leq 1/2 \\
      -1 & 1/2 \leq t\leq 1 \\
      0 & otherwise \\

similar to sound frequency information, Haar wavelet can also produce important and unique features of an image.

Classification can be done in two ways: unsupervised learning or supervised learning. For the first one, during training, no labels are given, we are mainly looking for cluster in data and look for the closest groups to let the dataset decide. For supervised training, labels are needed, and several methods such as Naive Bayesian, Linear Discrimination, SVM can be used to better classify data.

An important thing during training and testing is cross-validation. This allows us to see different combination of training and testing dataset and see how well the algorithm behaves generally.

**Algorithm Implementation and Development**

**.1 Eigenfaces

    \item Load cropped and uncropped faces dataset.
    \item Use Algorithm~\ref{alg:wavelet} to find the images' wavelet representation.
    
    \begin{algorithm}
    \begin{algorithmic}
    \STATE{get the datafile, m, n, nw}
        \FOR{$j = 1:column length of datafile$}
            \STATE{Reshape datafile(:, i) to shape m by n and store it as X}
            \STATE{apply dwt2(X, 'haar') and store results in cA, cH, cV, cD}
            \STATE{apply $wcodemat(cH)$ and store the result in cod cH1}
            \STATE{apply $wcodemat(cV)$ and store the result in cod cV1}
            \STATE{combine $cod_cH1+cod_cV1$ as cod edge}
            \STATE{reshape cod edge back to a column vector and add in dcData}
        \ENDFOR
    \STATE{return dcData}
    
    \item Apply SVD to the wavelet representation.
    \item The result is left later for analysis.

**.2 Music Classification**

The following is demonstrated for all three tests.

    \item Load music dataset
    \item Use SVD on music dataset
    \item Algorithm~\ref{alg:training} is used to train and test, and obtain rate of correctness.
    
    \begin{algorithm}
    \begin{algorithmic}
    \STATE{Define cross validation times as cross}
        \FOR{$j = 1:cross$}
            \STATE{Shuffle the dataset and put 80 percent into training and 20 percent into testing}
            \STATE{Labeling the training data}
            \STATE{apply Naive Bayesian method and make prediction}
            \STATE{apply Linear Discrimination method and make prediction}
            \STATE{apply SVM method and make prediction}
            \STATE{compare prediction to testing data and calculation the correct rate and memorize it}
        \ENDFOR
    \STATE{calculate average correct rate across validation}
    
    \item Obtain spectrogram of music dataset.
    \item Use SVD on music spectrogram dataset
    \item Algorithm~\ref{alg:training} is used to train and test, and obtain rate of correctness.

**Computational Results**

**.1 Eigenfaces

As we can see in the first four modes of U, in Figure~\ref{fig:eigenfaces}, copped faces show very clear human face features, while uncropped faces do not. This might be results of spread features across the dataset of uncropped faces, as we can see in Figure~\ref{fig:p1_proj}, the cropped dataset has more gathered features, while for uncropped faces, it's more spread out. We can also see the comparison between the rank of cropped and uncropped dataset in Figure~\ref{fig:p1_s}, meaning the modes for cropped dataset is 3 while for uncropped is 1.

![figure 1](https://github.com/EchoRLiu/Eigenfaces-Music-Genre-Classification/blob/master/eigenfaces.jpg)
![figure 2](https://github.com/EchoRLiu/Eigenfaces-Music-Genre-Classification/blob/master/p1_projections.jpg)
![figure 3](https://github.com/EchoRLiu/Eigenfaces-Music-Genre-Classification/blob/master/p1_s.jpg)

**.2 Music Classification

Through the three tests, we can see how spectrogram improves the prediction results.

In test 1, with 300 cross-validation, the best prediction correctness rate is $.6373$ with Naive Baynesian method for doing SVD on music dataset itself; while the best prediction correctness rate is $.84$ with Linear Discrimination method for doing SVD on music spectrogram.

![figure 4](https://github.com/EchoRLiu/Eigenfaces-Music-Genre-Classification/blob/master/test1_u.jpg)
![figure 5](https://github.com/EchoRLiu/Eigenfaces-Music-Genre-Classification/blob/master/test1_s.jpg)
![figure 6](https://github.com/EchoRLiu/Eigenfaces-Music-Genre-Classification/blob/master/test1_V.jpg)

In test 2, with 300 cross-validation, the best prediction correctness rate is $.6107$ with Naive Baynesian method for doing SVD on music dataset itself; while the best prediction correctness rate is $.9178$ with Linear Discrimination method for doing SVD on music spectrogram.

![figure 7](https://github.com/EchoRLiu/Eigenfaces-Music-Genre-Classification/blob/master/test2_u.jpg)
![figure 8](https://github.com/EchoRLiu/Eigenfaces-Music-Genre-Classification/blob/master/test2_s.jpg)
![figure 9](https://github.com/EchoRLiu/Eigenfaces-Music-Genre-Classification/blob/master/test2_v.jpg)

In test 3, with 300 cross-validation, the best prediction correctness rate is $.5619$ with Naive Baynesian method for doing SVD on music dataset itself; while the best prediction correctness rate is $.7588$ with Linear Discrimination method for doing SVD on music spectrogram.

![figure 10](https://github.com/EchoRLiu/Eigenfaces-Music-Genre-Classification/blob/master/test3_u.jpg)
![figure 11](https://github.com/EchoRLiu/Eigenfaces-Music-Genre-Classification/blob/master/test3_s.jpg)
![figure 12](https://github.com/EchoRLiu/Eigenfaces-Music-Genre-Classification/blob/master/test3_V.jpg)

We can also see as the dataset gets more complicated, the ability of supervised learning is reduced. We can also notice how after taking the spectrogram, the clusters of three classes are more spread out rather than clustered togteher at the beginning.

**Summary and Conclusions**

Through the two parts examples, we can see again how useful SVD is. We can also see how wavelet is used to obtain features on images and how supervised training can be used to classify data.
