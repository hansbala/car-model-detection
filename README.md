## Inspiration

- We are all pretty fond of cars, and we thought it would be cool if a search engine like Google or DuckDuckGo could predict what the exact model of the car was when looking at a car image. 
- This classification ability would be useful for search engines because when a user types “Toyota,” the search engine would optimally want to display different models of a Toyota and not just one model. The search engine would also want to only display a diversity of Toyota models and not models of any other brand 
- This is a classic classification problem where we need to categorize an image of a car into one of many labels (model of Toyota car) 

## What it does

- We aim to predict the model of the car (sub-categorization) under the umbrella of a single brand, which we have chosen as Toyota.

## Related Work

- [Vehicle Detection using Deep Learning](https://www.ijeat.org/wp-content/uploads/papers/v9i1s5/A10061291S52019.pdf) classifies and detects whether a vehicle is a car, bus or a bike. This paper creates a convolution neural network from scratch to classify and detect these vehicles using a modern convolution neural network based on fast regions.
- [Car Make Recognition Systems](https://www.researchgate.net/publication/329906346_Review_of_car_make_model_recognition_systems) is a review article on CMMR (Car Make and Model Recognition) in real-world images. The interesting and relatable part about CMMR is that it uses fine-grained classification on certain parts of the car to attain a better accuracy when determining make and model
- Other papers which resemble ours are listed below:
  - [Paper 1](https://link.springer.com/chapter/10.1007/978-981-10-6451-7_7)
  - [Paper 2](https://www.hindawi.com/journals/complexity/2021/6644861/)
  - [Paper 3](https://ieeexplore.ieee.org/abstract/document/9216368)

## Data and Dataset

- The dataset we are using is linked [here](https://www.kaggle.com/occultainsights/toyota-cars-over-20k-labeled-images)
- We plan on using about 12k images for training the CNN and 2k random images to test the accuracy of the network.

## Methodology

- Taking inspiration from the CNN model used in class, we plan to use 3 convolutional layers coupled with 3 dense layers (pooling), along with one output layer of 34 classification buckets (number of models of Toyota cars).
- For preprocessing, we plan on using a feature extractor (Computer Vision) to extract relevant portions of the car (ideally bumpers, shapes, doors, rear-view mirrors) and then pass these features as input into the first convolutional layer. The pre-processing step will be an important portion of the assignment as passing the image as a whole into the CNN would not be an efficient way of tackling the problem - pixels that would ideally not be considered part of the car (or what makes the car unique) might disturb the balance of the CNN. Feature extraction ensures that the unique features of what makes the car unique is what is being used as training data for the CNN
- The idea of classifying Toyota models is novel. However, the methodology of classifying car models using CNNs have already been proven to work with other similar classification problems. Although classification through CNNs have been tried and tested, we are extracting data about certain unique parts such as the bumper, roof, wheels, and more of Toyota models to further help accuracy when classifying for models

## Metrics

- We plan to first train the model through a training set of 12k Toyota car images, which will be further preprocessed down through extracting certain parts of the images that give further information about the Toyota model. With the extracted parts of the training images, we plan to run the data through a CNN model 
- Accuracy does apply for this project. We are trying to get the highest accuracy for classifying Toyota models when given an image of a car during testing. We would also like for our model to recognize if the testing image is not an image of any Toyota model or label given 
Base: 75% 
Target: 85%
Stretch: 90% for any testing set including pictures of other cars that look very similar to models of Toyota. Also, testing sets with different angles of Toyota models 

## Ethics

1. Why is Deep Learning a good approach to this problem?
A. Deep Learning CNNs are particularly good at classification problems. Using something like an SVM (State Vector Machine) or plain feature extraction with HMM (Hidden Markov Models) will probably result in lower accuracy. SVMs and HMMs are particularly good at conditional probability but since the number of models in each of these brands are considerably larger than binary classification problems, these techniques probably wouldn’t perform well here. Therefore, we think that Deep Learning techniques would be a pretty good approach to the problem.

2. Who are the major “stakeholders” in this problem, and what are the consequences of mistakes made by your algorithm?
A. The major stakeholders in this problem are:
    - the car manufacturers who produce the cars
    - search engines which use our algorithm to display the cars,
    - consumers who buy Toyotas. 
The potential consequences of mistakes made by our algorithm are:
    - Misinformation - if a consumer finds a car model which is priced lower than the one they are looking for then they might be in false belief about whether they can afford it.
    - Fraud detection - if our model makes mistakes it might harm the insurance companies who use our algorithm to detect the car model of insurance policy holders 


