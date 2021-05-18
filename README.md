# Feed forward multi-layer perceptron

Layer 0 (Input)
Layer 1 (Hidden)
Layer 2 (Hidden)
Layer 3 (Ouput)

4 ingredients:
1. dataset (what are you working on?)
    - classification = images & labels (some class name)
    - regression = images, target output values (e.g. stock prices)
    - object detection (another form of regression) = images, labels, bounding boxes top-left / bottom-right coordinates
    - instance/semantic segmentation = images, labels, pixels masks (e.g. tries to classify every pixel)
1. loss function (how good/bad you're doing)
   Regression
   - Mean Squared Error (MSE)
   - Mean Absolute Error (MAE)
   - Mean Percentage Error 
   Classification
   - Binary cross-entropy (dog/cat)
   - Categorical cross-entropy (dog, cat, panda)
   Object Detection
   - Cross-entropy "how good classifying" + regression loss "how good bounding box" (Mean Squared Loss)
1. model/architecture (perceptron, multi-layer networks, etc.)
   Classification
   - VGG
   - ResNet
   - MobieNet
   Object Detection 
   - MobileNet + YOLO
   - ResNet + SSD
   - VGG + R-CNN
   Instance.segment
   - U-Net
   - Mask R-CNN
1. optimization method (drive loss function lower)
   - *SDG*
   - *Adam*
   - *RMSprop*
   - Nadam
   - Rectified Adam (RAdam)
   - Adadelta
   - Adagrad
   
Maintain lab notebook that includes << requires discipline
1. assumptions
1. network ingredients
1. hyperparameters (ex. learning rate)
1. results of experiments
1. whether results match expectation and assumption

# Weight initialization for Neural Networks
- procedure to set weights of NN to small random value that define the start point of training
- how...?
- fill the array with zeros? (constant)
- with ones? (constant)
- sample from a normal dist? Gaussian function we define the mean and std dev.
- a uniform dist? random value from the range, where every value has equal probability of being drawn
- LeCun Uniform and Normal?
  - default for Torch and PyTorch
  - define F_in (fan in, # of inputs)
  - define F_out (fan out, # of outputs)
- Glorot/Xavier Uniform and Normal (<< recommended for standard/non-residual NN)
  - Defule weight init for Keras and TF
  - the limit is constructed by adding the F_in and F_out
- MSRA/Kaiming (<< recommended for deep residual NN)
  - Kaiming He, creator of ResNet
- Recommend
   - never use constant initializations
   - use Xavier/Glorot init for non-residual NN
   - Use Kaiming He/MSRA init for residual NN
   - Consider swapping ReLUs for Leaky ReLU or PReLU
   - treat weight init as hyperparameter that needs to be tuned

Notes:
- High learning rate = step in the right direction, overstepping a  local/global optimum
- Low learning rate = tiny steps in the right direction, won't overstep the local/global optimum, intractable amount of time to converge  
- Not possible to correctly model the XOR function witha single layer perceptron
- sigmoid activation f(x) disadvantages outputs of sigmoid are not 0-centered, kills the gradient
- tanh activation f(x) is 0-centered and gradients die out when neurons saturated
- ReLU activation (x) is 0 for negative inputs, increases linearly for +values, not saturable, computationally efficient, outperforms sigmoid and tanh
- Parameteric ReLU allows learning rate to be learned activation-by-activation
- ELU is the go-to activation f(x), once obtained accuracy and results are satisfied 
- NN are common in clustering, vector quantization, pattern association
- Label Binarizer does One-Hot encoding integer labels as  vector labels