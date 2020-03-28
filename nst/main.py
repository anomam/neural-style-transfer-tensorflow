"""Module containing main Neural Style Transfer class and methods"""
from keras import backend as K
from keras.models import Sequential
from keras.layers import AveragePooling2D, InputLayer
from keras.applications.vgg19 import preprocess_input, VGG19
from .utils import Image
import tensorflow as tf
from typing import Optional, Tuple
import numpy as np
from copy import deepcopy
import os
from tqdm import tqdm
from streamlit.DeltaGenerator import DeltaGenerator
import datetime as dt


class NeuralStyleTransfer:
    """Neural style transfer class that implements the whole workflow for
    transfering artistic style to pictures, as described in the NST paper by
    Gatys et al. (2015) (https://arxiv.org/abs/1508.06576),
    and using a Keras pre-trained VGG19 neural network and the Adam optimizer.
    """
    style_layers = {'block1_conv1': 0.2,
                    'block2_conv1': 0.2,
                    'block3_conv1': 0.2,
                    'block4_conv1': 0.2,
                    'block5_conv1': 0.2}

    def __init__(self, alpha: float = 5., beta: float = 100.,
                 gamma: float = 1e-3, dir_path: Optional[str] = None):
        """Initialize with the default loss weights and the directory
        path that will be used to write the outputs.

        Parameters
        ----------
        alpha
            Weight associated with content image loss (default = 5)
        beta
            Weight associated with style image loss (default = 100)
        gamma
            Weight associated with image variation loss (default = 1e-3)
        dir_path
            Directory where generated images will be written
            (default = current working directory)
        """
        self.dir_path = dir_path or os.getcwd()
        self.run_dir = None
        self.default_alpha = alpha
        self.default_beta = beta
        self.default_gamma = gamma
        # Image parameters
        self.img_height = None
        self.img_width = None
        self.color_channels = None
        self.input_shape = None
        # Initialize images
        self.img_content = None
        self.img_style = None
        self.input_content = None
        self.input_style = None
        self.input_generated = None
        # Tensorflow stuff
        self.session = None
        self.kmodel = None
        # Cost functions
        self.j_style = None
        self.j_content = None
        self.j_var = None
        self.j_total = None

    def load_content_image(self, fpath: str, content_size: int = 512,
                           interpolation: str = 'bilinear'):
        """Load content image, and scale it. Scaling the style image will
        affect the artistic features transfered to the generated image.

        Parameters
        ----------
        fpath
            Path of style image
        content_size
            Height to be used for scaling the content image
            (default = 512)
        interpolation
            Interpolation scheme to use when rescaling image
            (default = bilinear)
        """
        self.img_content = Image.from_fp(fpath, label='content')
        # Resize content image
        n_h, n_w, n_c = self.img_content.data.shape
        n_h_scaled = content_size
        n_w_scaled = int(n_w * content_size / n_h)
        self.img_content.resize(n_w_scaled, n_h_scaled,
                                resample=interpolation)
        model_input = np.expand_dims(deepcopy(self.img_content.data), axis=0)
        self.input_content = preprocess_input(model_input)
        # Save the content image shape for loading model
        self.img_height, self.img_width, self.color_channels = \
            self.img_content.data.shape
        self.input_shape = (1, self.img_height, self.img_width,
                            self.color_channels)

    def load_style_image(self, fpath: str, scale: float = 1.0,
                         interpolation: str = 'bilinear'):
        """Load style image, and scale it. Scaling the style image will
        affect the artistic features transfered to the generated image.

        Parameters
        ----------
        fpath
            Path of style image
        scale
            Scaling factor of style image (default = 1., no scaling)
        interpolation
            Interpolation scheme to use when rescaling image
            (default = bilinear)
        """
        self.img_style = Image.from_fp(fpath, label='content')
        if scale != 1.0:
            n_h, n_w, n_c = self.img_style.data.shape
            n_h_scaled = int(scale * n_h)
            n_w_scaled = int(scale * n_w)
            self.img_style.resize(n_w_scaled, n_h_scaled,
                                  resample=interpolation)
        model_input = np.expand_dims(deepcopy(self.img_style.data), axis=0)
        self.input_style = preprocess_input(model_input)

    def run(self, num_iterations: int = 1000, learning_rate: float = 2.0,
            alpha: Optional[float] = None, beta: Optional[float] = None,
            gamma: Optional[float] = None,
            noise_low: float = -20, noise_high: float = 20,
            noise_ratio: float = 0.6, write_steps: int = 20,
            style_layers: Optional[dict] = None,
            progress_bar: DeltaGenerator = None):
        """Run style transfer optimization that will generate the new image
        build from content and style images.

        Parameters
        ----------
        num_iterations
            Number of optimization steps to run (default = 1000)
        learning_rate
            Learning rate to use for the Adam optimizer (default = 2.0)
        alpha
            Weight associated with content image loss (default = default_alpha)
        beta
            Weight associated with style image loss (default = default_beta)
        gamma
            Weight associated with image variation loss
            (default = default_gamma)
        noise_low
            Min of random values to use when first creating the generated image
            from noise (default = -20)
        noise_high
            Max of random values to use when first creating the generated image
            from noise (default = 20)
        noise_ratio
            When first creating the generated image, ratio of amount coming
            from noise to amount coming from content image (default = 0.6)
        write_steps
            A generated image will be written every time after this number
            of iterations (default = 20)
        style_layers
            The weight to give to each VGG19 layer used in the style loss
            calculation (default = all equal weights)
        progress_bar
            Streamlit progress bar to be updated when training
            (default = None)
        """
        # Create a directory specific to this run
        self.run_dir = os.path.join(
            self.dir_path, dt.datetime.now().strftime("%Y%m%d-%H%M%S"))
        if not os.path.exists(self.run_dir):
            os.makedirs(self.run_dir)

        # Initialize Keras model to be used
        print('...creating Keras model to be trained')
        self.kmodel = self._init_model(self.img_content.data.shape)
        # For name formatting
        n_digits = len(str(abs(num_iterations)))
        # Input weights
        alpha = alpha or self.default_alpha
        beta = beta or self.default_beta
        gamma = self.default_gamma if gamma is None else gamma
        # Session: need to get session from Keras, because it contains
        # the pre-trained weight initialization for the model variables
        self.session = K.get_session()

        # get parameters for style cost
        style_layers = style_layers or self.style_layers
        # build total cost function
        print('...building cost function')
        self._build_cost_function(self.session, alpha, beta, gamma,
                                  style_layers)
        # Define optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate)
        # Define training step, and specify input variable explicitely,
        # otherwise will train all the variables in the graph
        train_step = optimizer.minimize(self.j_total,
                                        var_list=[self.kmodel.input])
        # All variables need to be initialized because we rely on tensorflow,
        # but we don't want to lose the pretrained weights
        self._custom_global_variable_initialization(self.session)

        # Assign generated image to variable
        self._generate_noise_image(low=noise_low, high=noise_high,
                                   noise_ratio=noise_ratio)
        K.set_value(self.kmodel.input, self.input_generated)

        print('...training')
        # Start progress bar
        if progress_bar is not None:
            progress_bar.progress(0)
        for i in tqdm(range(num_iterations)):
            # Update progress bar if any
            if progress_bar is not None:
                progress_bar.progress((i + 1) / num_iterations)
            # Print every `write_steps` iteration.
            if i % write_steps == 0:
                jt, jc, js, jv = self.session.run(
                    [self.j_total, self.j_content, self.j_style, self.j_var])
                print("Iteration: {}".format(i))
                print("total cost = {}".format(jt))
                print("content cost = {}".format(jc))
                print("style cost = {}".format(js))
                print("variation cost = {}".format(jv))

                # save current generated image in `dir_path`
                fname = str(i).zfill(n_digits) + '.png'
                fpath = os.path.join(self.run_dir, fname)
                img = self._model_input_to_image(self.input_generated, label=i)
                img.save(fpath)

            # Run minimization step
            self.session.run(train_step)
            # Get generated image
            self.input_generated = self.session.run(self.kmodel.input)

        print('...done training')
        # save last generated image
        fpath = os.path.join(self.run_dir, 'image_generated.png')
        img = self._model_input_to_image(self.input_generated, label='final')
        img.save(fpath)

    @property
    def img_generated(self) -> Image:
        """Deprocess the generated input into the generated image"""
        return self._model_input_to_image(self.input_generated)

    def _init_model(self, input_shape: Tuple) -> Sequential:
        """Create keras model used for neural style transfer
        - import VGG19 model with pre-trained weights
        - remove end layers if existing (no top)
        - change max pool layers into avg pool layers
        - set weights as non trainable (doesn't really matter since we'll use
        tensorflow directly for training)
        - use variable as input
        - use input shape from given image

        Check out original model build here:
        https://github.com/keras-team/keras-applications/blob/master/keras_applications/vgg19.py

        Parameters
        ----------
        input_shape
            Input shape of the image that will be used as input to the model:
            (input height, input width, number of channels)

        Returns
        -------
        model
            Keras sequential model
        """
        # Get pretrained keras model: if using docker image, the weights
        # will be saved inside the mounted volume (since it takes a while
        # to download)
        kmodel = VGG19(include_top=False, weights='imagenet',
                       input_tensor=None, input_shape=input_shape)
        # Build new model from pretrained keras model
        new_kmodel = Sequential(name='VGG19_nst')
        # Create variable input tensor
        var = K.zeros(shape=(1,) + input_shape)
        input_var = InputLayer(input_tensor=var)
        input_var.trainable = True
        # Add input layer
        new_kmodel.add(input_var)
        # Add pretrained layers
        i_pool = 1
        for layer in kmodel.layers[1:]:
            if 'pool' in layer.name:
                name = 'block%i_avgpool' % i_pool
                layer = AveragePooling2D((2, 2), strides=(2, 2),
                                         padding='same', name=name)
                i_pool += 1
            else:
                # clear all the nodes from the layer before adding it to model
                layer._inbound_nodes = []
                layer._outbound_nodes = []
            # make all layers non trainable
            layer.trainable = False
            new_kmodel.add(layer)

        # build: need to rebuild model wo input layer since already exist
        new_kmodel.build(None)

        return new_kmodel

    def _custom_global_variable_initialization(self, session: tf.Session):
        """Custom variable initialization:
        the problem is that tensorflow requires all variables to
        be initialized, but that would erase all the pre-trained values
        obtained with Keras. So here we save their values before
        initialization, and then reassign it afterwards.

        Parameters
        ----------
        session
            Tensorflow session where graph resides
        """
        # There are other hidden variables in the model that need to
        # be initialized, and we can't get their values
        all_variables = tf.global_variables()
        model_variables = self.kmodel.weights
        # save variable values
        list_values = []
        for var in model_variables:
            list_values.append(session.run(var))
        # run initialization on that variable only
        init_op = tf.initialize_variables(all_variables)
        session.run(init_op)
        # then assign previous value again
        for idx, var in enumerate(model_variables):
            K.set_value(var, list_values[idx])

    def _generate_noise_image(self, low: int = -20, high: int = 20,
                              noise_ratio: float = 0.6):
        """Generate noise image from combination of random noise and
        content image pixel values.

        Parameters
        ----------
        noise_low
            Min of random values to use when first creating the generated image
            from noise (default = -20)
        noise_high
            Max of random values to use when first creating the generated image
            from noise (default = 20)
        noise_ratio
            When first creating the generated image, ratio of amount coming
            from noise to amount coming from content image (default = 0.6)
        """
        shape_img = (1, self.img_height, self.img_width, self.color_channels)
        noise_data = np.random.uniform(low, high, shape_img).astype('float32')
        input_data = (noise_data * noise_ratio
                      + self.input_content * (1 - noise_ratio))
        self.input_generated = input_data

    def _build_cost_function(self, session: tf.Session, alpha: float,
                             beta: float, gamma: float, style_layers: dict):
        """Build total cost function, which is the weighted sum of the
        content loss, the style loss, and the variation loss functions.
        This will create the loss functions and sum them up.

        Parameters
        ----------
        session
            Tensorflow session where the graph resides
        alpha
            Weight associated with content image loss
        beta
            Weight associated with style image loss
        gamma
            Weight associated with image variation loss
        style_layers
            The weight to give to each VGG19 layer used in the style loss
            calculation
        """
        # Calculate costs
        self.j_content = self._compute_content_cost(self.input_content,
                                                    session=session)
        self.j_style = self._compute_style_cost(self.input_style,
                                                style_layers, session=session)
        self.j_var = self._compute_variation_cost(session)
        # Get total weighted cost
        self.j_total = self._total_cost(self.j_content, self.j_style,
                                        self.j_var, alpha, beta, gamma)

    def _model_input_to_image(self, input_data: np.ndarray,
                              label: Optional[str] = None) -> Image:
        """Undo all the preprocessing done for modeling and return image
        Reversing this:
        https://github.com/keras-team/keras-applications/blob/master/keras_applications/imagenet_utils.py#L60-L61

        Parameters
        ----------
        input_data
            Image input data used for model, of shape
            (1, image height, image width, number of channels)
        label
            Optional label for created image (default = None)
        """
        x = deepcopy(input_data)
        # Add what was subtracted
        mean = [103.939, 116.779, 123.68]
        x[..., 0] += mean[0]
        x[..., 1] += mean[1]
        x[..., 2] += mean[2]
        # BGR -> RGB
        x = x[..., ::-1]
        # Images need integer
        x = np.clip(x[0, ...], 0, 255).astype('uint8')
        return Image(x, label)

    @staticmethod
    def _total_cost(j_content: tf.Tensor, j_style: tf.Tensor, j_var: tf.Tensor,
                    alpha: float, beta: float, gamma: float) -> tf.Tensor:
        """Compute total weighted averaged cost of the content, style, and
        variation losses.

        Parameters
        ----------
        j_content
            Content loss
        j_style
            Style loss
        j_var
            Variation loss
        alpha
            Weight associated with content image loss
        beta
            Weight associated with style image loss
        gamma
            Weight associated with image variation loss

        Returns
        -------
        total loss
            Weighted sum of all losses
        """
        return alpha * j_content + beta * j_style + gamma * j_var

    def _compute_content_cost(self, input_content: np.ndarray,
                              session: Optional[tf.Session] = None
                              ) -> tf.Tensor:
        """
        Compute content loss

        Parameters
        ----------
        input_content
            Model input of content image
            Shape = (1, image height, image width, number of channels)
        session
            Session where graph resides (default = NST object session)

        Returns
        -------
        content loss
        """

        session = session or self.session
        # get model outputs
        output = self.kmodel.get_layer('block4_conv2').output
        # Get activations from content inputs
        K.set_value(self.kmodel.input, input_content)
        a_c = session.run(output)
        # Get symbolic activation for future updates of generated image
        a_g = output
        # Get shapes of activation
        _, n_h, n_w, n_c = a_c.shape
        # compute content loss
        j_content = 1. / (4 * n_h * n_w * n_c) * K.sum(K.square(a_c - a_g))

        return j_content

    def _compute_style_cost(self, input_style: np.ndarray, style_layers: dict,
                            session: tf.Session = None) -> tf.Tensor:
        """
        Compute the total style loss

        Parameters
        ----------
        input_style
            Model input for style image
        style_layers
            The weight to give to each VGG19 layer used in the style loss
            calculation
        session
            Session where graph resides (default = NST object session)

        Returns
        -------
        total style loss
        """

        session = session or self.session
        # initialize the overall style cost
        j_style = 0
        # Create temporary model to calculate activations for style image
        kmodel_style = self._init_model(self.img_style.data.shape)
        # Sum up losses for each layer
        for layer_name, coeff in style_layers.items():
            # Set input variable to style image inputs
            K.set_value(kmodel_style.input, input_style)
            # Set a_s to be the hidden layer activation from the layer we
            # have selected, by running the session on layer.output
            a_s = session.run(kmodel_style.get_layer(layer_name).output)
            # Get symbolic a_g
            a_g = self.kmodel.get_layer(layer_name).output
            # Compute style_cost for the current layer
            j_style_layer = self._compute_layer_style_cost(a_s, a_g, session)
            # Add coeff * J_style_layer of this layer to overall style cost
            j_style += coeff * j_style_layer

        return j_style

    @staticmethod
    def _compute_layer_style_cost(a_s: np.ndarray, a_g: tf.Tensor,
                                  session: tf.Session):
        """
        Calculate layer-specific style loss using style image and generated
        image activations. Style image and generated image number of channels
        must be equal.

        Parameters
        ----------
        a_s
            Activations of style image for layer
            Shape = (1, style img height, style img width, n channels)
        a_g
            Activations of generated image for layer
            Shape = (1, generated img height, generated img width, n channels)
        session
            Session where graph resides

        Returns
        -------
        Layer-specific style loss
        """

        # Retrieve dimensions from a_g (≈1 line)
        _, n_h, n_w, n_c = K.shape(a_g).eval(session=session)

        # Reshape the images to have them of shape (n_c, n_h*n_w) (≈2 lines)
        a_s = K.reshape(K.permute_dimensions(a_s, (3, 0, 1, 2)), (n_c, -1))
        a_g = K.reshape(K.permute_dimensions(a_g, (3, 0, 1, 2)), (n_c, -1))

        # Computing gram_matrices for both images S and G (≈2 lines)
        gs = NeuralStyleTransfer._gram_matrix(a_s)
        gg = NeuralStyleTransfer._gram_matrix(a_g)

        # Computing the loss (≈1 line)
        j_style_layer = 1 / (2 * n_c * n_w * n_h)**2 * K.sum(K.square(gs - gg))

        return j_style_layer

    def _compute_variation_cost(self, session: tf.Session) -> tf.Tensor:
        """Compute variation loss of Keras model input to keep the image locally
        coherent.

        Inspired by:
        https://github.com/keras-team/keras/blob/master/examples/neural_style_transfer.py

        Parameters
        ----------
        session
            Session where graph resides

        Returns
        -------
        Variation loss
        """
        # Get shape of input var
        input_var = self.kmodel.input
        _, n_h, n_w, n_c = K.shape(input_var).eval(session=session)
        a = K.square(
            input_var[:, :n_h - 1, :n_w - 1, :]
            - input_var[:, 1:, :n_w - 1, :])
        b = K.square(
            input_var[:, :n_h - 1, :n_w - 1, :]
            - input_var[:, :n_h - 1, 1:, :])
        return K.sum(K.pow(a + b, 1.25))

    @staticmethod
    def _gram_matrix(a: tf.Tensor) -> tf.Tensor:
        """
        Calculate gram matrix of given tensor.

        Parameters
        ----------
        a
            Input tensor for which to calculate gram matrix

        Returns
        -------
        ga
            Gram matrix of input tensor
        """
        ga = K.dot(a, K.transpose(a))
        return ga
