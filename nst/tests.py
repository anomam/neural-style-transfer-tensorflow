from .main import NeuralStyleTransfer
import tensorflow as tf
from keras import backend as K
from keras.models import Sequential
import numpy as np
import pytest
import os


TEST_DIR = os.path.dirname(__file__)
TEST_DATA_DIR = os.path.join(TEST_DIR, 'test_data')


@pytest.fixture(scope='function')
def nst_with_pix():
    """NST object with loaded test pictures"""
    nst = NeuralStyleTransfer()
    test_content_fpath = os.path.join(TEST_DATA_DIR, 'Louvre.jpg')
    test_style_fpath = os.path.join(TEST_DATA_DIR, 'Miro-1.jpg')
    nst.load_content_image(test_content_fpath)
    nst.load_style_image(test_style_fpath, scale=0.2)
    # load model
    nst.kmodel = nst._init_model(nst.img_content.data.shape)
    yield nst


def test_load_content_image():
    """Check that can load content image correctly,
    and resize it if wanted"""
    nst = NeuralStyleTransfer()
    test_content_fpath = os.path.join(TEST_DATA_DIR, 'Louvre.jpg')
    # Load without resizing
    nst.load_content_image(test_content_fpath)
    assert nst.img_content.data.shape == (512, 768, 3)
    np.testing.assert_array_equal(nst.img_content.data.shape,
                                  nst.input_content.shape[1:])
    # Load with resizing
    nst.load_content_image(test_content_fpath, content_size=400)
    assert nst.img_content.data.shape == (400, 600, 3)
    np.testing.assert_array_equal(nst.img_content.data.shape,
                                  nst.input_content.shape[1:])


def test_load_style_image():
    """Check that can load style image correctly,
    and scale it if wanted"""
    nst = NeuralStyleTransfer()
    test_style_fpath = os.path.join(TEST_DATA_DIR, 'Miro-1.jpg')
    # Load without resizing
    nst.load_style_image(test_style_fpath)
    assert nst.img_style.data.shape == (1067, 1067, 3)
    np.testing.assert_array_equal(nst.img_style.data.shape,
                                  nst.input_style.shape[1:])
    # Load with resizing
    nst.load_style_image(test_style_fpath, scale=0.5)
    assert nst.img_style.data.shape == (533, 533, 3)
    np.testing.assert_array_equal(nst.img_style.data.shape,
                                  nst.input_style.shape[1:])


def test_model_layers_single_inputs(nst_with_pix):
    """Check that intermediate keras model layer has single input"""
    nst = nst_with_pix
    kmodel = nst.kmodel
    assert isinstance(kmodel, Sequential)
    # Check that input for the whole model
    assert K.is_tensor(kmodel.input)
    assert kmodel.built
    # Check that single input for intermediate layer
    layer = kmodel.get_layer('block2_conv2')
    assert K.is_tensor(layer.input)


def test_compute_content_cost(nst_with_pix):
    """Check that calculating content cost correctly"""
    nst = nst_with_pix
    sess = K.get_session()
    j_content = nst._compute_content_cost(nst.input_content, session=sess)
    assert K.is_tensor(j_content)
    assert K.shape(j_content).eval(session=sess).shape[0] == 0
    np.testing.assert_allclose(j_content.eval(session=sess), 0.)


def test_compute_gram_matrix():
    """Check that computing gram matrix correctly"""
    # Get session
    sess = K.get_session()
    # Use placeholder with known value
    a = K.placeholder(shape=(3, 2))
    a_value = np.array([[-1.683445, 1.8942857],
                        [4.189092, 1.3846824],
                        [3.8925915, 2.3524866]])
    # Get value and compare to expected
    ga = NeuralStyleTransfer._gram_matrix(a)
    expected_ga = [[6.42230511, -4.42912197, -2.09668207],
                   [-4.42912197, 19.46583748, 19.56387138],
                   [-2.09668207, 19.56387138, 20.6864624]]
    np.testing.assert_allclose(sess.run(ga, feed_dict={a: a_value}),
                               expected_ga, atol=0, rtol=1e-6)


def test_compute_layer_style_cost():
    """Check that calculating layer style cost correctly
    - size of style will be different from one of generated matrix
    - except for number of channels
    """
    # Initializing
    sess = tf.Session()
    # Style input generated randomly
    a_s = np.array([[[[-1.683445, 1.8942857, 4.189092],
                      [1.3846824, 3.8925915, 2.3524866]],

                     [[-1.9202449, 4.6461368, -1.0375276],
                      [4.899456, -7.5360813, 3.4091651]]]],
                   dtype='float32')
    # Create a_g placeholder and its future value
    a_g = K.placeholder(shape=[1, 3, 3, 3])
    ag_value = np.array([[[[-0.39043474, -4.965909, -5.387548],
                           [4.572505, 1.1961036, 5.0099816],
                           [1.7304354, -0.13603461, -0.7514645]],

                          [[-3.0110965, 1.0130516, 7.4561086],
                           [0.51901615, -0.23328066, -0.8221154],
                           [0.69788367, 1.5624137, 0.11127031]],

                          [[3.7990131, -0.5115707, -5.364818],
                           [-4.8868036, -1.1914248, -0.12090659],
                           [7.0109277, -1.2259245, 4.2369]]]])
    # print(repr(sess.run(a_g)))
    j_layer_style = NeuralStyleTransfer._compute_layer_style_cost(a_s,
                                                                  a_g, sess)
    assert K.is_tensor(j_layer_style)
    np.testing.assert_allclose(sess.run(j_layer_style,
                                        feed_dict={a_g: ag_value}), 12.413997)


def test_compute_style_cost(nst_with_pix):
    """Check that the style cost computation works properly"""
    nst = nst_with_pix
    sess = K.get_session()
    # For testing, set generated image to content image
    K.set_value(nst.kmodel.input, nst.input_content)
    j_style = nst._compute_style_cost(nst.input_style, nst.style_layers,
                                      session=sess)
    # Run checks
    assert K.is_tensor(j_style)
    assert K.shape(j_style).eval(session=sess).shape[0] == 0
    np.testing.assert_allclose(sess.run(j_style), 96948430.)


def test_compute_variation_cost(nst_with_pix):
    """Check that the variation cost computation works properly"""
    nst = nst_with_pix
    sess = K.get_session()
    j_var = nst._compute_variation_cost(sess)
    assert K.is_tensor(j_var)
    assert K.shape(j_var).eval(session=sess).shape[0] == 0
    np.testing.assert_allclose(j_var.eval(session=sess), 0.)


def test_model_input_to_image():
    """Check that successfully transform model input back to image"""
    nst = NeuralStyleTransfer()
    test_img_fpath = os.path.join(TEST_DATA_DIR, 'Louvre.jpg')
    nst.load_content_image(test_img_fpath)
    img = nst._model_input_to_image(nst.input_content)
    np.testing.assert_allclose(nst.img_content.data, img.data)


def test_build_cost_function(nst_with_pix):
    """Check that total cost calculated correctly"""
    # Initialize
    nst = nst_with_pix
    session = K.get_session()
    alpha = 10
    beta = 30
    gamma = 30
    # Build total cost
    nst._build_cost_function(session, alpha, beta, gamma,
                             nst.style_layers)
    assert K.shape(nst.j_total).eval(session=session).shape[0] == 0
    np.testing.assert_allclose(session.run(nst.j_total), 109412940000.0)


def test_kmodel_with_variable_input(nst_with_pix):
    """Want to check that:
    - can use variable as input
    - that output change depending on input var
    - that the intermediate output tensor values change when changing input var
    """
    # Create nst
    nst = nst_with_pix
    # Create a model using variables as input
    kmodel = nst.kmodel
    assert isinstance(kmodel, Sequential)  # make sure could create model
    # somehow trainable weights not showing up
    assert len(kmodel.trainable_weights) == 0
    assert isinstance(kmodel.input, tf.Variable)  # input should be Variable
    assert len(kmodel.layers) == 21  # all the pretrained layers should be here

    # Catch intermediate output
    inter_output = kmodel.get_layer('block2_conv2').output
    # The input should be a variable
    assert isinstance(kmodel.input, tf.Variable)
    # Check model final output values
    sess = K.get_session()
    # Check final and intermediate outputs for case with variable as zeros
    np_out_zero = sess.run(kmodel.output)
    np_inter_zero = sess.run(inter_output)
    RTOL = 1e-5
    ATOL = 0.
    np.testing.assert_allclose(np_out_zero[0][0][0][0], 0.4045257,
                               atol=ATOL, rtol=RTOL)
    np.testing.assert_allclose(np_inter_zero[0][0][0][0], 0.20578316,
                               atol=ATOL, rtol=RTOL)

    # Check that value consistent when setting input var to one
    K.set_value(kmodel.input, np.ones((1,) + nst.img_content.data.shape))
    np_out_one = sess.run(kmodel.output)
    np_inter_one = sess.run(inter_output)
    np.testing.assert_allclose(np_out_one[0][0][0][0], 0.38975614,
                               atol=ATOL, rtol=RTOL)
    np.testing.assert_allclose(np_inter_one[0][0][0][0], 0.)  # changed
    np.testing.assert_allclose(np_inter_one[0][0][0][1], 2.1130981,
                               atol=ATOL, rtol=RTOL)

    # Check that predicts consistently values when using image input
    # Now instead of using model.predict() method, need to assign value to var
    K.set_value(kmodel.input, nst.input_content)
    enc_content = sess.run(kmodel.output)
    enc_inter_content = sess.run(inter_output)

    # check that final output worked
    assert enc_content.shape == (1, 16, 24, 512)
    np.testing.assert_allclose(enc_content[0][0][0][-2], 2.3606608,
                               atol=ATOL, rtol=RTOL)
    assert isinstance(enc_content, np.ndarray)
    np.testing.assert_allclose(enc_inter_content[0][0][0][1], 166.33893,
                               atol=ATOL, rtol=RTOL)
    # Even after prediction, intermediate layer should have single node
    assert len(kmodel.layers[4]._inbound_nodes) == 1


def test_run(nst_with_pix):
    """Check that training runs correctly:
    - make sure pre-trained weights not changing because of initialization
    - make sure that the pre-trained weights are not changing during training
    """
    nst = nst_with_pix

    # Check that initalization doesn't change pre-trained weights
    value_before_init = nst.kmodel.layers[0].get_weights()[0][0][0][0]
    nst.run(num_iterations=0)
    value_after_init = nst.kmodel.layers[0].get_weights()[0][0][0][0]
    # FIXME: this test currently fails...
    np.testing.assert_allclose(value_before_init, value_after_init)

    # Make sure they don't change after training either
    value_before_training = nst.kmodel.layers[0].get_weights()[0][0][0][0]
    nst.run(num_iterations=1)
    value_after_training = nst.kmodel.layers[0].get_weights()[0][0][0][0]
    # FIXME: this test currently fails...
    np.testing.assert_allclose(value_before_training, value_after_training)
