"""Streamlit app prototype to run neural style transfer"""

import streamlit as st
from nst.main import NeuralStyleTransfer
from nst.utils import Image
import os


# Define output directory
dir_path = 'outputs'
nst = NeuralStyleTransfer(dir_path=dir_path)


def main():
    """Function that will contain the app"""

    st.title('Neural Style Transfer app')

    ####################
    # --- Build side bar
    ####################
    st.sidebar.title('Transfer parameters')

    # Number of iterations
    st.sidebar.subheader('Image resizing')
    content_size = st.sidebar.slider('Content image size',
                                     100, 2000, 512)
    style_scale = st.sidebar.slider('Style image scaling', 0.1, 2., 0.5)

    # Number of iterations
    st.sidebar.subheader('High level hyper parameters')
    n_iter = st.sidebar.number_input('Number of iterations',
                                     min_value=1, value=1000)
    lr = st.sidebar.number_input('Learning rate', value=2.0)
    write_steps = st.sidebar.number_input(
        'Write image when iteration is multiple of', min_value=2,
        value=100)

    # Noise generation parameters
    st.sidebar.subheader('Noise image generation')
    noise_ratio = st.sidebar.slider(
        'Noise ratio: between content and randomness', 0., 1., 0.6)

    # Style loss weights
    st.sidebar.subheader('Total cost weights')
    alpha = st.sidebar.number_input('Content weight', min_value=0., value=5.)
    beta = st.sidebar.number_input('Style weight', min_value=0., value=100.)
    gamma = st.sidebar.number_input('Variation weight', min_value=0., value=0.)

    # Style loss weights
    st.sidebar.subheader('Style loss: layer weights')
    c_1 = st.sidebar.slider('Block1 conv1', 0., 1., 0.2)
    c_2 = st.sidebar.slider('Block2 conv1', 0., 1., 0.2)
    c_3 = st.sidebar.slider('Block3 conv1', 0., 1., 0.2)
    c_4 = st.sidebar.slider('Block4 conv1', 0., 1., 0.2)
    c_5 = st.sidebar.slider('Block5 conv1', 0., 1., 0.2)
    # Build style layers
    style_layers = {'block1_conv1': c_1, 'block2_conv1': c_2,
                    'block3_conv1': c_3, 'block4_conv1': c_4,
                    'block5_conv1': c_5}

    ########################
    # --- Build main content
    ########################
    # Get list of input files
    list_input_files = [fname for fname in os.listdir('inputs')
                        if os.path.isfile(os.path.join('inputs', fname))]
    # Select content image
    st.subheader('Please select a content image')
    fname_content = st.selectbox('Select a content image', list_input_files)
    fp_content = os.path.join('inputs', fname_content)
    # Put into workflow if path valid
    if os.path.isfile(fp_content):
        nst.load_content_image(fp_content, content_size=content_size)
        st.text('Content image loaded!')
        st.image(nst.img_content.as_pil, caption='content image', width=300)
    else:
        st.text('Content image file path invalid')

    # Load style image
    st.subheader('Please select a style image')
    fname_style = st.selectbox('Select a style image', list_input_files)
    fp_style = os.path.join('inputs', fname_style)
    # Put into workflow if path valid
    if os.path.isfile(fp_style):
        nst.load_style_image(fp_style, scale=style_scale)
        st.text('Style image loaded!')
        st.image(nst.img_style.as_pil, caption='style image', width=300)
    else:
        st.text('Style image file path invalid')

    # Training
    st.subheader('Now transfer style to your content image')
    st.write("Select the parameters in the sidebar to modulate the style "
             "transfer to your liking, then click on the train button!")
    train_button = st.button('Run style transfer')
    run_dir = None
    if train_button:
        st.text('starting training...')
        nst.run(num_iterations=n_iter, learning_rate=lr,
                alpha=alpha, beta=beta, gamma=gamma, noise_ratio=noise_ratio,
                write_steps=write_steps, style_layers=style_layers,
                progress_bar=st.progress(0))
        run_dir = nst.run_dir
        st.image(nst.img_generated.as_pil, caption='generated image',
                 width=500)
        st.text('training done.')

    # Explore created images
    st.subheader('Explore outputs')
    list_dirs = sorted([dirname for dirname in os.listdir(dir_path)
                        if os.path.isdir(os.path.join(dir_path, dirname))])
    output_dir = st.selectbox('Select an output directory', list_dirs)
    if output_dir:
        run_dir = os.path.join(dir_path, output_dir)
        list_files = sorted([fname for fname in os.listdir(run_dir)
                             if (os.path.isfile(os.path.join(run_dir, fname))
                                 and (fname[-3:] == 'png'))])
        fname = st.selectbox('Select a result file to show', list_files)
        if fname:
            # show image
            img = Image.from_fp(os.path.join(run_dir, fname))
            st.image(img.as_pil, caption=fname, width=300)


if __name__ == "__main__":
    main()
