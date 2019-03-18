class ModelConfig:
    input_dim = {'brain2D': [128, 96, 1], 'UT50k': [96, 128, 1]}     # Image Dimension
    output_dim = {'brain2D': [128, 96, 1], 'UT50k': [96, 128, 3]}
    # n_train = {'brain2D': 726, 'UT50k': 29442}
    n_train = {'brain2D': 726, 'UT50k': 1000}
    # n_test = {'brain2D': 80, 'UT50k': 9805}                             # Validation Epoch Size
    n_test = {'brain2D': 80, 'UT50k': 200}
    n_y = {'brain2D': 1, 'UT50k': 13}                                   # Dimension of external information
    attributes = {'brain2D': "age", 'UT50k': ["bin_class", "bin_gender", "bin_material"]}

    data_dir = {'brain2D': './data_loaders/Brain_img/2D/', 'UT50k': './data_loaders/UT50k'}
    sample_dir = {'brain2D': './data_loaders/brain2D_sample_', 'UT50k': './data_loaders/UT50k_sample_'}