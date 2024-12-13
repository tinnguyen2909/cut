from .base_options import BaseOptions


class ConvertOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument("--onnx_model_output", type=str,
                            default="onnx_model.onnx", help="onnx model output path")
        parser.add_argument('--results_dir', type=str,
                            default='./results/', help='saves results here.')
        parser.add_argument('--aspect_ratio', type=float,
                            default=1.0, help='aspect ratio of result images')
        parser.add_argument('--phase', type=str,
                            default='test', help='train, val, test, etc')
        # Dropout and Batchnorm has different behavioir during training and test.
        parser.add_argument('--eval', action='store_true',
                            help='use eval mode during test time.')
        parser.add_argument('--num_test', type=int, default=50,
                            help='how many test images to run')
        # parser.add_argument('--path_A', type=str, help='path to A', default="")
        # parser.add_argument('--path_B', type=str, help='path to B', default="")
        # parser.add_argument('--ext_A', type=str, default='jpg',
        #                     help='extension of A (eg: jpg, png)')
        # parser.add_argument('--ext_B', type=str, default='png',
        #                     help='extension of B (eg: jpg, png)')
        # rewrite devalue values
        # parser.set_defaults(model='test')
        # To avoid cropping, the load_size should be the same as crop_size
        parser.set_defaults(load_size=parser.get_default('crop_size'))
        self.isTrain = False
        return parser
