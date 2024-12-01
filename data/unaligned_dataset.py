import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import util.util as util


class UnalignedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        # self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        # self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'

        # if opt.phase == "test" and not os.path.exists(self.dir_A) \
        #    and os.path.exists(os.path.join(opt.dataroot, "valA")):
        #     self.dir_A = os.path.join(opt.dataroot, "valA")
        #     self.dir_B = os.path.join(opt.dataroot, "valB")
        path_As = [p.strip() for p in opt.path_A.split(',')]
        path_Bs = [p.strip() for p in opt.path_B.split(',')]
        self.A_paths = []
        self.B_paths = []
        self.map = {}
        for path_A, path_B in zip(path_As, path_Bs):
            self.map[os.path.abspath(path_A)] = path_B
            self.A_paths += sorted(make_dataset(path_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
            self.B_paths += sorted(make_dataset(path_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        A_img = Image.open(A_path)

        B_img = Image.open(B_path)
        if A_img.mode == "RGBA":
            rgb_img = Image.new("RGB", A_img.size, (255, 255, 255))
            rgb_img.paste(A_img, mask=A_img.split()[3])
            A_img = rgb_img
        if B_img.mode == "RGBA":
            rgb_img = Image.new("RGB", B_img.size, (255, 255, 255))
            rgb_img.paste(B_img, mask=B_img.split()[3])
            B_img = rgb_img
        B_width, B_height = B_img.size
        if B_width != B_height:
            max_side = max(B_width, B_height)
            square_img = Image.new("RGB", (max_side, max_side), (255, 255, 255))
            square_img.paste(B_img, (int((max_side - B_width) // 2), int((max_side - B_height) // 2)))
            B_img = square_img
        A_width, A_height = A_img.size
        if A_width != A_height:
            max_side = max(A_width, A_height)
            square_img = Image.new("RGB", (max_side, max_side), (255, 255, 255))
            square_img.paste(A_img, (int((max_side - A_width) // 2), int((max_side - A_height) // 2)))
            A_img = square_img
        if hasattr(self.opt, "remove_bg_A") and self.opt.remove_bg_A:
            A_img_filename = os.path.basename(A_path)
            A_img_png_filename = os.path.splitext(A_img_filename)[0] + '.webp'
            dir_B = self.map[os.path.dirname(os.path.abspath(A_path))]
            if os.path.isfile(os.path.join(dir_B, A_img_png_filename)):
                A_img_png = Image.open(os.path.join(dir_B, A_img_png_filename))
                if A_img_png.width != A_img.width or A_img_png.height != A_img.height:
                    A_img_png = A_img_png.resize(A_img.size, Image.Resampling.LANCZOS)
                mask = A_img_png.split()[3]
                rgb_img = Image.new("RGB", A_img.size, (255, 255, 255))
                rgb_img.paste(A_img, mask=mask)
                A_img = rgb_img
        A_img = A_img.convert("RGB")
        B_img = B_img.convert("RGB")
        if hasattr(self.opt, "enable_rotation") and self.opt.enable_rotation:
            if self.opt.rotation_probability > random.randint(0, 100):
                angle = random.uniform(-90, 90)
                B_img = B_img.rotate(angle, expand=True, fillcolor=(255, 255, 255))
            if self.opt.rotation_probability > random.randint(0, 100):
                angle = random.uniform(-90, 90)
                A_img = A_img.rotate(angle, expand=True, fillcolor=(255, 255, 255))

        # Apply image transformation
        # For CUT/FastCUT mode, if in finetuning phase (learning rate is decaying),
        # do not perform resize-crop data augmentation of CycleGAN.
        is_finetuning = self.opt.isTrain and self.current_epoch > self.opt.n_epochs
        modified_opt = util.copyconf(self.opt, load_size=self.opt.crop_size if is_finetuning else self.opt.load_size)
        transform = get_transform(modified_opt)
        A = transform(A_img)
        B = transform(B_img)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
