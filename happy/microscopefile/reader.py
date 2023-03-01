import os
from abc import ABC, abstractmethod

import bioformats
import javabridge
import numpy as np
import openslide
import pyvips


class Reader(ABC):
    def __init__(self, slide_path, lvl_x):
        self.slide_path = slide_path
        self.lvl_x = lvl_x

    # Tries to get whole slide image pixel size from metadata on image
    @abstractmethod
    def get_pixel_size(self, pixel_size):
        pass

    # returns the unscaled image/tile at bottom left (x,y) with W and H.
    # These are the dimensions of the WSI rather than the
    # target dimensions that models are trained on.
    @abstractmethod
    def get_img(self, x, y, w, h, bound=True):
        pass

    # Determine which slide reader library can deal with the file type and
    # return the appropriate class instantiation
    @staticmethod
    def new(slide_path, lvl_x):
        file_type = "." + os.path.split(slide_path)[1].split(".")[-1]
        if file_type in __LIBVIPS__:
            print("libvips format")
            return Libvips(slide_path, file_type, lvl_x)
        elif file_type in __OPENSLIDE__:
            print("Openslide format")
            return OpenSlideFile(slide_path, file_type, lvl_x)
        elif file_type in __BIOFORMATS__:
            print("Bioformat format")
            javabridge.start_vm(class_path=bioformats.JARS)
            return BioFormatsFile(slide_path, file_type, lvl_x)
        else:
            raise Exception(
                f'File format "{file_type}" not '
                f"understood by openslide, bioformats, or libvips."
            )


class BioFormatsFile(Reader):
    def __init__(self, slide_path, file_type, lvl_x):
        super().__init__(slide_path, lvl_x)
        self.file_type = file_type

    @property
    def reader(self):
        return bioformats.ImageReader(self.slide_path)

    @property
    def max_slide_width(self):
        return self.reader.rdr.getSizeX()

    @property
    def max_slide_height(self):
        return self.reader.rdr.getSizeY()

    def get_pixel_size(self, pixel_size):
        """Try to get the pixel size dimensions out of the metadata"""
        if not pixel_size:
            print(
                "WARNING: can't generate pixel size from bioformats files. "
                "Please input correct pixel size"
            )
        else:
            print(f"using input {pixel_size} pixel size")
        return pixel_size

    def get_img(self, x, y, w, h, bound=True):
        full_img = np.zeros((h, w, 3))
        bounded_w = w
        bounded_h = h
        x = min(self.max_slide_width - 1, x)
        y = min(self.max_slide_height - 1, y)
        if bound:
            bounded_w = min(w, self.max_slide_width - x)
            bounded_h = min(h, self.max_slide_height - y)

        # Note: Weirdness with axis flipping in image (x,y)
        # coords will be affected by this
        # W and H flipped in np array for image
        avail_img = (
            self.reader.read(z=self.lvl_x, XYWH=(x, y, bounded_w, bounded_h)) * 255
        )

        np.add(
            full_img[:bounded_h, :bounded_w],
            avail_img,
            out=full_img[:bounded_h, :bounded_w],
        )
        return full_img

    def stop_vm(self):
        javabridge.kill_vm()


class OpenSlideFile(Reader):
    def __init__(self, slide_path, file_type, lvl_x):
        super().__init__(slide_path, lvl_x)
        self.file_type = file_type

    @property
    def reader(self):
        return openslide.OpenSlide(self.slide_path)

    @property
    def max_slide_width(self):
        return self.reader.dimensions[0]

    @property
    def max_slide_height(self):
        return self.reader.dimensions[1]

    def get_pixel_size(self, pixel_size):
        """Try to get the pixel size dimensions out of the metadata"""
        pix_x_size = False
        try:
            pix_x_size = float(self.reader.properties["openslide.mpp-x"])
            pix_y_size = float(self.reader.properties["openslide.mpp-y"])
        except:
            pass
        if pix_x_size:
            assert pix_x_size == pix_y_size
            return pix_y_size
        else:
            return pixel_size

    def get_img(self, x, y, w, h, bound=True):
        full_img = np.zeros((h, w, 3))
        bounded_w = w
        bounded_h = h
        x = int(min(self.max_slide_width - 1, x))
        y = int(min(self.max_slide_height - 1, y))
        if bound:
            bounded_w = min(w, self.max_slide_width - x)
            bounded_h = min(h, self.max_slide_height - y)

        # Note: Weirdness with axis flipping in image (x,y)
        # coords will be affected by this
        #  W and H flipped in np array for image
        avail_img = self.reader.read_region(
            location=(x, y), level=self.lvl_x, size=(bounded_w, bounded_h)
        )
        avail_img = np.array(avail_img)[:, :, 0:3]

        np.add(
            full_img[:bounded_h, :bounded_w],
            avail_img,
            out=full_img[:bounded_h, :bounded_w],
        )
        return full_img


class Libvips(Reader):
    def __init__(self, slide_path, file_type, lvl_x):
        super().__init__(slide_path, lvl_x)
        self.file_type = file_type

    @property
    def reader(self):
        if self.file_type == ".scn":
            return pyvips.Image.new_from_file(
                self.slide_path, access="sequential", autocrop=True
            )
        else:
            return pyvips.Image.new_from_file(self.slide_path, access="sequential")

    @property
    def max_slide_width(self):
        return self.reader.get("width")

    @property
    def max_slide_height(self):
        return self.reader.get("height")

    @property
    def region(self):
        return pyvips.Region.new(self.reader)

    def get_pixel_size(self, pixel_size):
        """Try to get the pixel size dimensions out of the metadata"""
        try:
            pix_x_size = float(self.reader.get("openslide.mpp-x"))
            pix_y_size = float(self.reader.get("openslide.mpp-y"))
            assert round(pix_x_size, 4) == round(pix_y_size, 4)
            print(f"Pixel size {pix_x_size} found and extracted from slide metadata")
            return pix_y_size
        except:
            print(f"Slide Pixel size not found, using input pixel size {pixel_size}")
            return pixel_size

    def get_img(self, x, y, w, h, bound=True):
        bounded_w = w
        bounded_h = h
        x = int(min(self.max_slide_width - 1, x))
        y = int(min(self.max_slide_height - 1, y))
        if bound:
            bounded_w = min(w, self.max_slide_width - x)
            bounded_h = min(h, self.max_slide_height - y)

        avail_img = self.region.fetch(x, y, bounded_w, bounded_h)

        full_img = np.ndarray(
            buffer=avail_img,
            dtype=np.uint8,
            shape=[bounded_h, bounded_w, self.reader.bands],
        )[:, :, 0:3]

        return full_img


__BIOFORMATS__ = {
    ".sld",
    ".aim",
    ".al3d",
    ".gel",
    ".am",
    ".amiramesh",
    ".grey",
    ".hx",
    ".labels",
    ".cif",
    ".img",
    ".hdr",
    ".sif",
    ".png",
    ".afi",
    ".svs",
    ".htd",
    ".pnl",
    ".avi",
    ".arf",
    ".exp",
    ".spc",
    ".sdt",
    ".1sc",
    ".pic",
    ".raw",
    ".scn",
    ".ims",
    ".cr2",
    ".crw",
    ".ch5",
    ".c01",
    ".dib",
    ".vsi",
    ".xml",
    ".wpi",
    ".dv",
    ".r3d",
    ".dcm",
    ".dicom",
    ".v",
    ".eps",
    ".epsi",
    ".ps",
    ".flex",
    ".mea",
    ".res",
    ".fits",
    ".dm3",
    ".dm4",
    ".dm2",
    ".vff",
    ".gif",
    ".naf",
    ".his",
    ".ndpi",
    ".ndpis",
    ".vms",
    ".txt",
    ".bmp",
    ".jpg",
    ".i2i",
    ".ics",
    ".ids",
    ".fff",
    ".seq",
    ".ipw",
    ".hed",
    ".mod",
    ".liff",
    ".obf",
    ".msr",
    ".xdce",
    ".frm",
    ".inr",
    ".tif",
    ".tiff",
    ".ipl",
    ".ipm",
    ".dat",
    ".par",
    ".jp2",
    ".jpk",
    ".jpx",
    ".klb",
    ".xv",
    ".bip",
    ".fli",
    ".lei",
    ".lif",
    ".sxm",
    ".l2d",
    ".lim",
    ".stk",
    ".nd",
    ".mnc",
    ".mrw",
    ".mng",
    ".stp",
    ".mrc",
    ".st",
    ".ali",
    " .map",
    " .rec",
    ".mrcs",
    ".nef",
    ".nii",
    ".nii.gz",
    ".nd2",
    ".nrrd",
    ".nhdr",
    ".apl",
    ".mtb",
    ".tnb",
    ".obsep",
    ".oib",
    ".oif",
    ".oir",
    ".ome.tiff",
    ".ome.tif",
    ".ome.tf2",
    ".ome.tf8",
    ".ome.btf",
    ".ome",
    ".ome.xml",
    ".top",
    ".pcoraw",
    ".pcx",
    ".pds",
    ".csv",
    ".im3",
    "etc.",
    ".qptiff",
    ".pbm",
    ".pgm",
    ".ppm",
    ".psd",
    ".bin",
    ".pict",
    ".cfg",
    ".spe",
    ".afm",
    ".mov",
    ".sm2",
    ".sm3",
    ".xqd",
    ".xqf",
    ".cxd",
    ".spi",
    ".tga",
    ".tf2",
    ".tf8",
    ".btf",
    ".vws",
    ".tfr",
    ".ffr",
    ".zfr",
    ".zfp",
    ".2fl",
    ".pr3",
    ".fdf",
    ".hdf",
    ".dti",
    ".xys",
    ".html",
    ".mvd2",
    ".acff",
    ".wat",
    ".wlz",
    ".lms",
    ".zvi",
    ".czi",
    ".lsm",
    ".mdb",
}

__OPENSLIDE__ = {
    ".svs",
    ".mrxs",
    ".tif",
    ".vms",
    ".vmu",
    ".ndpi",
    ".scn",
    ".tiff",
    ".svslide",
    ".bif",
}

__LIBVIPS__ = {
    ".vips",
    ".svs",
    ".mrxs",
    ".tif",
    ".vms",
    ".vmu",
    ".ndpi",
    ".scn",
    ".tiff",
    ".svslide",
    ".bif",
}
