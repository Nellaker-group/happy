import numpy as np
from PIL import Image
from tqdm import tqdm

import happy.db.eval_runs_interface as db


class MicroscopeFile:
    """In memory class representing a run over a whole slide image.

    This specifies how to read the slide file, how to tile the slide with or without
    overlap, how to scale between the pixel size of the slide and a target pixel
    size that a model has been trained with, whether to run over subsections of
    the slide, and how much of the slide's nuclei and cell predictions are done.

    Args:
        id: the evalrun id from the database
        reader: a Reader object which knows which slide reading library to use
        width: the desired width of tiles for nuclei detection
        height: the desired height of tiles for nuclei detection
        target_pixel_size: the pixel size that the model was trained with
        slide_pixel_size: the pixel size of the slide
        overlap: the amount of overlap between tiles
        subsect_x: the x coordinate of the top left corner of the subsection
        subsect_y: the y coordinate of the top left corner of the subsection
        subsect_h: the height of the subsection
        subsect_w: the width of the subsection
        nucs_done: whether nuclei detection has been done for this slide
        cells_done: whether cell classification has been done for this slide
    """

    def __init__(
        self,
        id,
        reader,
        width,
        height,
        target_pixel_size,
        slide_pixel_size,
        overlap,
        subsect_x,
        subsect_y,
        subsect_h,
        subsect_w,
        nucs_done,
        cells_done,
    ):
        self.id = id
        self.reader = reader
        self.target_tile_width = width
        self.target_tile_height = height
        self.target_pixel_size = target_pixel_size
        self.overlap = overlap
        self.subsect_x = subsect_x
        self.subsect_y = subsect_y
        self.subsect_h = subsect_h
        self.subsect_w = subsect_w
        self.nucs_done = nucs_done
        self.cells_done = cells_done

        self.slide_pixel_size = self.reader.get_pixel_size(slide_pixel_size)
        self.rescale_ratio = self._get_rescale_ratio()
        self.max_slide_width = self.reader.max_slide_width
        self.max_slide_height = self.reader.max_slide_height

        # STATE
        if nucs_done and cells_done:
            print("This evaluation run has been completed. Nuclei and cells are done")

        if not nucs_done:
            if db.run_state_exists(id):
                print("getting tile coordinates from db")
                self.tile_xy_list = db.get_run_state(id)
            else:
                print("generating tile coordinates")
                self.tile_xy_list = self._get_tile_xys(
                    int(self.target_tile_width * self.rescale_ratio),
                    int(self.target_tile_height * self.rescale_ratio),
                    int(self.overlap * self.rescale_ratio),
                )
                db.save_new_tile_state(id, self.tile_xy_list)

    # number of tiles in file
    def num_tiles(self):
        if self.tile_xy_list:
            return len(self.tile_xy_list)
        else:
            _, _, max_x, max_y = self._subsect_xywh()
            overlap = int(self.overlap * self.rescale_ratio)
            w = int(self.target_tile_width * self.rescale_ratio)
            h = int(self.target_tile_height * self.rescale_ratio)
            num_rows, num_columns = self._get_num_rows_cols(w, h, max_x, max_y, overlap)
            return num_rows * num_columns

    def mark_finished_nuclei(self):
        self.nucs_done = True
        db.mark_nuclei_as_done(self.id)

    def mark_finished_cells(self):
        self.cells_done = True
        db.mark_cells_as_done(self.id)

    # Returns rescaled image at (x,y) coords with specified width and height
    def get_tile_by_coords(self, x, y, w, h):
        return self._get_rescaled_img(
            x, y, w * self.rescale_ratio, h * self.rescale_ratio, w, h
        )

    # Returns rescaled image with cell (x,y) centre coords with specified width and height
    def get_cell_tile_by_cell_coords(self, cell_x, cell_y, target_w, target_h):
        w = int(target_w * self.rescale_ratio)
        h = int(target_h * self.rescale_ratio)

        tile_x = int(cell_x - (w / 2))
        tile_y = int(cell_y - (h / 2))

        return self._get_rescaled_img(tile_x, tile_y, w, h, target_w, target_h)

    # calculates the correct scaling ratio based on pixel size and the pixel size models expect
    def _get_rescale_ratio(self):
        if self.slide_pixel_size and self.target_pixel_size:
            scale_ratio = self.target_pixel_size / self.slide_pixel_size
            if scale_ratio <= 0.3:
                print(
                    "WARNING: Very high scaling necessary. "
                    "Quality of tiles passed to models may be low"
                )
            return scale_ratio
        else:
            return 1

    # returns img after being correctly scaled back to original target dimensions
    def _get_rescaled_img(self, x, y, w, h, target_w, target_h):
        orig_img = self.reader.get_img(x, y, int(w), int(h))

        pil_orig_image = Image.fromarray(orig_img.astype("uint8"))
        rescaled = pil_orig_image.resize([target_w, target_h])
        return np.asarray(rescaled)

    # Gets all (x,y) top left coords for all tiles in WSI with dimensions and overlap
    def _get_tile_xys(self, w, h, overlap):
        min_x, min_y, max_x, max_y = self._subsect_xywh()

        num_rows, num_columns = self._get_num_rows_cols(w, h, max_x, max_y, overlap)
        print(f"rows: {num_rows}, columns: {num_columns}")

        xy_list = []
        for row in tqdm(range(num_rows)):
            if row == 0:
                x = min_x + row * w
            else:
                x = min_x + (row * (w - overlap))
            xy_list.extend([(x, i) for i in range(min_y, max_y + min_y, h - overlap)])

        return xy_list

    # calculates number of rows and colums given sizes and overlap
    def _get_num_rows_cols(self, w, h, max_x, max_y, overlap):
        num_rows = int(max_x / (w - overlap))
        num_columns = int(max_y / (h - overlap))
        if max_x - (num_rows * w) > 0:
            num_rows += 1
        if max_y - (num_columns * h) > 0:
            num_columns += 1
        return num_rows, num_columns

    # helper function for picking subsects
    def _subsect_xywh(self):
        if self.subsect_x:
            print("Using subsection")
            try:
                assert self.subsect_y
                assert self.subsect_w
                assert self.subsect_h
            except AssertionError:
                raise (
                    f"Subsection x is defined, but y={self.subsect_y}, "
                    f"width={self.subsect_w} or height={self.subsect_h} missing"
                )
            return self.subsect_x, self.subsect_y, self.subsect_w, self.subsect_h
        else:
            return 0, 0, self.max_slide_width, self.max_slide_height

    # Rough threshold for empty sets of pixels. Either all white, all black, or grey.
    def _img_is_empty(self, img):
        avg_rgb = np.mean(img, axis=(0, 1))
        # img is white
        if np.all(avg_rgb > 245):
            return True
        # img is black
        if np.all(avg_rgb < 10):
            return True
        # check if img is grey
        sorted_flat_img = np.sort(img, axis=None)
        portion_of_size = int(sorted_flat_img.size / 10)
        ratio_darkest_brightest = np.mean(sorted_flat_img[:portion_of_size]) / np.mean(
            sorted_flat_img[-portion_of_size:]
        )
        # img is grey
        return True if ratio_darkest_brightest > 0.95 else False
