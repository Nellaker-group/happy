import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from histolab.slide import Slide
from histolab.masks import TissueMask

import happy.db.eval_runs_interface as db
import sys


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
        full_slide_path
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
        self.mask_array = self.get_mask(full_slide_path)  #mask array need to be pass into MSData set as well!!!

        # STATE
        if nucs_done and cells_done:
            print("This evaluation run has been completed. Nuclei and cells are done")

        if not nucs_done:
            if db.run_state_exists(id):
                print(f"getting tile coordinates from db, run id = {id}")
                self.tile_xy_list = db.get_run_state(id)
            else:
                print("generating tile coordinates")
                # get small tile xys from the mask array
                self.tile_xy_list = self.get_nuc_small_tile_xy (self.mask_array,self.overlap,self.target_tile_width,self.target_tile_height) 
                print(f'number of tile generated {len(self.tile_xy_list)}')

                # save new small tile xys to db
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

    # Returns rescaled image at (x,y) coords with specified width and height (nuclei tile)
    def get_tile_by_coords(self, x, y, w, h):
        return self._get_rescaled_img(
            x, y, w * self.rescale_ratio, h * self.rescale_ratio, w, h
        )

    # Returns rescaled image with cell (x,y) centre coords with specified width and height
    def get_cell_tile_by_cell_coords(self, cell_x, cell_y, target_w, target_h):
        w = int(target_w * self.rescale_ratio*200/224) #modification made for target_w and target_h being 224
        h = int(target_h * self.rescale_ratio*200/224)

        tile_x = int(cell_x - (w / 2))
        tile_y = int(cell_y - (h / 2))

        return self._get_rescaled_img(tile_x, tile_y, w, h, target_w, target_h)

        # Returns rescaled image with top left coords of the big tile with specified width and height (NUCLEI INFERENCE)
    def get_big_tile_by_coords_nuc(self, bt_x, bt_y, big_tile_size):
        w = int(big_tile_size * self.rescale_ratio) 
        h = w

        tile_x = int(bt_x)
        tile_y = int(bt_y)

        return self._get_rescaled_img(tile_x, tile_y, w, h, big_tile_size, big_tile_size)

    # Returns rescaled image with top left coords of the big tile with specified width and height (CELL INFERENCE)
    def get_big_tile_by_coords_cell(self, bt_x, bt_y, big_tile_size):
        w = int(big_tile_size * self.rescale_ratio*200/224) # modification made for target_w and target_h being 224
        h = w

        tile_x = int(bt_x)
        tile_y = int(bt_y)

        return self._get_rescaled_img(tile_x, tile_y, w, h, big_tile_size, big_tile_size)

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

    def get_big_tile_xy (self,coord,tile_tilesize,tile_overlap=224):

        """
        Locate and save the top left x,y coordinates of the big tiles

        Parameters:
        -coord: WSI nuclei coordinates from the nuclei inference step
        -tile_tilesize: big tile size (same height and weight) AFTER RESIZED 
        -tile_overlap : px overlap between 2 big tiles,  AFTER RESIZED 

        Returns:
        A list of the top left x,y coordinates of the big tiles 

        """
        coord=np.array(coord)
        df = pd.DataFrame(coord.tolist())
        assert len(df)>0, f"All cells are evaluated. To re-run all cells, add [re_run_all_cell] flag to your cell_inference.py arguments."

        #rescale
        rescale_ratio = self.rescale_ratio*200/224 #cell image 224 crop specific rescale ratio 
        tilesize=tile_tilesize* rescale_ratio #actual tile size (WSI)
        overlap=tile_overlap* rescale_ratio #actual overlap size (WSI)

        min_y = min(df['y'])-overlap
        max_y = max(df['y'])+overlap # +/- (overlap/2) to allow cell image on the edge to be cropped out

        #calculate how many rows of tile 
        num_tile_row = int((max_y-min_y)/(tilesize-overlap))
        #plus one tile if did not fully covered 
        if (max_y-min_y) - (num_tile_row* (tilesize-overlap)+overlap )> 0:
            num_tile_row = num_tile_row + 1

        #start to extract     
        point_saver=[]
        for i in range(0,num_tile_row):
            y_start = min_y + i*(tilesize-overlap)
            y_end = y_start + tilesize 
            x_start = min(df['x'].loc[(df['y']>=y_start) & (df['y']<=y_end)])-overlap
            x_end = max(df['x'].loc[(df['y']>=y_start) & (df['y']<=y_end)])+overlap

            #calculate how many tiles needed in one particular row
            num_tile_column = int((x_end - x_start)/(tilesize-overlap))
            #plus one tile if did not fully covered 
            if (x_end-x_start) - (num_tile_column* (tilesize-overlap)+overlap )> 0:
                num_tile_column = num_tile_column + 1

            for j in range(0,num_tile_column) :
                x=int(x_start+j*(tilesize-overlap))
                y=int(y_start)
                point_saver.append((x,y))
            
        return (point_saver)


    def get_xys_in_big_tile (self, tile_xy, coord,tile_tilesize, tile_overlap=224):

        """
        Locate and save the top left x,y coordinates of the big tiles

        Parameters:
        -tile_xy: single big tile top left xy
        -rescale_ratio: rescale ration defined in class
        -coord: WSI nuclei coordinates from the nuclei inference step
        -tile_tilesize: big tile size (same height and weight) AFTER RESIZED 
        -tile_overlap : px overlap between 2 big tiles,  AFTER RESIZED 

        Returns:
        A array of the centriod x,y coordinates of all the cell within a certain big tile (in relation to WSI)

        """   

        #rescale
        rescale_ratio = self.rescale_ratio*200/224 # This rescale ratio map big tile back to WSI
        tilesize=tile_tilesize* rescale_ratio #actual tile size (WSI)
        overlap=tile_overlap* rescale_ratio #actual overlap size (WSI)
        
        # define big tile boundary 
        x_min = tile_xy [0]
        x_min_pad = tile_xy [0]+ (overlap/2) # cell within padding area (overlap/2) would be exclude
        x_max = x_min + tilesize
        x_max_pad = x_max - (overlap/2)

        y_min = tile_xy [1]
        y_min_pad = tile_xy [1]+ (overlap/2)
        y_max = y_min + tilesize 
        y_max_pad = y_max - (overlap/2)

        # converting all cell coordinated to a numpy array
        array=np.array([(point['x'], point['y']) for point in coord], dtype=[('x', int), ('y', int)])

        #select all nuclei xy within bounds
        extract_xy_wsi = array[(array['x'] >= x_min_pad) & (array['x'] <= x_max_pad) & 
                        (array['y'] >= y_min_pad) & (array['y'] <= y_max_pad)]


        return (extract_xy_wsi)

    #converting wsi coordinate to big tile coordinate for cell inference resizeing (for ResNet 224px input)
    def wsi_to_tile_cell(self, big_tile_x, wsi_x):
        rescale_ratio = self.rescale_ratio*200/224 # rescale ratio map big tile back to WSI
        convert_ratio= 1/rescale_ratio # rescale ratio map WSI to big tile
        tile_x=int((wsi_x-big_tile_x)*convert_ratio)
        return(tile_x)
    
        #converting wsi coordinate to big tile coordinate for nuclei inference resizeing
    def wsi_to_tile_nuclei(self, big_tile_x, wsi_x):
        rescale_ratio = self.rescale_ratio 
        convert_ratio= 1/rescale_ratio 
        tile_x=int((wsi_x-big_tile_x)*convert_ratio)
        return(tile_x)


    # calculates number of rows and colums given sizes and overlap
    def _get_num_rows_cols(self, w, h, max_x, max_y, overlap):
        num_rows = int(max_x / (w - overlap))
        num_columns = int(max_y / (h - overlap))
        if max_x - (num_rows * w) > 0: #should be max_x - (num_rows * (w-overlap)+overlap) > 0 to be precise, but with small tiles it doesn't make big differences
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
        # Changed to match Odense liver biopsy pixels
        # TODO: have the thresholds as an input parameter
        empty_flag = False
        avg_rgb = np.mean(img, axis=(0, 1))
        # img is white
        if np.all(avg_rgb > 220):
            empty_flag = True
        # img is black
        if np.all(avg_rgb < 10):
            empty_flag = True
        # check if img is grey
        sorted_flat_img = np.sort(img, axis=None)
        portion_of_size = int(sorted_flat_img.size / 10)
        ratio_darkest_brightest = np.mean(sorted_flat_img[:portion_of_size]) / np.mean(
            sorted_flat_img[-portion_of_size:]
        )
        # img is grey
        if ratio_darkest_brightest > 0.90:
            empty_flag = True

        return empty_flag
    
    #with slide id, generate an array representing the tissue mask (tissue location) from thumbnail matching the WSI coordinate
    def get_mask(self,slide_path):
        # slide_path = db.get_slide_path_by_id(slide_id)
        testslide = Slide(slide_path,processed_path='projects/liver/histolab_test') #!! hard coded path need to be improve. 'process_path' is a must!
        wsi_w=testslide.dimensions[0]
        wsi_h=testslide.dimensions[1]
        tissue_mask = TissueMask()
        mask_image = tissue_mask(testslide)
        tn_w=mask_image.shape[1] #np array shape calls number of rows (y) first then number of elements in each ro w
        tn_h=mask_image.shape[0]
        rescale_x = int(wsi_w/tn_w)
        rescale_y = int(wsi_h/tn_h)

        assert rescale_x == rescale_y, 'x and y rescalling factors do not match'

        mask_array = np.array(list(zip(*np.where(mask_image==True))))*rescale_x
        return (mask_array)

    #with masking, check whether a nuclei tile is empty
    def check_nuc_tile_blank(self,nuc_xy,mask_array):
    
        mask_df = pd.DataFrame(mask_array.tolist(),columns=['y','x'])
        # nuc_xy is with WSI reference frame!!!!!!!!

        x_min=nuc_xy[0]
        y_min=nuc_xy[1]
        rescale_ratio = self.rescale_ratio
        w=self.target_tile_width* rescale_ratio
        h=self.target_tile_height* rescale_ratio

        x_max=x_min+w
        y_max=y_min+h

        check_df= mask_df[mask_df.y.between(left=y_min,right=y_max) &
                        mask_df.x.between(left=x_min,right=x_max)]

        empty_flag = False
        if len(check_df) == 0:
            empty_flag =True
        
        return empty_flag
    
    # first get all small nuc tile locations, treating them as cell xy locations (results then saved into the main.db)
    def get_nuc_small_tile_xy (self, mask_array,tile_overlap,tile_w,tile_h):

        df = pd.DataFrame(mask_array.tolist(),columns=['y','x'])

        #rescale
        rescale_ratio = self.rescale_ratio
        overlap=tile_overlap* rescale_ratio #actual overlap size (WSI)
        w=tile_w* rescale_ratio
        h=tile_h* rescale_ratio
        min_y = min(df['y'])-overlap
        max_y = max(df['y'])+overlap
        

        #calculate how many rows of tile 
        num_tile_row = int((max_y-min_y)/(h-overlap))
        #plus one tile if did not fully covered 
        if (max_y-min_y) - (num_tile_row* (h-overlap)+overlap )> 0:
            num_tile_row = num_tile_row + 1

        #start to extract     
        point_saver=[]
        for i in range(0,num_tile_row):
            y_start = min_y + i*(h-overlap)
            y_end = y_start + h
            x_start = min(df['x'].loc[(df['y']>=y_start) & (df['y']<=y_end)])-overlap
            x_end = max(df['x'].loc[(df['y']>=y_start) & (df['y']<=y_end)])+overlap

            #calculate how many tiles needed in one particular row
            num_tile_column = int((x_end - x_start)/(w-overlap))
            #plus one tile if did not fully covered 
            if (x_end-x_start) - (num_tile_column* (w-overlap)+overlap )> 0:
                num_tile_column = num_tile_column + 1

            for j in range(0,num_tile_column) :
                x=int(x_start+j*(w-overlap))
                y=int(y_start)
                point_saver.append((x,y))
        point_saver = [(max(0, x), max(0, y)) for x, y in point_saver] 
        return (point_saver)

    # helper function to stop while loop successfully for get_nuc_big_tile_xys
    def _safe_min(self, dataframe, column):
        """Safely get the minimum value from a DataFrame column."""
        if dataframe.empty:
            return None
        return dataframe[column].min()

    # with small nuc tile xy, get big tile xy
    def get_nuc_big_tile_xys(self, small_tile_xys, tile_tilesize,tile_w,tile_h):

        df = pd.DataFrame(small_tile_xys.tolist())
        df.columns= ['index','x','y']
        #rescaling
        rescale_ratio = self.rescale_ratio
        tilesize=tile_tilesize* rescale_ratio #actual tile size (WSI)
        w=tile_w* rescale_ratio
        h=tile_h* rescale_ratio

        min_y = min(df['y'])
        max_y = max(df['y'])
        
        num_tile_row=int((max_y + h - min_y)/(tilesize-h))
        if (max_y-min_y) - (num_tile_row* (tilesize-h)+h )> 0:
            num_tile_row = num_tile_row + 1
        #print(f'num_tile_row={num_tile_row}')

        big_tile_list=[]
        while not df.empty:
            for i in range (num_tile_row):
                #print(f'start of row {i}')
                miny=self._safe_min(df,'y')
                if miny is None:
                    print("DataFrame is empty!")
                    break
                y1= min(df['y']) + i*(tilesize - h)
                y2= y1 + tilesize - h
                row_df= df.loc[(df['y'] >= y1) & (df['y'] <= y2)]
                #print(f'Row {i} has {len(row_df)} small tiles in total, with range {y1} to {y2}')
                df = df.merge(row_df, indicator=True, how='left')
                df = df[df['_merge'] == 'left_only'].drop('_merge', axis=1)
                while not row_df.empty:
                    x1=min(row_df['x'])
                    x2= x1 + tilesize - w 
                    x1=int(x1)
                    y1=int(y1)
                    big_tile_list.append((x1,y1))
                    #print('big tile xy saved')
                    row_df=row_df.loc[row_df['x']>x2]
                    row_df = row_df.reset_index(drop=True)


                #print(f"Remaining tiles: {len(df)}")
        return big_tile_list


    def small_tile_xy_big_tile(self,small_tile_xys,big_tile_xy,tile_tilesize,small_w,small_h):

        #rescaling
        rescale_ratio = self.rescale_ratio
        tilesize=tile_tilesize* rescale_ratio #actual tile size (WSI)
        w=small_w* rescale_ratio
        h=small_h* rescale_ratio



        big_x_min=big_tile_xy[0]
        big_y_min=big_tile_xy[1]

        big_x_max=big_x_min+tilesize-w
        big_y_max=big_y_min+tilesize-h

        # small tile data from main.db also comes with the index, which is required for subseqeunce steps
        df = pd.DataFrame(small_tile_xys.tolist())
        df.columns= ['index','x','y']

        filtered_df = df[(df['x'] >= big_x_min) &
                          (df['x'] <= big_x_max) & 
                          (df['y'] >= big_y_min) & 
                          (df['y'] <= big_y_max)]


        return filtered_df