"""
Exports the bounding boxes of VAL_REGION / TEST_REGION rectangle annotations as a CSV
of x,y,width,height for HAPPY tissue training (--val-patch-files / --test-patch-files).

Run once for validation and once for test:
 - validation: regionClass = "VAL_REGION", fileName ending "_val_patches.csv"
 - test:       regionClass = "TEST_REGION", fileName ending "_test_patches.csv"

Edit regionClass, fileName and saveDir below. The output goes in the project's
graph_splits/ directory, which is where tissue_train.py looks for these files.
"""
import static qupath.lib.gui.scripting.QPEx.*

// EDIT: "VAL_REGION" for validation, "TEST_REGION" for test
def regionClass = "VAL_REGION"

def slideName = getCurrentServer().getPath().split("/")[-1]
def slideNumber = slideName.split("-")[0]   // adjust the split index to your slide naming

// EDIT: use "_test_patches.csv" when regionClass is "TEST_REGION"
def fileName = slideNumber + "_val_patches.csv"
// EDIT: absolute path to your project's graph_splits/ directory
def saveDir = "/../projects/placenta/graph_splits/" + fileName

// Get all rectangle annotations of the chosen region class
def boxes = getAnnotationObjects().findAll({ it.getPathClass().getName() == regionClass })

def FILE_HEADER = 'x,y,width,height'

try (PrintWriter writer = new PrintWriter(new File(saveDir))) {
    StringBuilder sb = new StringBuilder()
    sb.append(FILE_HEADER)
    sb.append('\n')

    // one row per region (supports multiple val/test rectangles)
    for (box in boxes) {
        def roi = box.getROI()
        sb.append(String.join(',',
            ((int) roi.getBoundsX()).toString(),
            ((int) roi.getBoundsY()).toString(),
            ((int) roi.getBoundsWidth()).toString(),
            ((int) roi.getBoundsHeight()).toString()))
        sb.append('\n')
    }

    print(sb.toString())
    writer.write(sb.toString())
    print("Saved " + boxes.size() + " " + regionClass + " region(s) to " + saveDir)

} catch (FileNotFoundException e) {
    print(e.getMessage())
}
