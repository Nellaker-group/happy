"""
Gets local coordinates of points in "TAnnot" boxes and saves them and the boxes as csvs
"""
import static qupath.lib.gui.scripting.QPEx.*

def classNames = ["CYT", "HOF", "SYN", "FIB", "VEN", "VMY", "MAT", "WBC", "MES", "EVT", "KNT", "EPI", "MAC"] as String[]

def slideName = getCurrentServer().getPath().split("/")[-1]
def slideNumber = slideName.split("-")[0]

def fileName = slideNumber + "_from_groovy.csv"
def saveDir = "/../projects/placenta/results/annotation_csvs/" + fileName

// Get all manually annotated box areas
def allBoxAnnot = getAnnotationObjects().findAll({ it.getPathClass().getName() == "TAnnot" })

// Get upper left coord of box and width and height
def xs = allBoxAnnot.collect({ (int) it.getROI().getBoundsX() })
def ys = allBoxAnnot.collect({ (int) it.getROI().getBoundsY() })
def widths = allBoxAnnot.collect({ (int) it.getROI().getBoundsWidth() })
def heights = allBoxAnnot.collect({ (int) it.getROI().getBoundsHeight() })

// Get all cell class points
def points = getAnnotationObjects().findAll({
    it.getPathClass().getName() != "TAnnot"
            && it.getPathClass().getName() != "FAnnot" && it.getPathClass().getName() != "Discuss"
})
// Remove classes without any points
nullClasses = []
for (className in classNames) {
    if (points.find({ it.getPathClass().getName() == className }) == null) {
        nullClasses.add(className)
    }
}
classNames -= nullClasses

def getRelativeCoords(cellClassArray, x, y, width, height) {
    // Get the global coordinates of cell classes in the box
    pointsInBox = cellClassArray.getROI().getAllPoints().findAll({
        it.getX() >= x && it.getX() <= (x + width) && it.getY() >= y && it.getY() <= (y + height)
    })
    // Convert these to the coordinates relative to the box
    return pointsInBox.collect({
        [(int) (it.getX() - x), (int) (it.getY() - y)]
    })
};

def getPointsInBoxes(classNames, points, xs, ys, widths, heights) {
    def pointsDict = [:]
    def pointsInBoxDict = [:]
    for (className in classNames) {
        classPoints = points.find({ it.getPathClass().getName() == className })
        if (classPoints != null) {
            pointsDict[className] = classPoints
            pointsInBoxDict[className] = []

            // Loop through each annotation box and extract the relative coordinates of each cell class in the box
            for (int i = 0; i < xs.size(); i++) {
                // Append such that the box indexes and coordinate indexes match
                pointsInBoxDict[className] << getRelativeCoords(pointsDict[className], xs[i], ys[i], widths[0], heights[0])
            };
        }
    }
    return pointsInBoxDict
}

def pointsInBoxes = getPointsInBoxes(classNames, points, xs, ys, widths, heights)


def buildRows(sb, boxi, cellArray, x, y, cellName) {
    for (int pointi = 0; pointi < cellArray[boxi].size(); pointi++) {
        sb.append(String.join(',', x.toString(), y.toString(),
                cellArray[boxi][pointi][0].toString(),
                cellArray[boxi][pointi][1].toString(), cellName))
        sb.append('\n')
    }
};

// Save these to a file with columns boxx, boxy, pointx, pointy, class
def FILE_HEADER = 'bx,by,px,py,class'

// Write to csv
try (PrintWriter writer = new PrintWriter(new File(saveDir))) {
    StringBuilder sb = new StringBuilder();
    sb.append(FILE_HEADER)
    sb.append('\n')

    // For each box, write rows for each point
    for (className in classNames) {
        for (int boxi = 0; boxi < xs.size(); boxi++) {
            buildRows(sb, boxi, pointsInBoxes[(className)], xs[boxi], ys[boxi], className)
        }
    }

    print(sb.toString())

    writer.write(sb.toString());
    print("done!")

} catch (FileNotFoundException e) {
    print(e.getMessage())
}
